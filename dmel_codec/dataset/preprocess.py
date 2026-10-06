import logging
import sys
import os
from lhotse import CutSet
from lhotse import RecordingSet, SupervisionSet, CutSet, MonoCut, Recording
from lightning import LightningDataModule
from dmel_codec.utils.utils import open_filelist
from dmel_codec.utils.logger import RankedLogger
from time import time
from tqdm import tqdm
from lhotse.supervision import SupervisionSegment
import random

log = RankedLogger(__name__, rank_zero_only=False)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class LhotsePreProcess(LightningDataModule):
    def __init__(
        self,
        # hparams required
        output_dir: str,
        stage: str,
        codec_or_lm: str,

        # recordings, supervisions
        train_recordings_paths: list[str] | None = None,
        train_supervisions_paths: list[str] | None = None,
        train_recordings_filelist: list[str] | None = None,
        train_supervisions_filelist: list[str] | None = None,
        val_recordings_paths: list[str] | None = None,
        val_supervisions_paths: list[str] | None = None,
        val_recordings_filelist: list[str] | None = None,
        val_supervisions_filelist: list[str] | None = None,
        test_recordings_paths: list[str] | None = None,
        test_supervisions_paths: list[str] | None = None,
        test_recordings_filelist: list[str] | None = None,
        test_supervisions_filelist: list[str] | None = None,

        # cuts
        train_cuts_paths: list[str] | None = None,
        val_cuts_paths: list[str] | None = None,
        test_cuts_paths: list[str] | None = None,
        train_cuts_filelist: list[str] | None = None,
        val_cuts_filelist: list[str] | None = None,
        test_cuts_filelist: list[str] | None = None,

        # prefix for recording, Optional
        train_recordings_prefix: list[str] | None = None,
        val_recordings_prefix: list[str] | None = None,
        test_recordings_prefix: list[str] | None = None,

        # prefix for recording filelist, Optional
        train_recordings_filelist_prefix: list[str] | None = None,
        val_recordings_filelist_prefix: list[str] | None = None,
        test_recordings_filelist_prefix: list[str] | None = None,

        # prefix for cuts, Optional
        train_cuts_prefix: list[str] | None = None,
        val_cuts_prefix: list[str] | None = None,
        test_cuts_prefix: list[str] | None = None,

        # prefix for cuts filelist, Optional
        train_cuts_filelist_prefix: list[str] | None = None,
        val_cuts_filelist_prefix: list[str] | None = None,
        test_cuts_filelist_prefix: list[str] | None = None,

        # other hparams
        max_samples: int | None = None,
        sample_rate: int = 24000,
        window_size: int | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        num_jobs: int = 10,
        shuffle_train_cuts: bool = False,
    ):
        """
        output_dir: str
            note: Required, for save cutset

        recordings, supervisions, cuts, filelist:
            note: These four parameters are mutually exclusive, you can only provide one of them
            note: filelist means you provide a txt file, and the txt file contains the absolute path info

        prefix:
            note: all path in one filelist use the same prefix
            note: prefix can be None, if you are absolute path in your audio source

        stage: fit, validate, test
            note: if you are fit stage, you must provide train and val info
            note: if you are validate stage, you must provide val info
            note: if you are test stage, you must provide test info

        max_samples: int = 128
            note: for fit stage, only use max_samples samples to evaluate, speed up train

        window_size: int | None = None
            note: for cutset, how many seconds in one cut, default None, just for train cutset
        num_jobs: int = 10
            note: for train cutset, how many jobs to use in split cutset into windows, default 10

        min_duration: float | None = None
            note: for cutset, min duration, default None, just for train cutset

        max_duration: float | None = None
            note: for cutset, max duration, default None, just for train cutset

        """
        super().__init__()

        self.save_hyperparameters(logger=False) # use LightningDataModule easy save hparams

        self.hparams_check()  # check hparams and load filelist if filelist is not None

    def simplify_cut(self, cut):
        # 保留原始的 AudioSource
        audio_source = cut.recording.sources[0]
        
        # 创建简化版的 Recording, 保留text
        simplified_recording = Recording(
            id=cut.recording.id,
            sources=[audio_source],
            sampling_rate=cut.recording.sampling_rate,
            num_samples=cut.recording.num_samples,
            duration=cut.duration,
            channel_ids=cut.recording.channel_ids
        )

        if self.hparams.codec_or_lm == "codec":
            # 创建简化版的 Cut
            simplified_cut = MonoCut(
                id=cut.id,
                start=cut.start,
                duration=cut.duration,
                channel=cut.channel,
                recording=simplified_recording
            )
        elif self.hparams.codec_or_lm == "lm":
            # 获取原始的 supervision 信息
            supervisions = [            
                SupervisionSegment(    
                    id=cut.id,
                    recording_id=cut.recording.id,
                    start=cut.start,
                    duration=cut.duration,
                    text=cut.supervisions[0].text,
                )]

            # 创建简化版的 Cut
            simplified_cut = MonoCut(
                id=cut.id,
                start=cut.start,
                duration=cut.duration,
                channel=cut.channel,
                recording=simplified_recording,
                supervisions=supervisions
            )

        return simplified_cut
        

    def process_cuts_for_train(self, cuts: CutSet):
        log.info(f"original cuts: {cuts}")
        cuts = cuts.to_eager()
        cuts = cuts.map(self.simplify_cut)
        
        if self.hparams.window_size is not None:
            start_time = time()
            cuts = cuts.cut_into_windows(duration=self.hparams.window_size, num_jobs=self.hparams.num_jobs)
            end_time = time()
            log.info(f"cut_into_windows time: {end_time - start_time}")
        if self.hparams.min_duration is not None:
            start_time = time()
            cuts = cuts.filter(lambda cut: cut.duration >= self.hparams.min_duration)
            end_time = time()
            log.info(f"filter min_duration time: {end_time - start_time}")
        if self.hparams.max_duration is not None:
            start_time = time()
            cuts = cuts.filter(lambda cut: cut.duration <= self.hparams.max_duration)
            end_time = time()
            log.info(f"filter max_duration time: {end_time - start_time}")

        cuts = cuts.to_eager()
        log.info(f"processed cuts: {cuts}")
        return cuts

    def hparams_check(self):
        if self.hparams.stage == "fit":
            self._train_stage_hparams_check()
            self._val_stage_hparams_check()

        if self.hparams.stage == "validate":
            self._val_stage_hparams_check()

        if self.hparams.stage == "test":
            self._test_stage_hparams_check()

    def save_cutset(self):
        # fit == train, need train and val dataset
        if self.hparams.stage == "fit":
            self._save_train_cutset()
            self._save_val_cutset()

        elif self.hparams.stage == "validate":
            self._save_val_cutset()

        elif self.hparams.stage == "test":
            self._save_test_cutset()

    # check train hparams and load train filelist if train filelist is not None
    def _train_stage_hparams_check(self):
        # output_dir
        assert self.hparams.output_dir is not None, "output_dir must be provided"
        if os.path.exists(self.hparams.output_dir):
            log.info(f"output_dir {self.hparams.output_dir} already exists")
        else:
            os.makedirs(self.hparams.output_dir, exist_ok=True)
        
        if self.hparams.window_size is None:
            log.info(f"window_size is None, do not split audio into windows")
        if self.hparams.min_duration is None:
            log.info(f"min_duration is None, do not filter min_duration")
        if self.hparams.max_duration is None:
            log.info(f"max_duration is None, do not filter max_duration")

        # train_recordings_filelist and train_supervisions_filelist and train_cuts_filelist
        if self.hparams.train_recordings_filelist is not None:
            assert len(self.hparams.train_recordings_filelist) == len(
                self.hparams.train_supervisions_filelist
            ), "train_recordings_filelist and train_supervisions_filelist must have the same length"
            if self.hparams.train_recordings_prefix is not None:
                assert len(self.hparams.train_recordings_filelist) == len(
                    self.hparams.train_recordings_prefix
                ), "train_recordings_filelist and train_recordings_prefix must have the same length"
            self.hparams.train_recordings_paths_list = []
            self.hparams.train_supervisions_paths_list = []
            for recordings_path, supervisions_path in zip(
                self.hparams.train_recordings_filelist,
                self.hparams.train_supervisions_filelist,
            ):
                tmp_recordings_list = open_filelist(recordings_path)
                tmp_supervisions_list = open_filelist(supervisions_path)
                assert len(tmp_recordings_list) == len(
                    tmp_supervisions_list
                ), "recordings and supervisisons absolute path in filelist must have the same length"
                self.hparams.train_recordings_paths_list.append(tmp_recordings_list)
                self.hparams.train_supervisions_paths_list.append(tmp_supervisions_list)
        else:
            self.hparams.train_recordings_paths_list = None
            self.hparams.train_supervisions_paths_list = None

        if self.hparams.train_cuts_filelist is not None:
            self.hparams.train_cuts_paths_list = []
            for path in self.hparams.train_cuts_filelist:
                self.hparams.train_cuts_paths_list.append(open_filelist(path))
        else:
            self.hparams.train_cuts_paths_list = None

        # train_recordings_paths and train_supervisions_paths and train_cuts_paths
        if self.hparams.train_recordings_paths is not None:
            assert len(self.hparams.train_recordings_paths) == len(
                self.hparams.train_supervisions_paths
            ), "train_recordings_paths and train_supervisions_paths must have the same length"
            if self.hparams.train_recordings_prefix is not None:
                assert len(self.hparams.train_recordings_paths) == len(
                    self.hparams.train_recordings_prefix
                ), "train_recordings_paths and train_recordings_prefix must have the same length"

        # train_prefix == None
        if (
            self.hparams.train_recordings_prefix is None
            and self.hparams.train_recordings_paths is not None
        ):
            log.info(
                f"train_recordings_prefix is None, pls check your train_recordings source is absolute path"
            )

        if (
            self.hparams.train_cuts_prefix is None
            and self.hparams.train_cuts_paths is not None
        ):
            log.info(
                f"train_cuts_prefix is None, pls check your train_cuts source is absolute path"
            )

        if (
            self.hparams.train_recordings_filelist_prefix is None
            and self.hparams.train_recordings_filelist is not None
        ):
            log.info(
                f"train_recordings_filelist_prefix is None, pls check your train_recordings_filelist source is absolute path"
            )

        if (
            self.hparams.train_cuts_filelist_prefix is None
            and self.hparams.train_cuts_filelist is not None
        ):
            log.info(
                f"train_cuts_filelist_prefix is None, pls check your train_cuts_filelist source is absolute path"
            )

        if (
            self.hparams.train_recordings_paths is None
            and self.hparams.train_cuts_paths is None
            and self.hparams.train_recordings_filelist is None
            and self.hparams.train_cuts_filelist is None
        ):
            raise ValueError(
                "train_recordings_paths, train_cuts_paths, train_recordings_filelist, train_cuts_filelist must be provided at least one"
            )

    # check val hparams and load val filelist if val filelist is not None
    def _val_stage_hparams_check(self):
        assert self.hparams.output_dir is not None, "output_dir must be provided"
        # val_recordings_filelist and val_supervisions_filelist and val_cuts_filelist
        if self.hparams.val_recordings_filelist is not None:
            assert len(self.hparams.val_recordings_filelist) == len(
                self.hparams.val_supervisions_filelist
            ), "val_recordings_filelist and val_supervisions_filelist must have the same length"
            if self.hparams.val_recordings_prefix is not None:
                assert len(self.hparams.val_recordings_filelist) == len(
                    self.hparams.val_recordings_prefix
                ), "val_recordings_filelist and val_recordings_prefix must have the same length"
            self.hparams.val_recordings_paths_list = []
            self.hparams.val_supervisions_paths_list = []
            for recordings_path, supervisions_path in zip(
                self.hparams.val_recordings_filelist,
                self.hparams.val_supervisions_filelist,
            ):
                tmp_recordings_list = open_filelist(recordings_path)
                tmp_supervisions_list = open_filelist(supervisions_path)
                assert len(tmp_recordings_list) == len(
                    tmp_supervisions_list
                ), "recordings and supervisisons absolute path in filelist must have the same length"
                self.hparams.val_recordings_paths_list.append(tmp_recordings_list)
                self.hparams.val_supervisions_paths_list.append(tmp_supervisions_list)
        else:
            self.hparams.val_recordings_paths_list = None
            self.hparams.val_supervisions_paths_list = None

        if self.hparams.val_cuts_filelist is not None:
            self.hparams.val_cuts_paths_list = []
            for path in self.hparams.val_cuts_filelist:
                self.hparams.val_cuts_paths_list.append(open_filelist(path))
        else:
            self.hparams.val_cuts_paths_list = None

        # val_recordings_paths and val_supervisions_paths and val_cuts_paths
        if self.hparams.val_recordings_paths is not None:
            assert len(self.hparams.val_recordings_paths) == len(
                self.hparams.val_supervisions_paths
            ), "val_recordings_paths and val_supervisions_paths must have the same length"
            if self.hparams.val_recordings_prefix is not None:
                assert len(self.hparams.val_recordings_paths) == len(
                    self.hparams.val_recordings_prefix
                ), "val_recordings_paths and val_recordings_prefix must have the same length"

        # val_prefix == None
        if (
            self.hparams.val_recordings_prefix is None
            and self.hparams.val_recordings_paths is not None
        ):
            log.info(
                f"val_recordings_prefix is None, pls check your val_recordings source is absolute path"
            )

        if (
            self.hparams.val_cuts_prefix is None
            and self.hparams.val_cuts_paths is not None
        ):
            log.info(
                f"val_cuts_prefix is None, pls check your val_cuts source is absolute path"
            )

        if (
            self.hparams.val_recordings_filelist_prefix is None
            and self.hparams.val_recordings_filelist is not None
        ):
            log.info(
                f"val_recordings_filelist_prefix is None, pls check your val_recordings_filelist source is absolute path"
            )

        if (
            self.hparams.val_cuts_filelist_prefix is None
            and self.hparams.val_cuts_filelist is not None
        ):
            log.info(
                f"val_cuts_filelist_prefix is None, pls check your val_cuts_filelist source is absolute path"
            )

        if (
            self.hparams.val_recordings_paths is None
            and self.hparams.val_cuts_paths is None
            and self.hparams.val_recordings_filelist is None
            and self.hparams.val_cuts_filelist is None
        ):
            raise ValueError(
                "val_recordings_paths, val_cuts_paths, val_recordings_filelist, val_cuts_filelist must be provided at least one"
            )

        if self.hparams.max_samples is not None:
            log.info(f"val stage just use {self.hparams.max_samples} samples")
        elif self.hparams.max_samples is None:
            log.info(f"val stage use all samples")

    # check test hparams and load test filelist if test filelist is not None
    def _test_stage_hparams_check(self):
        assert self.hparams.output_dir is not None, "output_dir must be provided"
        # test_recordings_filelist and test_supervisions_filelist and test_cuts_filelist
        if self.hparams.test_recordings_filelist is not None:
            assert len(self.hparams.test_recordings_filelist) == len(
                self.hparams.test_supervisions_filelist
            ), "test_recordings_filelist and test_supervisions_filelist must have the same length"
            if self.hparams.test_recordings_prefix is not None:
                assert len(self.hparams.test_recordings_filelist) == len(
                    self.hparams.test_recordings_prefix
                ), "test_recordings_filelist and test_recordings_prefix must have the same length"
            self.hparams.test_recordings_paths_list = []
            self.hparams.test_supervisions_paths_list = []
            for recordings_path, supervisions_path in zip(
                self.hparams.test_recordings_filelist,
                self.hparams.test_supervisions_filelist,
            ):
                tmp_recordings_list = open_filelist(recordings_path)
                tmp_supervisions_list = open_filelist(supervisions_path)
                assert len(tmp_recordings_list) == len(
                    tmp_supervisions_list
                ), "recordings and supervisisons absolute path in filelist must have the same length"
                self.hparams.test_recordings_paths_list.append(tmp_recordings_list)
                self.hparams.test_supervisions_paths_list.append(tmp_supervisions_list)
        else:
            self.hparams.test_recordings_paths_list = None
            self.hparams.test_supervisions_paths_list = None

        if self.hparams.test_cuts_filelist is not None:
            self.hparams.test_cuts_paths_list = []
            for path in self.hparams.test_cuts_filelist:
                self.hparams.test_cuts_paths_list.append(open_filelist(path))
        else:
            self.hparams.test_cuts_paths_list = None

        # train_recordings_paths and train_supervisions_paths and train_cuts_paths
        if self.hparams.test_recordings_paths is not None:
            assert len(self.hparams.test_recordings_paths) == len(
                self.hparams.test_supervisions_paths
            ), "test_recordings_paths and test_supervisions_paths must have the same length"
            if self.hparams.test_recordings_prefix is not None:
                assert len(self.hparams.test_recordings_paths) == len(
                    self.hparams.test_recordings_prefix
                ), "test_recordings_paths and test_recordings_prefix must have the same length"

        # train_prefix == None
        if (
            self.hparams.test_recordings_prefix is None
            and self.hparams.test_recordings_paths is not None
        ):
            log.info(
                f"test_recordings_prefix is None, pls check your test_recordings source is absolute path"
            )

        if (
            self.hparams.test_cuts_prefix is None
            and self.hparams.test_cuts_paths is not None
        ):
            log.info(
                f"test_cuts_prefix is None, pls check your test_cuts source is absolute path"
            )

        if (
            self.hparams.test_recordings_filelist_prefix is None
            and self.hparams.test_recordings_filelist is not None
        ):
            log.info(
                f"test_recordings_filelist_prefix is None, pls check your test_recordings_filelist source is absolute path"
            )

        if (
            self.hparams.test_cuts_filelist_prefix is None
            and self.hparams.test_cuts_filelist is not None
        ):
            log.info(
                f"test_cuts_filelist_prefix is None, pls check your test_cuts_filelist source is absolute path"
            )

        if (
            self.hparams.test_recordings_paths is None
            and self.hparams.test_cuts_paths is None
            and self.hparams.test_recordings_filelist is None
            and self.hparams.test_cuts_filelist is None
        ):
            raise ValueError(
                "test_recordings_paths, test_cuts_paths, test_recordings_filelist, test_cuts_filelist must be provided at least one"
            )

        if self.hparams.max_samples is not None:
            log.info(f"test stage just use {self.hparams.max_samples} samples")
        elif self.hparams.max_samples is None:
            log.info(f"test stage use all samples")

    # load train dataset
    def _save_train_cutset(self):
        # train_recordings and train_supervisions
        self.train_cuts = CutSet()
        self.train_recordings = RecordingSet()
        self.train_supervisions = SupervisionSet()
        if self.hparams.train_recordings_paths is not None:
            for idx, path in enumerate(self.hparams.train_recordings_paths):
                tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                if self.hparams.train_recordings_prefix is not None:
                    prefix = self.hparams.train_recordings_prefix[idx]
                    if prefix != "":
                        self.train_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.train_recordings += tmp_recordings
                else:
                    self.train_recordings += tmp_recordings

            for path in self.hparams.train_supervisions_paths:
                self.train_supervisions += SupervisionSet.from_jsonl_lazy(path)

        # train_recordings_filelist and train_supervisions_filelist
        if self.hparams.train_recordings_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.train_recordings_paths_list
            ):  # self.hparams.train_recordings_paths_list: [[xxx], [xxx]]
                tmp_recordings = RecordingSet()
                for path in path_list:
                    tmp_recordings += RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                # all recordings in one filelist.txt use the same prefix
                if self.hparams.train_recordings_filelist_prefix is not None:
                    prefix = self.hparams.train_recordings_filelist_prefix[idx]
                    if prefix != "":
                        self.train_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.train_recordings += tmp_recordings
                else:
                    self.train_recordings += tmp_recordings

            # self.hparams.train_supervisions_paths_list: [[xxx], [xxx]]
            for path_list in self.hparams.train_supervisions_paths_list:
                for path in path_list:
                    self.train_supervisions += SupervisionSet.from_jsonl_lazy(path)

        if (self.hparams.train_recordings_paths is not None) or (
            self.hparams.train_recordings_filelist is not None
        ):

            self.train_cuts += CutSet.from_manifests(
                recordings=self.train_recordings, supervisions=self.train_supervisions
            )
            self.train_cuts = self.process_cuts_for_train(self.train_cuts)

        # train_cuts
        if self.hparams.train_cuts_paths is not None:
            for idx, path in enumerate(self.hparams.train_cuts_paths):
                tmp_cuts = CutSet.from_jsonl_lazy(path)
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                if self.hparams.train_cuts_prefix is not None:
                    prefix = self.hparams.train_cuts_prefix[idx]
                    if prefix != "":
                        tmp_cuts = tmp_cuts.with_recording_path_prefix(prefix)
                tmp_cuts = self.process_cuts_for_train(tmp_cuts)
                self.train_cuts += tmp_cuts

        # train_cuts_filelist
        if self.hparams.train_cuts_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.train_cuts_paths_list
            ):  # self.hparams.train_cuts_paths_list: [[xxx], [xxx]]
                tmp_cuts = CutSet()
                # all cuts in one filelist.txt use the same prefix
                if self.hparams.train_cuts_filelist_prefix is not None:
                    prefix = self.hparams.train_cuts_filelist_prefix[idx]
                else:
                    prefix = ""
                for path in tqdm(path_list, desc="load train_cuts_filelist"):
                    tmp_tmp_cuts =  CutSet.from_jsonl_lazy(path)
                    # resample
                    tmp_tmp_cuts = tmp_tmp_cuts.resample(self.hparams.sample_rate)
                    if prefix != "":
                        tmp_tmp_cuts = tmp_tmp_cuts.with_recording_path_prefix(prefix)
                    # process cuts one by one is faster than process all cuts at once
                    tmp_tmp_cuts = self.process_cuts_for_train(tmp_tmp_cuts)
                    tmp_cuts += tmp_tmp_cuts
                self.train_cuts += tmp_cuts

        windows = self.hparams.window_size if self.hparams.window_size is not None else "None"
        min_duration = self.hparams.min_duration if self.hparams.min_duration is not None else "None"
        max_duration = self.hparams.max_duration if self.hparams.max_duration is not None else "None"
        shuffle_train_cuts = "True" if self.hparams.shuffle_train_cuts else "False"

        save_file_name = f"train_cuts_windows-{windows}_min_duration-{min_duration}_max_duration-{max_duration}_shuffle-{shuffle_train_cuts}.jsonl.gz"
        all_cuts_duration = 0.0
        all_cuts_num = 0
        for cut in tqdm(self.train_cuts, desc="calculate train_cuts duration"):
            all_cuts_duration += cut.duration
            all_cuts_num += 1
        log.info(f"all cuts duration: {all_cuts_duration}")
        log.info(f"all cuts num: {all_cuts_num}")

        # shuffle cuts
        if self.hparams.shuffle_train_cuts:
            log.info(f"shuffle train_cuts start")
            self.train_cuts = self.train_cuts.shuffle(rng=random.Random(666))
            log.info(f"shuffle train_cuts success")

        self.train_cuts.to_file(
            os.path.join(self.hparams.output_dir, save_file_name)
        )
        log.info(f"save train_cuts to {os.path.join(self.hparams.output_dir, save_file_name)}")

    # load val dataset
    def _save_val_cutset(self):
        self.val_cuts = CutSet()
        self.val_recordings = RecordingSet()
        self.val_supervisions = SupervisionSet()
        # val_recordings and val_supervisions
        if self.hparams.val_recordings_paths is not None:
            for idx, path in enumerate(self.hparams.val_recordings_paths):
                tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                if self.hparams.val_recordings_prefix is not None:
                    prefix = self.hparams.val_recordings_prefix[idx]
                    if prefix != "":
                        self.val_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.val_recordings += tmp_recordings
                else:
                    self.val_recordings += tmp_recordings

            for path in self.hparams.val_supervisions_paths:
                self.val_supervisions += SupervisionSet.from_jsonl_lazy(path)

        # val_recordings_filelist and val_supervisions_filelist
        if self.hparams.val_recordings_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.val_recordings_paths_list
            ):  # self.hparams.val_recordings_paths_list: [[xxx], [xxx]]
                tmp_recordings = RecordingSet()
                for path in path_list:
                    tmp_recordings += RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                # all recordings in one filelist.txt use the same prefix
                if self.hparams.val_recordings_filelist_prefix is not None:
                    prefix = self.hparams.val_recordings_filelist_prefix[idx]
                    if prefix != "":
                        self.val_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.val_recordings += tmp_recordings
                else:
                    self.val_recordings += tmp_recordings

        if (self.hparams.val_recordings_paths is not None) or (
            self.hparams.val_recordings_filelist is not None
        ):

            self.val_cuts += CutSet.from_manifests(
                recordings=self.val_recordings, supervisions=self.val_supervisions
            )

        # val_cuts
        if self.hparams.val_cuts_paths is not None:
            for idx, path in enumerate(self.hparams.val_cuts_paths):
                tmp_cuts = CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                if self.hparams.val_cuts_prefix is not None:
                    prefix = self.hparams.val_cuts_prefix[idx]
                    if prefix != "":
                        self.val_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.val_cuts += tmp_cuts
                else:
                    self.val_cuts += tmp_cuts

        # val_cuts_filelist
        if self.hparams.val_cuts_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.val_cuts_paths_list
            ):  # self.hparams.val_cuts_paths_list: [[xxx], [xxx]]
                tmp_cuts = CutSet()
                for path in path_list:
                    tmp_cuts += CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                # all cuts in one filelist.txt use the same prefix
                if self.hparams.val_cuts_filelist_prefix is not None:
                    prefix = self.hparams.val_cuts_filelist_prefix[idx]
                    if prefix != "":
                        self.val_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.val_cuts += tmp_cuts
                else:
                    self.val_cuts += tmp_cuts

        if self.hparams.max_samples is not None:
            self.val_cuts = self.val_cuts.sample(self.hparams.max_samples)

        if self.hparams.output_dir is not None:
            self.val_cuts.to_file(
                os.path.join(self.hparams.output_dir, f"val_cuts_sample-{self.hparams.max_samples}.jsonl.gz")
            )
        log.info(f"save val_cuts to {os.path.join(self.hparams.output_dir, f'val_cuts_sample-{self.hparams.max_samples}.jsonl.gz')}")

    # load test dataset
    def _save_test_cutset(self):
        self.test_cuts = CutSet()
        self.test_recordings = RecordingSet()
        self.test_supervisions = SupervisionSet()
        # test_recordings and test_supervisions
        if self.hparams.test_recordings_paths is not None:
            for idx, path in enumerate(self.hparams.test_recordings_paths):
                tmp_recordings = RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                if self.hparams.test_recordings_prefix is not None:
                    prefix = self.hparams.test_recordings_prefix[idx]
                    if prefix != "":
                        self.test_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.test_recordings += tmp_recordings
                else:
                    self.test_recordings += tmp_recordings

            for path in self.hparams.test_supervisions_paths:
                self.test_supervisions += SupervisionSet.from_jsonl_lazy(path)

        # test_recordings_filelist
        if self.hparams.test_recordings_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.test_recordings_paths_list
            ):  # self.hparams.test_recordings_paths_list: [[xxx], [xxx]]
                tmp_recordings = RecordingSet()
                for path in path_list:
                    tmp_recordings += RecordingSet.from_jsonl_lazy(path)
                # resample
                tmp_recordings = tmp_recordings.resample(self.hparams.sample_rate)
                # all recordings in one filelist.txt use the same prefix
                if self.hparams.test_recordings_filelist_prefix is not None:
                    prefix = self.hparams.test_recordings_filelist_prefix[idx]
                    if prefix != "":
                        self.test_recordings += tmp_recordings.with_path_prefix(prefix)
                    else:
                        self.test_recordings += tmp_recordings
                else:
                    self.test_recordings += tmp_recordings

        if (self.hparams.test_recordings_paths is not None) or (
            self.hparams.test_recordings_filelist is not None
        ):

            self.test_cuts += CutSet.from_manifests(
                recordings=self.test_recordings, supervisions=self.test_supervisions
            )

        # test_cuts
        if self.hparams.test_cuts_paths is not None:
            for idx, path in enumerate(self.hparams.test_cuts_paths):
                tmp_cuts = CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                if self.hparams.test_cuts_prefix is not None:
                    prefix = self.hparams.test_cuts_prefix[idx]
                    if prefix != "":
                        self.test_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.test_cuts += tmp_cuts

        # test_cuts_filelist
        if self.hparams.test_cuts_filelist is not None:
            for idx, path_list in enumerate(
                self.hparams.test_cuts_paths_list
            ):  # self.hparams.test_cuts_paths_list: [[xxx], [xxx]]
                tmp_cuts = CutSet()
                for path in path_list:
                    tmp_cuts += CutSet.from_jsonl_lazy(path)
                # resample
                tmp_cuts = tmp_cuts.resample(self.hparams.sample_rate)
                # all cuts in one filelist.txt use the same prefix
                if self.hparams.test_cuts_filelist_prefix is not None:
                    prefix = self.hparams.test_cuts_filelist_prefix[idx]
                    if prefix != "":
                        self.test_cuts += tmp_cuts.with_recording_path_prefix(prefix)
                    else:
                        self.test_cuts += tmp_cuts

        if self.hparams.max_samples is not None:
            self.test_cuts = self.test_cuts.sample(self.hparams.max_samples)

        if self.hparams.output_dir is not None:
            self.test_cuts.to_file(
                os.path.join(self.hparams.output_dir, f"test_cuts_sample-{self.hparams.max_samples}.jsonl.gz")
            )
        log.info(f"save test_cuts to {os.path.join(self.hparams.output_dir, f'test_cuts_sample-{self.hparams.max_samples}.jsonl.gz')}")


if __name__ == "__main__":
    # Example: build Lhotse cutsets for LibriTTS at 24 kHz.
    # Replace the placeholder paths with your local Lhotse manifests.
    data_module = LhotsePreProcess(
        train_recordings_paths=[
            "/path/to/lhotse/libritts/libritts_recordings_train-clean-100.jsonl.gz",
        ],
        train_supervisions_paths=[
            "/path/to/lhotse/libritts/libritts_supervisions_train-clean-100.jsonl.gz",
        ],
        train_recordings_prefix=[
            "/path/to/speech/24kHz/LibriTTS",
        ],
        val_recordings_paths=[
            "/path/to/lhotse/libritts/libritts_recordings_dev-clean.jsonl.gz",
        ],
        val_supervisions_paths=[
            "/path/to/lhotse/libritts/libritts_supervisions_dev-clean.jsonl.gz",
        ],
        val_recordings_prefix=["/path/to/speech/24kHz/LibriTTS"],
        window_size=None,
        min_duration=None,
        max_duration=30,
        num_jobs=40,
        sample_rate=24000,
        output_dir="/path/to/output",
        stage="fit",
        max_samples=128,
        shuffle_train_cuts=True,
        codec_or_lm="codec",
    )

    data_module.save_cutset()
