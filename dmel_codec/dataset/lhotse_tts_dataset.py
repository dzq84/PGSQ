from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from lightning import LightningDataModule
from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.sampling.base import CutSampler, TimeConstraint
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dmel_codec.utils.logger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=False)


@dataclass
class CappedDurationConstraint(TimeConstraint):
    """Batch by the duration that remains after the training crop."""

    max_sample_duration: float = 3.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_sample_duration <= 0:
            raise ValueError("max_sample_duration must be positive")

    def _capped_duration(self, example) -> float:
        return min(float(example.duration), self.max_sample_duration)

    def add(self, example) -> None:
        if self.max_duration is not None:
            duration = self._maybe_apply_quadratic_correction(
                self._capped_duration(example)
            )
            self.current += duration
            self.longest_seen = max(self.longest_seen, duration)
        self.num_cuts += 1

    def measure_length(self, example) -> float:
        return self._capped_duration(example)

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        self.max_sample_duration = float(
            state_dict.pop("max_sample_duration", self.max_sample_duration)
        )
        super().load_state_dict(state_dict)

    def __add__(self, other: "CappedDurationConstraint"):
        if not isinstance(other, CappedDurationConstraint):
            return NotImplemented
        if (
            self.max_duration != other.max_duration
            or self.max_cuts != other.max_cuts
            or self.quadratic_duration != other.quadratic_duration
            or self.max_sample_duration != other.max_sample_duration
        ):
            raise ValueError("Cannot combine different capped duration constraints")
        return CappedDurationConstraint(
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            current=self.current + other.current,
            num_cuts=self.num_cuts + other.num_cuts,
            longest_seen=max(self.longest_seen, other.longest_seen),
            quadratic_duration=self.quadratic_duration,
            concatenate_cuts=self.concatenate_cuts,
            max_sample_duration=self.max_sample_duration,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CappedDurationConstraint)
            and super().__eq__(other)
            and self.max_sample_duration == other.max_sample_duration
        )


class ResumableDynamicBucketingSampler(DynamicBucketingSampler):
    """Add checkpoint continuation for the custom capped-duration constraint."""

    _STATE_SCHEMA_VERSION = 1

    def state_dict(self) -> dict[str, Any]:
        if self.constraint is None:
            return super().state_dict()
        if not isinstance(self.constraint, CappedDurationConstraint):
            raise TypeError(
                "Only CappedDurationConstraint is supported for sampler resume"
            )
        state = CutSampler.state_dict(self)
        state.update(
            {
                "dmel_schema_version": self._STATE_SCHEMA_VERSION,
                "max_duration": self.max_duration,
                "max_cuts": self.max_cuts,
                "consistent_ids": self.consistent_ids,
                "buffer_size": self.buffer_size,
                "num_cuts_for_bins_estimate": self.num_cuts_for_bins_estimate,
                "quadratic_duration": self.quadratic_duration,
                "constraint": self.constraint.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.constraint is None:
            super().load_state_dict(state_dict)
            return
        if not isinstance(self.constraint, CappedDurationConstraint):
            raise TypeError(
                "Only CappedDurationConstraint is supported for sampler resume"
            )
        state = dict(state_dict)
        version = state.pop("dmel_schema_version")
        if version != self._STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported sampler state schema: {version}")
        self.max_duration = state.pop("max_duration")
        self.max_cuts = state.pop("max_cuts")
        self.consistent_ids = state.pop("consistent_ids")
        self.buffer_size = state.pop("buffer_size")
        self.num_cuts_for_bins_estimate = state.pop("num_cuts_for_bins_estimate")
        self.quadratic_duration = state.pop("quadratic_duration")
        saved_constraint = CappedDurationConstraint()
        saved_constraint.load_state_dict(state.pop("constraint"))
        if saved_constraint != self.constraint:
            raise ValueError("Sampler constraint differs from the checkpoint")
        CutSampler.load_state_dict(self, state)
        self._fast_forward()


class LhotseTTSDataset(Dataset):
    """Load already-batched Lhotse cuts without changing waveform amplitude."""

    def __init__(
        self,
        *,
        max_audio_seconds: float | None = None,
        random_crop: bool = False,
        crop_seed: int | None = None,
    ) -> None:
        if random_crop and max_audio_seconds is None:
            raise ValueError("random_crop requires max_audio_seconds")
        self.max_audio_seconds = max_audio_seconds
        self.random_crop = random_crop
        self.crop_seed = crop_seed

    def _crop_fraction(self, cut_id: str) -> float:
        if self.crop_seed is None:
            return float(torch.rand(1).item())
        digest = hashlib.sha256(f"{self.crop_seed}:{cut_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    def _crop(self, cut):
        if not self.random_crop or cut.duration <= self.max_audio_seconds:
            return cut
        max_offset = cut.duration - float(self.max_audio_seconds)
        offset = self._crop_fraction(cut.id) * max_offset
        return cut.truncate(
            offset=offset,
            duration=float(self.max_audio_seconds),
            preserve_id=True,
        )

    def __getitem__(self, cuts: CutSet) -> dict[str, Any]:
        audio_list: list[torch.Tensor] = []
        audio_lengths: list[int] = []
        texts: list[str | None] = []
        cut_ids: list[str] = []
        speaker_ids: list[str | None] = []
        audio_paths: list[str] = []

        for original_cut in cuts:
            cut = self._crop(original_cut)
            audio = torch.from_numpy(cut.load_audio()).to(torch.float32)
            if audio.ndim != 2 or audio.shape[0] != 1:
                raise ValueError(
                    f"Expected mono audio [1, T] for {original_cut.id}, got {audio.shape}"
                )
            if not torch.isfinite(audio).all():
                raise ValueError(f"Non-finite waveform samples in {original_cut.id}")

            supervision = (
                original_cut.supervisions[0] if original_cut.supervisions else None
            )
            source = original_cut.recording.sources[0].source
            audio_list.append(audio)
            audio_lengths.append(audio.shape[-1])
            texts.append(supervision.text if supervision else None)
            cut_ids.append(original_cut.id)
            speaker_ids.append(
                supervision.speaker
                if supervision and supervision.speaker
                else original_cut.id.split("_", maxsplit=1)[0]
            )
            audio_paths.append(str(source))

        if not audio_list:
            raise ValueError("Received an empty CutSet batch")
        max_length = max(audio_lengths)
        audios = torch.stack(
            [F.pad(audio, (0, max_length - audio.shape[-1])) for audio in audio_list]
        )
        return {
            "text": texts,
            "audios": audios,
            "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
            "cut_ids": cut_ids,
            "speaker_ids": speaker_ids,
            "audio_paths": audio_paths,
        }


class LhotseDataModule(LightningDataModule):
    def __init__(
        self,
        stage: str,
        train_cuts_path: str | None = None,
        val_cuts_path: str | None = None,
        test_cuts_path: str | None = None,
        train_max_durations: float | None = None,
        val_max_durations: float | None = None,
        test_max_durations: float | None = None,
        train_num_workers: int | None = None,
        val_num_workers: int | None = None,
        test_num_workers: int | None = None,
        pin_memory: bool = False,
        world_size: int | None = None,
        rank: int | None = None,
        seed: int = 2026,
        train_max_audio_seconds: float | None = 3.0,
        train_random_crop: bool = True,
        train_crop_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._check_hparams()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None
        self._pending_train_sampler_state = None

    def _check_hparams(self) -> None:
        if self.hparams.stage not in {"fit", "validate", "test"}:
            raise ValueError("stage must be one of: fit, validate, test")
        if self.hparams.stage == "fit":
            self._require("train")
            self._require("val")
        elif self.hparams.stage == "validate":
            self._require("val")
        else:
            self._require("test")

    def _require(self, split: str) -> None:
        path_key = f"{split}_cuts_path"
        duration_key = f"{split}_max_durations"
        workers_key = f"{split}_num_workers"
        for key in (path_key, duration_key, workers_key):
            if self.hparams.get(key) is None:
                raise ValueError(f"{key} must be provided for stage={self.hparams.stage}")

    def _distributed_context(self) -> tuple[int, int]:
        trainer = getattr(self, "trainer", None)
        world_size = self.hparams.world_size
        rank = self.hparams.rank
        if world_size is None:
            world_size = trainer.world_size if trainer is not None else 1
        if rank is None:
            rank = trainer.global_rank if trainer is not None else 0
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed context rank={rank}, world_size={world_size}")
        return int(world_size), int(rank)

    def _sampler(
        self,
        cuts: CutSet,
        *,
        max_duration: float,
        shuffle: bool,
        buffer_size: int = 20_000,
        sample_duration_cap: float | None = None,
    ) -> ResumableDynamicBucketingSampler:
        world_size, rank = self._distributed_context()
        sample_count = sum(1 for _ in islice(cuts, 10))
        if sample_count == 0:
            raise ValueError("Cannot build a sampler from an empty CutSet")
        bucketing = (
            {"duration_bins": []}
            if sample_count == 1
            else {"num_buckets": sample_count}
        )
        constraint = (
            CappedDurationConstraint(
                max_duration=max_duration,
                max_sample_duration=sample_duration_cap,
            )
            if sample_duration_cap is not None
            else None
        )
        return ResumableDynamicBucketingSampler(
            cuts,
            max_duration=None if constraint is not None else max_duration,
            constraint=constraint,
            shuffle=shuffle,
            # Lhotse duplicates tail cuts to balance ranks when drop_last=False.
            # Dropping only the distributed tail preserves strict rank disjointness.
            drop_last=world_size > 1,
            world_size=world_size,
            rank=rank,
            seed=int(self.hparams.seed),
            buffer_size=buffer_size,
            **bucketing,
        )

    def setup(self, stage: str | None = None) -> None:
        active_stage = stage or self.hparams.stage
        if active_stage == "fit":
            train_cuts = CutSet.from_file(Path(self.hparams.train_cuts_path))
            val_cuts = CutSet.from_file(Path(self.hparams.val_cuts_path))
            self.train_dataset = LhotseTTSDataset(
                max_audio_seconds=self.hparams.train_max_audio_seconds,
                random_crop=bool(self.hparams.train_random_crop),
                crop_seed=self.hparams.train_crop_seed,
            )
            self.val_dataset = LhotseTTSDataset()
            self.train_sampler = self._sampler(
                train_cuts,
                max_duration=float(self.hparams.train_max_durations),
                shuffle=True,
                buffer_size=100_000,
                sample_duration_cap=(
                    float(self.hparams.train_max_audio_seconds)
                    if self.hparams.train_random_crop
                    else None
                ),
            )
            if self._pending_train_sampler_state is not None:
                self.train_sampler.load_state_dict(self._pending_train_sampler_state)
                self._pending_train_sampler_state = None
            self.val_sampler = self._sampler(
                val_cuts,
                max_duration=float(self.hparams.val_max_durations),
                shuffle=False,
            )
            log.info("train_sampler: %s", self.train_sampler)
            log.info("val_sampler: %s", self.val_sampler)
        elif active_stage == "validate":
            val_cuts = CutSet.from_file(Path(self.hparams.val_cuts_path))
            self.val_dataset = LhotseTTSDataset()
            self.val_sampler = self._sampler(
                val_cuts,
                max_duration=float(self.hparams.val_max_durations),
                shuffle=False,
            )
        elif active_stage == "test":
            test_cuts = CutSet.from_file(Path(self.hparams.test_cuts_path))
            self.test_dataset = LhotseTTSDataset()
            self.test_sampler = self._sampler(
                test_cuts,
                max_duration=float(self.hparams.test_max_durations),
                shuffle=False,
            )
        else:
            raise ValueError(f"Unsupported setup stage: {active_stage}")

    @staticmethod
    def _loader(dataset, sampler, num_workers: int, pin_memory: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            self.train_sampler,
            int(self.hparams.train_num_workers),
            bool(self.hparams.pin_memory),
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset,
            self.val_sampler,
            int(self.hparams.val_num_workers),
            bool(self.hparams.pin_memory),
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset,
            self.test_sampler,
            int(self.hparams.test_num_workers),
            bool(self.hparams.pin_memory),
        )

    def state_dict(self) -> dict[str, Any]:
        if self.train_sampler is None:
            return {}
        sampler_state = self.train_sampler.state_dict()
        prefetched_batches = 0
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            completed = int(
                trainer.fit_loop.epoch_loop.batch_progress.current.completed
            )
            diagnostics = sampler_state["diagnostics"]
            current_epoch = diagnostics["current_epoch"]
            epoch_stats = diagnostics["stats_per_epoch"][current_epoch]
            yielded = int(epoch_stats["kept_batches"])
            if completed > yielded:
                raise RuntimeError(
                    "Train sampler completed more batches than it yielded: "
                    f"completed={completed}, yielded={yielded}"
                )
            prefetched_batches = yielded - completed
            # Lightning prefetches one batch. Only completed batches belong in
            # the checkpoint continuation point; fast-forward recomputes cuts.
            epoch_stats["kept_batches"] = completed
        return {
            "schema_version": 1,
            "discarded_prefetched_batches": prefetched_batches,
            "train_sampler": sampler_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = dict(state_dict)
        if state.pop("schema_version") != 1:
            raise ValueError("Unsupported datamodule state schema")
        prefetched_batches = int(state.pop("discarded_prefetched_batches", 0))
        if prefetched_batches < 0:
            raise ValueError("discarded_prefetched_batches must be non-negative")
        sampler_state = state.pop("train_sampler")
        if state:
            raise ValueError(f"Unexpected datamodule state keys: {sorted(state)}")
        if self.train_sampler is None:
            self._pending_train_sampler_state = sampler_state
        else:
            self.train_sampler.load_state_dict(sampler_state)
