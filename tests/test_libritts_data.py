from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
import lightning.pytorch as pl
from lhotse import CutSet

from dmel_codec.dataset.lhotse_tts_dataset import (
    CappedDurationConstraint,
    LhotseDataModule,
    LhotseTTSDataset,
)
from dmel_codec.dataset.make_subset_manifest import make_subset_manifest
from dmel_codec.dataset.prepare_libritts import SPLITS, prepare_libritts, sha256_file
from dmel_codec.evaluation.run_codec_eval import evaluate
from dmel_codec.tools.training_exposure import replay_training_exposure
from dmel_codec.utils.callbacks import (
    FinalizedModelCheckpoint,
    StopAfterValidationCheckpoint,
)


def _make_tiny_libritts(root: Path) -> None:
    speaker_lines = []
    for split_index, split in enumerate(SPLITS, start=1):
        speaker = str(10_000 + split_index)
        chapter = str(20_000 + split_index)
        chapter_dir = root / split / speaker / chapter
        chapter_dir.mkdir(parents=True)
        transcript_lines = []
        for utterance_index, duration in enumerate((1.0, 4.0)):
            utterance_id = f"{speaker}_{chapter}_{utterance_index:06d}_000000"
            samples = np.full(round(24_000 * duration), 0.2, dtype=np.float32)
            sf.write(chapter_dir / f"{utterance_id}.wav", samples, 24_000, subtype="PCM_16")
            transcript_lines.append(
                f"{utterance_id}\tOriginal sentence {utterance_index}.\t"
                f"FULL normalized sentence {utterance_index}.\n"
            )
        (chapter_dir / f"{speaker}_{chapter}.trans.tsv").write_text(
            "".join(transcript_lines), encoding="utf-8"
        )
        speaker_lines.append(f"{speaker} | F | {split} | 1.00 | Test Speaker\n")
    (root / "SPEAKERS.txt").write_text("".join(speaker_lines), encoding="utf-8")


def _prepare_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    data_root = tmp_path / "LibriTTS"
    output_dir = tmp_path / "manifests"
    _make_tiny_libritts(data_root)
    for directory, _, _ in os.walk(data_root):
        Path(directory).chmod(0o555)
    summary = prepare_libritts(
        data_root, output_dir, num_jobs=2, sample_size=1, seed=7
    )
    return data_root, output_dir, summary


def test_prepare_libritts_is_audited_hashed_and_idempotent(tmp_path: Path) -> None:
    data_root, output_dir, summary = _prepare_fixture(tmp_path)

    assert summary["data_root"] == str(data_root.absolute())
    assert summary["resolved_data_root"] == str(data_root.resolve())
    assert set(summary["splits"]) == set(SPLITS)
    assert all(metadata["utterances"] == 2 for metadata in summary["splits"].values())
    assert all(
        metadata["utterances_shorter_than_3s"] == 1
        for metadata in summary["splits"].values()
    )
    assert len(summary["manifests"]) == 11
    for name, metadata in summary["manifests"].items():
        assert sha256_file(output_dir / name) == metadata["sha256"]

    eval_cut = next(iter(CutSet.from_file(output_dir / "dev-clean_eval.jsonl.gz")))
    assert eval_cut.supervisions[0].text.startswith("FULL normalized sentence")
    assert eval_cut.supervisions[0].speaker
    codec_cut = next(
        iter(CutSet.from_file(output_dir / "train-clean-100_codec.jsonl.gz"))
    )
    assert not codec_cut.supervisions

    summary_mtime = (output_dir / "manifest_summary.json").stat().st_mtime_ns
    repeated = prepare_libritts(data_root, output_dir, num_jobs=1, sample_size=1)
    assert repeated == json.loads(
        (output_dir / "manifest_summary.json").read_text(encoding="utf-8")
    )
    assert (output_dir / "manifest_summary.json").stat().st_mtime_ns == summary_mtime


def test_dataset_shapes_lengths_text_and_amplitude(tmp_path: Path) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)
    codec_cuts = CutSet.from_file(output_dir / "train-clean-100_codec.jsonl.gz")
    torch.manual_seed(2026)
    batch = LhotseTTSDataset(max_audio_seconds=3.0, random_crop=True)[codec_cuts]

    assert batch["audios"].ndim == 3
    assert batch["audios"].shape[:2] == (2, 1)
    assert batch["audio_lengths"].shape == (2,)
    assert sorted(batch["audio_lengths"].tolist()) == [24_000, 72_000]
    assert batch["text"] == [None, None]
    assert len(batch["cut_ids"]) == 2
    assert 0.19 < batch["audios"].abs().max().item() < 0.21

    eval_cuts = CutSet.from_file(output_dir / "dev-clean_eval.jsonl.gz")
    eval_batch = LhotseTTSDataset()[eval_cuts]
    assert all(text.startswith("FULL normalized sentence") for text in eval_batch["text"])
    assert sorted(eval_batch["audio_lengths"].tolist()) == [24_000, 96_000]


def test_cut_id_crop_is_reproducible_across_dataset_processes() -> None:
    first = LhotseTTSDataset(
        max_audio_seconds=3.0,
        random_crop=True,
        crop_seed=2026,
    )
    repeated = LhotseTTSDataset(
        max_audio_seconds=3.0,
        random_crop=True,
        crop_seed=2026,
    )
    other_seed = LhotseTTSDataset(
        max_audio_seconds=3.0,
        random_crop=True,
        crop_seed=2027,
    )

    assert first._crop_fraction("cut-a") == repeated._crop_fraction("cut-a")
    assert first._crop_fraction("cut-a") != first._crop_fraction("cut-b")
    assert first._crop_fraction("cut-a") != other_seed._crop_fraction("cut-a")
    assert 0.0 <= first._crop_fraction("cut-a") < 1.0


def test_capped_duration_constraint_matches_training_crop(tmp_path: Path) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)
    cuts = CutSet.from_file(output_dir / "train-clean-100_codec.jsonl.gz")
    long_cut = next(cut for cut in cuts if cut.duration == 4.0)
    constraint = CappedDurationConstraint(
        max_duration=6.0,
        max_sample_duration=3.0,
    )

    constraint.add(long_cut)
    constraint.add(long_cut)
    assert constraint.measure_length(long_cut) == 3.0
    assert constraint.current == 6.0
    assert not constraint.exceeded()
    assert constraint.close_to_exceeding()

    restored = CappedDurationConstraint()
    restored.load_state_dict(constraint.state_dict())
    assert restored == constraint
    assert restored.current == constraint.current


def test_capped_training_sampler_resumes_at_the_next_exact_batch(tmp_path: Path) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)
    common = {
        "stage": "fit",
        "train_cuts_path": str(output_dir / "train-clean-100_codec.jsonl.gz"),
        "val_cuts_path": str(output_dir / "dev-clean_eval.jsonl.gz"),
        "train_max_durations": 3.0,
        "val_max_durations": 5.0,
        "train_num_workers": 0,
        "val_num_workers": 0,
        "world_size": 1,
        "rank": 0,
        "seed": 2026,
        "train_crop_seed": 2026,
    }
    source = LhotseDataModule(**common)
    source.setup("fit")
    source_iterator = iter(source.train_sampler)
    first_batch = [cut.id for cut in next(source_iterator)]
    state = source.state_dict()

    def drain(iterator) -> list[list[str]]:
        batches = []
        while True:
            try:
                batches.append([cut.id for cut in next(iterator)])
            except StopIteration:
                return batches

    remaining = drain(source_iterator)

    restored = LhotseDataModule(**common)
    restored.setup("fit")
    restored.load_state_dict(state)
    restored_remaining = drain(iter(restored.train_sampler))

    assert first_batch
    assert remaining
    assert restored_remaining == remaining


class _CutRecordingModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_cut_ids: list[tuple[str, ...]] = []

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.seen_cut_ids.append(tuple(batch["cut_ids"]))
        return (self.weight * batch["audios"].mean()).square()

    def validation_step(self, batch, batch_idx):
        del batch_idx
        self.log("val_loss", (self.weight * batch["audios"].mean()).square())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_lightning_checkpoint_continues_the_exact_capped_sampler_batch(
    tmp_path: Path,
) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)
    common = {
        "stage": "fit",
        "train_cuts_path": str(output_dir / "train-clean-100_codec.jsonl.gz"),
        "val_cuts_path": str(output_dir / "dev-clean_eval.jsonl.gz"),
        "train_max_durations": 3.0,
        "val_max_durations": 5.0,
        "train_num_workers": 2,
        "val_num_workers": 0,
        "world_size": 1,
        "rank": 0,
        "seed": 2026,
        "train_crop_seed": 2026,
    }

    def checkpoint(directory: Path) -> FinalizedModelCheckpoint:
        return FinalizedModelCheckpoint(
            dirpath=directory,
            filename="step-{step:08d}",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
            save_on_train_epoch_end=False,
            save_top_k=1,
            save_last=True,
        )

    def trainer(callbacks) -> pl.Trainer:
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            max_epochs=10,
            val_check_interval=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=1,
        )

    uninterrupted_model = _CutRecordingModel()
    trainer([checkpoint(tmp_path / "uninterrupted")]).fit(
        uninterrupted_model,
        datamodule=LhotseDataModule(**common),
    )

    segmented_checkpoint = checkpoint(tmp_path / "segmented")
    first_model = _CutRecordingModel()
    trainer([segmented_checkpoint, StopAfterValidationCheckpoint(1)]).fit(
        first_model,
        datamodule=LhotseDataModule(**common),
    )
    # The controlled stop preserves the validation-time ``last`` checkpoint so
    # every interval remains resumable even when its step is outside top-k.
    resume_path = Path(segmented_checkpoint.last_model_path)
    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert "LhotseDataModule" in payload

    resumed_model = _CutRecordingModel()
    trainer([checkpoint(tmp_path / "segmented")]).fit(
        resumed_model,
        datamodule=LhotseDataModule(**common),
        ckpt_path=resume_path,
    )

    assert first_model.seen_cut_ids + resumed_model.seen_cut_ids == (
        uninterrupted_model.seen_cut_ids
    )


def test_distributed_training_samplers_are_disjoint(tmp_path: Path) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)
    common = {
        "stage": "fit",
        "train_cuts_path": str(output_dir / "train_codec.jsonl.gz"),
        "val_cuts_path": str(output_dir / "dev-clean_eval.jsonl.gz"),
        "train_max_durations": 4.5,
        "val_max_durations": 5.0,
        "train_num_workers": 0,
        "val_num_workers": 0,
        "world_size": 2,
        "seed": 2026,
    }
    modules = [LhotseDataModule(**common, rank=rank) for rank in (0, 1)]
    for module in modules:
        module.setup("fit")
        module.train_sampler.set_epoch(3)

    rank_ids = []
    for module in modules:
        rank_ids.append(
            {
                cut.id
                for batch in module.train_sampler
                for cut in batch
            }
        )
    assert rank_ids[0]
    assert rank_ids[1]
    assert rank_ids[0].isdisjoint(rank_ids[1])


def test_training_sampler_shuffle_is_reproducible_per_epoch(tmp_path: Path) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)

    def order(epoch: int) -> list[str]:
        module = LhotseDataModule(
            stage="fit",
            train_cuts_path=str(output_dir / "train_codec.jsonl.gz"),
            val_cuts_path=str(output_dir / "dev-clean_eval.jsonl.gz"),
            train_max_durations=4.5,
            val_max_durations=5.0,
            train_num_workers=0,
            val_num_workers=0,
            world_size=1,
            rank=0,
            seed=2026,
        )
        module.setup("fit")
        module.train_sampler.set_epoch(epoch)
        return [cut.id for batch in module.train_sampler for cut in batch]

    assert order(4) == order(4)
    assert order(4) != order(5)


def test_training_exposure_replays_independent_cropped_batches(tmp_path: Path) -> None:
    _, output_dir, _ = _prepare_fixture(tmp_path)
    manifest = output_dir / "train_codec.jsonl.gz"
    config_path = tmp_path / "resolved_config.yaml"
    output_path = tmp_path / "training_exposure.json"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "module": {
                        "_target_": (
                            "dmel_codec.dataset.lhotse_tts_dataset."
                            "LhotseDataModule"
                        ),
                        "stage": "fit",
                        "train_cuts_path": str(manifest),
                        "val_cuts_path": str(output_dir / "dev-clean_eval.jsonl.gz"),
                        "test_cuts_path": str(output_dir / "test-clean_eval.jsonl.gz"),
                        "train_max_durations": 4.5,
                        "val_max_durations": 5.0,
                        "test_max_durations": 5.0,
                        "train_num_workers": 2,
                        "val_num_workers": 2,
                        "test_num_workers": 2,
                        "pin_memory": False,
                        "world_size": None,
                        "rank": None,
                        "seed": 2026,
                        "train_max_audio_seconds": 3.0,
                        "train_random_crop": True,
                    }
                },
                "experiment": {"trainer": {"max_steps": 4}},
            }
        ),
        encoding="utf-8",
    )

    result = replay_training_exposure(config_path, output=output_path)

    assert output_path.is_file()
    assert result["replay"]["steps"] == 4
    assert result["replay"]["effective_duration_seconds"] <= 18.0
    assert result["replay"]["sampled_cuts"] >= 4
    assert result["replay"]["cuts_per_batch"]["max"] >= 1
    assert result["sampler"]["waveforms_concatenated"] is False
    assert result["manifest"]["crop_capped_duration_seconds"] == 12.0
    assert result["sampler"]["duration_constraint_is_strict"] is False


def test_hydra_style_prepare_arguments(tmp_path: Path) -> None:
    from dmel_codec.dataset.prepare_libritts import _normalize_hydra_style_args

    assert _normalize_hydra_style_args(
        ["data.root=/datasets/LibriTTS", f"output_dir={tmp_path}"]
    ) == ["--data-root", "/datasets/LibriTTS", "--output-dir", str(tmp_path)]


def test_subset_manifest_merge_hash_selection_and_reuse(tmp_path: Path) -> None:
    _, manifest_dir, _ = _prepare_fixture(tmp_path)
    first_input = manifest_dir / "train-clean-100_codec.jsonl.gz"
    second_input = manifest_dir / "train-clean-360_codec.jsonl.gz"
    output = tmp_path / "derived" / "clean-subset.jsonl.gz"

    first = make_subset_manifest(
        [first_input, second_input], output, max_cuts=2, seed=2027
    )
    first_bytes = output.read_bytes()
    repeated = make_subset_manifest(
        [first_input, second_input], output, max_cuts=2, seed=2027
    )
    selected = list(CutSet.from_file(output))

    assert first == repeated
    assert output.read_bytes() == first_bytes
    assert first["cut_count"] == 2
    assert len({cut.id for cut in selected}) == 2
    assert sum(first["split_counts"].values()) == 2
    assert first["output"]["sha256"] == sha256_file(output)


def test_subset_manifest_rejects_duplicate_inputs(tmp_path: Path) -> None:
    _, manifest_dir, _ = _prepare_fixture(tmp_path)
    source = manifest_dir / "train-clean-100_codec.jsonl.gz"
    with pytest.raises(ValueError, match="Input manifests must be unique"):
        make_subset_manifest([source, source], tmp_path / "duplicate.jsonl.gz")


def test_identity_evaluator_is_complete_and_batch_invariant(tmp_path: Path) -> None:
    _, manifest_dir, _ = _prepare_fixture(tmp_path)
    common = {
        "system": "identity",
        "manifest": manifest_dir / "dev-clean_eval.jsonl.gz",
        "split": "dev-clean",
        "output_dir": tmp_path / "evaluations",
        "cache_root": tmp_path / "cache",
        "model_cache_root": tmp_path / "models",
        "device": "cpu",
        "seed": 2026,
        "max_cuts": 2,
        "warmup_cuts": 1,
        "remeasure_rtf": False,
        "audio_samples": 0,
        "metrics": "si_sdr,mr_stft,ms_log_mel",
        "test_gate": None,
        "asr_model_metadata": None,
        "speaker_model_metadata": None,
        "visqol_metadata": None,
        "bigvgan_checkpoint": None,
        "bigvgan_config": None,
        "dmel_range": None,
        "dac_checkpoint": None,
        "n_quantizers": 2,
        "ours_checkpoint": None,
    }
    first = evaluate(Namespace(**common, run_id="batch-one", max_duration=1.1))
    second = evaluate(Namespace(**common, run_id="batch-all", max_duration=10.0))

    assert first["utterances"] == second["utterances"] == 2
    assert first["failed_utterances"] == second["failed_utterances"] == 0
    assert first["metrics"] == second["metrics"]
    for run_id in ("batch-one", "batch-all"):
        run_dir = common["output_dir"] / run_id
        assert sum(1 for _ in (run_dir / "per_utterance.jsonl").open()) == 2
        assert (run_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
        config = yaml.safe_load(
            (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        environment = json.loads(
            (run_dir / "environment.json").read_text(encoding="utf-8")
        )
        assert config["metric_cpu_threads"] == 8
        assert environment["torch_cpu_threads"] == 8
        for name in (
            "resolved_config.yaml",
            "environment.json",
            "git_state.json",
            "data_manifest.json",
            "model_metadata.json",
            "summary.json",
        ):
            assert (run_dir / name).is_file()
