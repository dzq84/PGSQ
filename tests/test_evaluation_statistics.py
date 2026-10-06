from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from dmel_codec.evaluation.adapters import OursAdapter
from dmel_codec.evaluation.bitrate import (
    CodeStatisticsAccumulator,
    nominal_bitrate_kbps,
)
from dmel_codec.evaluation.cache import (
    JsonCache,
    ReconstructionCache,
    reconstruction_cache_key,
    sha256_file,
)
from dmel_codec.evaluation.dependency import CodeDependencyAccumulator
from dmel_codec.evaluation.statistics import (
    corpus_delta_wer,
    corpus_rtf,
    corpus_wer,
    paired_speaker_cluster_bootstrap,
    speaker_cluster_bootstrap,
)
from dmel_codec.tools.audit_dmel_range import histogram_quantiles
from dmel_codec.evaluation.summarize import summarize_records, write_summary
from dmel_codec.evaluation.run_codec_eval import (
    _cached_metric_value,
    _metric_cache_key,
    _ours_model_complexity,
    _result_seed,
    _validate_test_gate,
)
from dmel_codec.tools.make_paper_assets import (
    _dependency_mean,
    _include_in_main_table,
    _include_in_rate_quality,
    _is_ablation,
    _load_run,
    _rate_family,
    _rate_quality_runs,
    _result_summary_tex,
    _system_label,
    _validate_test_gate_coverage,
    generate_paper_assets,
)
from dmel_codec.tools.freeze_test_gate import freeze_test_gate
from dmel_codec.tools.pilot_gate import evaluate_pilot_gate


def _record(cut_id: str, speaker: str, value: float, substitutions: int) -> dict:
    counts = {
        "hits": 9,
        "substitutions": substitutions,
        "deletions": 0,
        "insertions": 0,
        "reference_words": 9 + substitutions,
    }
    return {
        "run_id": "test-run",
        "system": "Identity",
        "variant": "identity",
        "seed": 2026,
        "split": "dev-clean",
        "cut_id": cut_id,
        "speaker_id": speaker,
        "duration": 1.0,
        "config_hash": "abc",
        "checkpoint_hash": "checkpoint-abc",
        "nominal_bitrate_kbps": 0.0,
        "entropy_bitrate_kbps": 0.0,
        "serialized_bitrate_kbps": None,
        "token_frame_rate": 0.0,
        "num_parallel_streams": 0,
        "codebook_cardinalities": [],
        "encoder_seconds": 0.0,
        "decoder_seconds": 0.0,
        "metrics": {"stoi": value},
        "metric_failures": [],
        "failed": False,
        "asr": {"reference": counts, "reconstruction": counts},
    }


def test_ours_result_seed_comes_from_checkpoint_config() -> None:
    ours = object.__new__(OursAdapter)
    ours.config = OmegaConf.create({"seed": 2027})
    assert _result_seed(ours, analysis_seed=2026) == 2027
    assert _result_seed(object(), analysis_seed=2026) == 2026


def test_ours_complexity_excludes_training_discriminator_and_external_vocoder(
    tmp_path,
) -> None:
    class ToyCodec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)
            self.discriminator = torch.nn.Linear(2, 1)
            self.waveform_discriminator = torch.nn.Linear(3, 2)
            self.vocoder = torch.nn.Linear(4, 4)
            self.vocoder.requires_grad_(False)

    checkpoint = tmp_path / "training.ckpt"
    checkpoint.write_bytes(b"training-state")
    complexity = _ours_model_complexity(ToyCodec(), checkpoint)

    assert complexity == {
        "codec_parameters": 8,
        "codec_trainable_parameters": 8,
        "training_only_parameters": 11,
        "external_pretrained_parameters": 20,
        "training_checkpoint_size_bytes": 14,
        "training_checkpoint_includes_optimizer_state": True,
    }


def test_nominal_and_empirical_bitrate_exclude_padding() -> None:
    assert nominal_bitrate_kbps(75, [1024, 1024]) == 1.5
    accumulator = CodeStatisticsAccumulator([4, 4])
    codes = torch.tensor(
        [
            [[0, 1, 3], [1, 1, 3]],
            [[0, 1, 2], [0, 1, 2]],
        ]
    )
    accumulator.update(codes, torch.tensor([2, 3]))
    stats = accumulator.stream_statistics()
    assert accumulator.total_frames == 5
    assert stats[0].observations == 5
    assert stats[0].symbols_used == 3
    assert stats[1].symbols_used == 3
    assert accumulator.counts[0][3] == 0
    assert accumulator.counts[1][3] == 0
    assert accumulator.empirical_bitrate_kbps(10) < nominal_bitrate_kbps(10, [4, 4])


def test_dependency_nmi_and_probe_detect_copied_stream() -> None:
    accumulator = CodeDependencyAccumulator([4, 4, 4], sample_limit=400, seed=7)
    generator = torch.Generator().manual_seed(11)
    for cut in range(20):
        source = torch.randint(0, 4, (20,), generator=generator)
        independent = torch.randint(0, 4, (20,), generator=generator)
        accumulator.update(torch.stack((source, source, independent)), f"cut-{cut:02d}")
    result = accumulator.as_dict()

    assert result["observed_frames"] == 400
    assert result["sampled_frames"] == 400
    assert result["normalized_mutual_information_matrix"][0][1] > 0.95
    assert result["normalized_mutual_information_matrix"][0][2] < 0.1
    target = result["probe_results"][1]
    assert target["accuracy"] > 0.95
    assert target["accuracy_lift"] > 0.5


def test_dependency_probe_split_is_nonempty_for_two_frames() -> None:
    accumulator = CodeDependencyAccumulator([2, 2], sample_limit=2, seed=2026)
    accumulator.update(torch.tensor([[0, 1], [1, 0]]), "tiny-cut")

    result = accumulator.as_dict()

    assert result["sampled_frames"] == 2
    assert all(item["train_frames"] == 1 for item in result["probe_results"])
    assert all(item["test_frames"] == 1 for item in result["probe_results"])


def test_dmel_histogram_quantiles_are_deterministic() -> None:
    histogram = torch.tensor([1, 2, 4, 2, 1])

    result = histogram_quantiles(histogram, -2.0, 2.0, [0.0, 0.5, 1.0])

    assert result == {"0": -2.0, "0.5": 0.0, "1": 2.0}


def test_dependency_adjusted_nmi_handles_sparse_high_cardinality() -> None:
    generator = torch.Generator().manual_seed(2026)
    left = torch.randint(0, 1024, (50_000,), generator=generator)
    right = torch.randint(0, 1024, (50_000,), generator=generator)

    independent = CodeDependencyAccumulator._normalized_mutual_information(
        left, right, 1024, 1024
    )
    copied = CodeDependencyAccumulator._normalized_mutual_information(
        left, left, 1024, 1024
    )

    assert independent["plugin_normalized_mutual_information"] > 0.3
    assert independent["expected_mutual_information_bits"] > 3.0
    assert abs(independent["adjusted_normalized_mutual_information"]) < 0.01
    assert copied["adjusted_normalized_mutual_information"] > 0.999

    sparse_left = left.remainder(32)
    sparse_right = right.remainder(32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sparse = CodeDependencyAccumulator._normalized_mutual_information(
            sparse_left, sparse_right, 1024, 1024
        )
    assert abs(sparse["adjusted_normalized_mutual_information"]) < 0.01


def test_dependency_bottom_k_sample_is_update_order_independent() -> None:
    cuts = [
        (f"cut-{index}", torch.tensor([[index % 3] * 4, [index % 2] * 4]))
        for index in range(20)
    ]
    forward = CodeDependencyAccumulator([3, 2], sample_limit=25, seed=9)
    reverse = CodeDependencyAccumulator([3, 2], sample_limit=25, seed=9)
    for cut_id, codes in cuts:
        forward.update(codes, cut_id)
    for cut_id, codes in reversed(cuts):
        reverse.update(codes, cut_id)

    assert forward.as_dict() == reverse.as_dict()


def test_cache_key_covers_every_registered_identity_field(tmp_path) -> None:
    base = {
        "system_config_hash": "config-a",
        "checkpoint_hash": "checkpoint-a",
        "cut_id": "cut-a",
        "decode_seed_or_policy": "zeros",
    }
    key = reconstruction_cache_key(**base)
    for field in base:
        changed = dict(base)
        changed[field] += "-changed"
        assert reconstruction_cache_key(**changed) != key

    cache = ReconstructionCache(tmp_path)
    cache.save(key, {"waveform": torch.ones(1, 12), "audio_length": 12})
    loaded = cache.load(key)
    assert loaded is not None
    assert torch.equal(loaded["waveform"], torch.ones(1, 12))


def test_metric_cache_binds_source_metadata_and_caches_only_finite_successes(
    tmp_path,
) -> None:
    fields = {
        "metric": "pesq_wb",
        "metric_metadata_hash": "metadata-a",
        "metrics_source_hash": "source-a",
        "manifest_hash": "manifest-a",
        "cut_id": "cut-a",
        "reconstruction_key": "reconstruction-a",
    }
    key = _metric_cache_key(**fields)
    assert key != _metric_cache_key(**{**fields, "metrics_source_hash": "source-b"})
    assert key != _metric_cache_key(**{**fields, "metric_metadata_hash": "metadata-b"})
    assert key != _metric_cache_key(**{**fields, "reconstruction_key": "other"})

    cache = JsonCache(tmp_path / "metrics")
    calls = 0

    def compute() -> float:
        nonlocal calls
        calls += 1
        return 3.25

    assert _cached_metric_value(cache, key, compute) == 3.25
    assert _cached_metric_value(cache, key, compute) == 3.25
    assert calls == 1

    invalid_key = _metric_cache_key(**{**fields, "cut_id": "invalid"})
    with pytest.raises(ValueError, match="non-finite"):
        _cached_metric_value(cache, invalid_key, lambda: float("nan"))
    assert cache.load(invalid_key) is None


def test_corpus_wer_uses_global_edit_counts_not_utterance_mean() -> None:
    records = [_record("a", "1", 1.0, 1), _record("b", "2", 0.9, 0)]
    records[1]["asr"]["reference"] = {
        "hits": 1,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_words": 1,
    }
    assert corpus_wer(records, "reference") == pytest.approx(1 / 11)
    assert corpus_wer(list(reversed(records)), "reference") == pytest.approx(1 / 11)
    assert corpus_delta_wer(records) == pytest.approx(1 / 19 - 1 / 11)
    assert corpus_rtf(records, "encoder") == pytest.approx(0.0)


def test_summary_rebuild_and_cluster_bootstrap_are_deterministic(tmp_path) -> None:
    records = [
        _record("a", "speaker-a", 1.0, 0),
        _record("b", "speaker-a", 0.8, 1),
        _record("c", "speaker-b", 0.6, 2),
    ]
    summary = summarize_records(records)
    assert summary["utterances"] == 3
    assert summary["metrics"]["stoi"]["mean"] == pytest.approx(0.8)
    assert summary["corpus_wer"]["reference"] == pytest.approx(3 / 30)
    assert summary["checkpoint_hash"] == "checkpoint-abc"
    assert summary["token_frame_rate"] == 0.0
    assert summary["num_parallel_streams"] == 0

    first = speaker_cluster_bootstrap(records, metric="stoi", resamples=200, seed=7)
    second = speaker_cluster_bootstrap(records, metric="stoi", resamples=200, seed=7)
    assert first == second
    assert first["speakers"] == 2
    groups = {
        speaker: [record for record in records if record["speaker_id"] == speaker]
        for speaker in ("speaker-a", "speaker-b")
    }
    generator = np.random.default_rng(7)
    slow_estimates = []
    for _ in range(200):
        chosen = generator.choice(sorted(groups), size=len(groups), replace=True)
        sampled = [record for speaker in chosen for record in groups[str(speaker)]]
        slow_estimates.append(
            np.mean([record["metrics"]["stoi"] for record in sampled])
        )
    assert first["ci95"] == pytest.approx(np.quantile(slow_estimates, (0.025, 0.975)))
    delta = speaker_cluster_bootstrap(
        records, metric="wer_delta", resamples=200, seed=7
    )
    assert delta["point_estimate"] == pytest.approx(0.0)
    decoder_rtf = speaker_cluster_bootstrap(
        records, metric="decoder_rtf", resamples=200, seed=7
    )
    assert decoder_rtf["point_estimate"] == pytest.approx(0.0)

    right = [
        dict(record, metrics={"stoi": record["metrics"]["stoi"] - 0.1})
        for record in records
    ]
    paired = paired_speaker_cluster_bootstrap(
        records, right, metric="stoi", resamples=200, seed=7
    )
    assert paired["point_estimate"] == pytest.approx(0.1)
    assert paired["paired_cuts"] == 3
    assert paired["ci95"] == pytest.approx((0.1, 0.1))

    records_path = tmp_path / "per_utterance.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "summary.json"
    observed = write_summary(records_path, output)
    assert json.loads(output.read_text(encoding="utf-8")) == observed
    assert len(observed["per_utterance_sha256"]) == 64
    int(observed["per_utterance_sha256"], 16)


def test_summary_rejects_duplicate_cuts() -> None:
    records = [_record("same", "1", 1.0, 0), _record("same", "1", 1.0, 0)]
    with pytest.raises(ValueError, match="Duplicate cut IDs"):
        summarize_records(records)


def test_summary_counts_metric_failures_without_dropping_the_utterance() -> None:
    record = _record("failed-metric", "speaker-a", 1.0, 0)
    record["metrics"] = {"stoi": None}
    record["metric_failures"] = [
        {"metric": "stoi", "error_type": "RuntimeError", "message": "expected"}
    ]
    summary = summarize_records([record])
    assert summary["utterances"] == 1
    assert summary["failed_utterances"] == 0
    assert summary["metrics"]["stoi"] == {
        "count": 0,
        "failures": 1,
        "mean": None,
        "median": None,
    }
    assert summary["failure_reasons"] == {"stoi:RuntimeError": 1}


def test_paper_assets_are_deterministic_and_reject_edited_summary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [
        _record("a", "speaker-a", 1.0, 0),
        _record("b", "speaker-b", 0.8, 1),
    ]
    records_path = run_dir / "per_utterance.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    write_summary(records_path, run_dir / "summary.json")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    output = tmp_path / "paper"

    first = generate_paper_assets(
        [run_dir],
        output_directory=output,
        protocol=protocol,
        allow_incomplete=True,
    )
    second = generate_paper_assets(
        [run_dir],
        output_directory=output,
        protocol=protocol,
        allow_incomplete=True,
    )
    assert first == second
    assert first["schema_version"] == 2
    assert first["status"] == "PAPER_ASSETS_DIAGNOSTIC"
    assert first["test_gate"] is None
    assert set(first["generated"]) == {
        "artifacts/ablations.csv",
        "artifacts/main_results.csv",
        "artifacts/seed_aggregates.json",
        "figures/code_dependency.pdf",
        "figures/method.pdf",
        "figures/rate_quality.pdf",
        "tables/ablations.tex",
        "tables/main_results.tex",
        "tables/results_summary.tex",
    }
    assert (output / "artifacts/result_manifest.json").is_file()
    assert (output / "tables/results_summary.tex").read_text(encoding="utf-8") == (
        "% Frozen result prose requires a schema-v3 test gate.\n"
    )

    summary_path = run_dir / "summary.json"
    edited = json.loads(summary_path.read_text(encoding="utf-8"))
    edited["metrics"]["stoi"]["mean"] = 0.123
    summary_path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly rebuild"):
        generate_paper_assets(
            [run_dir],
            output_directory=output,
            protocol=protocol,
            allow_incomplete=True,
        )


def test_paper_assets_aggregate_training_seeds_without_hiding_raw_runs(
    tmp_path,
) -> None:
    run_directories = []
    for seed, visqol, substitutions in ((2026, 4.0, 0), (2027, 4.2, 1)):
        run_dir = tmp_path / f"seed-{seed}"
        run_dir.mkdir()
        records = []
        for cut_id, speaker in (("a", "speaker-a"), ("b", "speaker-b")):
            record = _record(cut_id, speaker, visqol, substitutions)
            record.update(
                {
                    "run_id": f"ours-m-seed-{seed}",
                    "system": "Ours-M",
                    "variant": "ours",
                    "seed": seed,
                    "nominal_bitrate_kbps": 1.746,
                    "metrics": {
                        "visqol_speech": visqol,
                        "pesq_wb": 2.0 + 0.1 * (seed - 2026),
                        "speaker_cosine": 0.7 + 0.01 * (seed - 2026),
                    },
                    "model_complexity": {
                        "codec_parameters": 10,
                        "external_pretrained_parameters": 20,
                    },
                }
            )
            record["asr"]["reference"] = {
                **record["asr"]["reference"],
                "hits": 9,
                "substitutions": 0,
                "reference_words": 9,
            }
            records.append(record)
        records_path = run_dir / "per_utterance.jsonl"
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        write_summary(records_path, run_dir / "summary.json")
        run_directories.append(run_dir)

    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    output = tmp_path / "paper"
    manifest = generate_paper_assets(
        run_directories,
        output_directory=output,
        protocol=protocol,
        allow_incomplete=True,
    )

    table = (output / "tables/main_results.tex").read_text(encoding="utf-8")
    assert table.count("Ours-M") == 1
    assert r"\mathbin{\pm}" in table
    assert r"\begin{tabular*}{\textwidth}" in table
    assert r"\resizebox" not in table
    assert "Seeds & kbps & Hz" in table
    assert "Enc./Dec. RTF" not in table
    assert "Codec M" not in table
    assert "Ext. M" not in table
    aggregate = json.loads(
        (output / "artifacts/seed_aggregates.json").read_text(encoding="utf-8")
    )["operating_points"][0]
    assert aggregate["seed_count"] == 2
    assert aggregate["seeds"] == [2026, 2027]
    assert aggregate["metrics"]["visqol_speech"]["mean"] == pytest.approx(4.1)
    assert aggregate["metrics"]["visqol_speech"]["sample_std"] == pytest.approx(
        0.1414213562
    )
    csv_text = (output / "artifacts/main_results.csv").read_text(encoding="utf-8")
    assert csv_text.count("ours-m-seed-") == 2
    assert len(manifest["runs"]) == 2


@pytest.mark.parametrize(
    ("streams", "expected"),
    ((1, False), (2, True), (3, True), (4, False)),
)
def test_main_table_keeps_only_rate_bracketing_dac_points(
    streams: int, expected: bool
) -> None:
    run = argparse.Namespace(
        summary={"variant": "dac", "num_parallel_streams": streams}
    )
    assert _include_in_main_table(run) is expected


@pytest.mark.parametrize(
    ("system", "variant", "expected_main", "expected_rate"),
    (
        ("Ground Truth", "identity", False, False),
        ("dMel-same-front-end (adapted)", "dmel", True, False),
        ("Ours-L", "ours", True, True),
        ("Ours-M", "ours", True, True),
        ("Ours-H", "ours", False, True),
    ),
)
def test_diagnostic_paper_display_keeps_the_low_rate_focus(
    system: str,
    variant: str,
    expected_main: bool,
    expected_rate: bool,
) -> None:
    run = argparse.Namespace(summary={"system": system, "variant": variant})
    assert _include_in_main_table(run) is expected_main
    assert _include_in_rate_quality(run) is expected_rate


def test_diagnostic_rate_curve_replaces_base_m_with_promoted_mplus() -> None:
    runs = [
        argparse.Namespace(
            summary={"system": system, "variant": "ours", "split": split}
        )
        for split in ("test-clean", "test-other")
        for system in ("Ours-L", "Ours-M", "Ours-M+ (joint)")
    ]
    selected = _rate_quality_runs(
        runs,
        role_lookup=None,
        display_roles=None,
    )
    systems_by_split = {
        split: {
            run.summary["system"] for run in selected if run.summary["split"] == split
        }
        for split in ("test-clean", "test-other")
    }
    assert systems_by_split == {
        "test-clean": {"Ours-L", "Ours-M+ (joint)"},
        "test-other": {"Ours-L", "Ours-M+ (joint)"},
    }


def test_frozen_roles_override_diagnostic_name_filters() -> None:
    h_run = argparse.Namespace(summary={"system": "Ours-H", "variant": "ours"})
    assert _include_in_main_table(
        h_run,
        run_roles={"ours_h_seed2026"},
        display_roles={"ours_h_seed2026"},
    )
    assert not _include_in_rate_quality(
        h_run,
        run_roles={"ours_h_seed2026"},
        display_roles={"flagship_m_seed2026"},
    )


def test_dmel_decoder_is_labeled_as_libritts_trained_and_plotted_separately() -> None:
    direct = {
        "system": "dMel-same-front-end (adapted)",
        "model_complexity": {"codec_parameters": 0},
    }
    decoder = {
        "system": "dMel-quantization-aware-decoder (adapted)",
        "model_complexity": {"codec_parameters": 6_421_092},
    }

    assert _system_label(direct) == r"dMel adapted$^{A,E}$"
    assert _system_label(decoder) == r"dMel adapted + dec.$^{L,A,E}$"
    assert _rate_family(direct["system"]) == ("Adapted dMel", "#009e73", "D")
    assert _rate_family(decoder["system"]) == (
        "Adapted dMel + decoder",
        "#009e73",
        "P",
    )


def test_promoted_joint_mplus_is_not_classified_as_fullband_ablation() -> None:
    promoted = argparse.Namespace(
        summary={
            "system": "Ours-M+ (full codec + full vocoder + waveform GAN)",
            "run_id": "mplus-joint-final",
        }
    )
    fullband = argparse.Namespace(
        summary={"system": "Ours-M joint full-band", "run_id": "ablation"}
    )
    random_groups = argparse.Namespace(
        summary={"system": "Ours-M random groups", "run_id": "ablation"}
    )

    assert _system_label(promoted.summary) == r"Ours-M+ joint$^{L,E}$"
    assert _is_ablation(promoted) is False
    assert _is_ablation(fullband) is True
    assert _is_ablation(random_groups) is True


def test_dependency_table_prefers_explicit_adjusted_nmi_and_reports_probe_lift() -> (
    None
):
    run = argparse.Namespace(
        dependency={
            "pairs": [
                {
                    "normalized_mutual_information": 0.91,
                    "adjusted_normalized_mutual_information": 0.03,
                },
                {
                    "normalized_mutual_information": 0.81,
                    "adjusted_normalized_mutual_information": 0.05,
                },
            ],
            "probe_results": [
                {"accuracy_lift": 0.10},
                {"accuracy_lift": 0.14},
            ],
        }
    )

    assert _dependency_mean(run, "adjusted_nmi") == pytest.approx(0.04)
    assert _dependency_mean(run, "probe_lift") == pytest.approx(0.12)


def _frozen_result_summary_inputs() -> tuple[list[argparse.Namespace], dict]:
    specs = [
        ("mel_oracle", "mel_oracle", "Mel Oracle", "mel_oracle", 2026, 0.0),
        (
            "dmel_same_frontend",
            "dmel",
            "dMel-same-front-end (adapted)",
            "dmel",
            2026,
            37.5,
        ),
        (
            "dmel_quantization_aware_decoder",
            "dmel_decoder",
            "dMel-quantization-aware-decoder (adapted)",
            "dmel_decoder",
            2026,
            37.5,
        ),
        ("dac_q1", "dac", "DAC 24 kHz q=1", "dac", 2026, 0.75),
        ("dac_q2", "dac", "DAC 24 kHz q=2", "dac", 2026, 1.50),
        ("dac_q3", "dac", "DAC 24 kHz q=3", "dac", 2026, 2.25),
        ("ours_075_seed2026", "ours", "Ours-0.75", "ours", 2026, 0.743),
        ("base_m_seed2026", "ours", "Ours-M", "ours", 2026, 1.746),
        (
            "joint_fullband_seed2026",
            "ours",
            "Ours-M joint full-band",
            "ours",
            2026,
            1.746,
        ),
        (
            "random_groups_seed2026",
            "ours",
            "Ours-M random groups",
            "ours",
            2026,
            1.746,
        ),
        *[
            (
                f"flagship_m_seed{seed}",
                "ours",
                "Ours-M+ (full codec + full vocoder + waveform GAN)",
                "ours",
                seed,
                1.746,
            )
            for seed in (2026, 2027, 2028)
        ],
    ]
    runs = []
    role_lookup = {}
    for split_index, split in enumerate(("test-clean", "test-other")):
        for role_index, (role, adapter, system, variant, seed, bitrate) in enumerate(
            specs
        ):
            config_hash = f"config-{role}-{seed}"
            checkpoint_hash = f"checkpoint-{role}-{seed}"
            wer = 0.02 + 0.002 * split_index + 0.0001 * role_index
            summary = {
                "run_id": f"{role}-{split}",
                "system": system,
                "variant": variant,
                "seed": seed,
                "split": split,
                "config_hash": config_hash,
                "checkpoint_hash": checkpoint_hash,
                "bitrate_kbps": {"nominal": bitrate, "entropy": bitrate * 0.8},
                "token_frame_rate": 75.0 if adapter == "dac" else 23.4375,
                "corpus_wer": {"reconstruction": wer, "delta": wer - 0.01},
                "metrics": {
                    "visqol_speech": {"mean": 4.0 + 0.001 * role_index},
                    "pesq_wb": {"mean": 2.0 + 0.001 * role_index},
                    "speaker_cosine": {"mean": 0.7 + 0.001 * role_index},
                },
                "model_complexity": {
                    "codec_parameters": (
                        6_421_092 if role == "dmel_quantization_aware_decoder" else 0
                    )
                },
            }
            resolved_config = {
                "system": adapter,
                "decode_seed_or_policy": "frozen-policy",
            }
            run = argparse.Namespace(
                summary=summary,
                resolved_config=resolved_config,
                dependency={
                    "pairs": [{"adjusted_normalized_mutual_information": 0.01}],
                    "probe_results": [{"accuracy_lift": 0.02}],
                },
            )
            identity = (
                adapter,
                system,
                variant,
                seed,
                config_hash,
                checkpoint_hash,
                "frozen-policy",
            )
            role_lookup[identity] = {role}
            runs.append(run)
    return runs, role_lookup


def test_frozen_result_summary_covers_rates_controls_and_dependencies() -> None:
    runs, role_lookup = _frozen_result_summary_inputs()

    result = _result_summary_tex(runs, role_lookup)
    assert result.startswith("% Generated from frozen test runs.")
    assert r"\paragraph{Rate--quality.}" in result
    assert r"Ours-M+ joint$^{L,E}$" in result
    assert r"\mathbin{\pm}" in result
    assert "test-clean/test-other" in result
    assert "At the 0.743-kbps boundary" in result
    assert "These controls are not rate-matched" in result
    assert "a lower value alone is not evidence" in result
    assert "dev-clean" not in result

    with pytest.raises(ValueError, match="dac_q1"):
        _result_summary_tex(
            [run for run in runs if run.summary["run_id"] != "dac_q1-test-other"],
            role_lookup,
        )


def test_pilot_gate_requires_two_completed_repairs_before_no_go(tmp_path) -> None:
    def make_run(
        name: str,
        system: str,
        bitrate: float,
        substitutions: int,
        quality: tuple[float, float, float],
        domain_failure: bool = False,
    ):
        directory = tmp_path / name
        directory.mkdir()
        records = []
        for cut_id, speaker in (("a", "speaker-a"), ("b", "speaker-b")):
            record = _record(cut_id, speaker, quality[0], substitutions)
            record.update(
                {
                    "run_id": name,
                    "system": system,
                    "variant": name,
                    "nominal_bitrate_kbps": bitrate,
                    "metrics": {
                        "visqol_speech": quality[0],
                        "pesq_wb": quality[1],
                        "speaker_cosine": quality[2],
                    },
                }
            )
            record["asr"]["reference"] = {
                **record["asr"]["reference"],
                "substitutions": 0,
                "reference_words": 9,
            }
            records.append(record)
        if domain_failure:
            records[0]["duration"] = 0.5
            records[0]["metrics"]["visqol_speech"] = None
            records[0]["metric_failures"] = [
                {
                    "metric": "visqol_speech",
                    "error_type": "MetricDomainError",
                    "message": "ViSQOL speech requires at least 0.64 s",
                },
                {
                    "metric": "wer_reference",
                    "error_type": "MetricDomainError",
                    "message": "ASR reference is empty after text normalization",
                },
                {
                    "metric": "wer_reconstruction",
                    "error_type": "MetricDomainError",
                    "message": "ASR reference is empty after text normalization",
                },
            ]
        records_path = directory / "per_utterance.jsonl"
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        write_summary(records_path, directory / "summary.json")
        return directory

    ours = make_run("ours", "Ours-M", 1.7463776043357018, 3, (2.0, 1.1, 0.5), True)
    q2 = make_run("q2", "DAC q=2", 1.5, 1, (3.5, 1.6, 0.7), True)
    q3 = make_run("q3", "DAC q=3", 2.25, 0, (4.0, 2.0, 0.8), True)

    repair = evaluate_pilot_gate(ours, q2, q3, repair_rounds=0)
    assert repair["strict_quality_dominated_by_both_dac_points"] is True
    assert repair["execplan_no_go_condition_met"] is False
    assert repair["disposition"] == "controlled_repair"
    assert repair["metric_domain_exclusions"] == {"visqol_speech": ["a"]}

    no_go = evaluate_pilot_gate(ours, q2, q3, repair_rounds=2)
    assert no_go["execplan_no_go_condition_met"] is True
    assert no_go["disposition"] == "escalate_no_go"

    q3_mismatch = make_run("q3-mismatch", "DAC q=3", 2.25, 0, (4.0, 2.0, 0.8))
    with pytest.raises(ValueError, match="different metric-domain exclusion sets"):
        evaluate_pilot_gate(ours, q2, q3_mismatch, repair_rounds=0)


def test_test_split_requires_an_untruncated_frozen_gate(tmp_path) -> None:
    missing_gate = argparse.Namespace(split="test-clean", max_cuts=None, test_gate=None)
    with pytest.raises(FileNotFoundError, match="--test-gate"):
        _validate_test_gate(missing_gate)

    gate = tmp_path / "TEST_GATE_FROZEN"
    gate.write_text("frozen\n", encoding="utf-8")
    _validate_test_gate(
        argparse.Namespace(split="test-other", max_cuts=None, test_gate=gate)
    )
    with pytest.raises(ValueError, match="--max-cuts"):
        _validate_test_gate(
            argparse.Namespace(split="test-clean", max_cuts=1, test_gate=gate)
        )


def test_frozen_gate_binds_system_metrics_manifest_and_sources(
    tmp_path, monkeypatch
) -> None:
    def write_dev_run(
        split: str,
        *,
        adapter: str = "identity",
        system: str = "Identity",
        config_hash: str = "abc",
        cut_ids: tuple[str, str] = ("a", "b"),
    ) -> Path:
        run_dir = tmp_path / f"{adapter}-{split}"
        run_dir.mkdir()
        records = [
            _record(cut_ids[0], "speaker-a", 1.0, 0),
            _record(cut_ids[1], "speaker-b", 0.8, 1),
        ]
        for record in records:
            record["split"] = split
            record["run_id"] = f"{adapter}-{split}"
            record["system"] = system
            record["variant"] = adapter
            record["config_hash"] = config_hash
            record["checkpoint_hash"] = f"checkpoint-{config_hash}"
        records_path = run_dir / "per_utterance.jsonl"
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        write_summary(records_path, run_dir / "summary.json")
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "system": adapter,
                    "system_config_hash": config_hash,
                    "decode_seed_or_policy": adapter,
                    "metrics": ["stoi"],
                    "manifest_sha256": {
                        "test-clean": "clean-hash",
                        "test-other": "other-hash",
                    }.get(split),
                    "max_cuts": None,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "model_metadata.json").write_text(
            json.dumps({"metrics": {"revision": "metric-v1"}}), encoding="utf-8"
        )
        (run_dir / "data_manifest.json").write_text(
            json.dumps({"cuts": 2}), encoding="utf-8"
        )
        (run_dir / "git_state.json").write_text(
            json.dumps({"commit": "deadbeef", "dirty": False}), encoding="utf-8"
        )
        return run_dir

    def write_requirements(roles: dict, name: str = "requirements.yaml") -> Path:
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "paper_display": {
                        "schema_version": 1,
                        "main_table_roles": list(roles),
                        "ablation_table_roles": list(roles),
                        "rate_quality_roles": list(roles),
                    },
                    "roles": roles,
                }
            ),
            encoding="utf-8",
        )
        return path

    def write_registry(entries: list[dict], name: str) -> Path:
        path = tmp_path / name
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "status": "FINAL_SYSTEMS_PREREGISTERED",
                    "systems": entries,
                }
            ),
            encoding="utf-8",
        )
        return path

    identity_requirement = {
        "adapter": "identity",
        "systems": ["Identity"],
        "variant": "identity",
        "seed": 2026,
        "nominal_bitrate_kbps": 0.0,
        "token_frame_rate_hz": 0.0,
        "num_parallel_streams": 0,
    }

    clean_run = write_dev_run("dev-clean")
    other_run = write_dev_run("dev-other")
    manifest_summary = tmp_path / "manifest_summary.json"
    manifest_summary.write_text(
        json.dumps(
            {
                "manifests": {
                    "test-clean_eval.jsonl.gz": {"sha256": "clean-hash"},
                    "test-other_eval.jsonl.gz": {"sha256": "other-hash"},
                }
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen\n", encoding="utf-8")
    gate_path = tmp_path / "TEST_GATE_FROZEN"
    repository_root = Path(__file__).resolve().parents[1]
    requirements_path = write_requirements({"identity": identity_requirement})
    monkeypatch.setattr(
        "dmel_codec.tools.freeze_test_gate.DEFAULT_SYSTEM_REQUIREMENTS",
        requirements_path,
    )
    registry_path = write_registry(
        [
            {
                "role": "identity",
                "dev_clean_run": str(clean_run.resolve()),
                "dev_other_run": str(other_run.resolve()),
            }
        ],
        "identity-registry.yaml",
    )
    gate = freeze_test_gate(
        system_registry=registry_path,
        manifest_summary=manifest_summary,
        protocol=protocol,
        output=gate_path,
        metrics=["stoi"],
        repository_root=repository_root,
        allow_dirty=True,
    )
    assert gate["schema_version"] == 3
    assert gate["status"] == "TEST_GATE_DIAGNOSTIC"
    assert gate["dev_protocol"] == {
        "selection_split": "dev-clean",
        "robustness_split": "dev-other",
    }
    assert gate["systems"][0]["selection_run_id"] == "identity-dev-clean"
    assert gate["systems"][0]["robustness"]["run_id"] == "identity-dev-other"
    assert gate["systems"][0]["roles"] == ["identity"]
    assert gate["final_system_registry"]["roles"] == ["identity"]
    assert gate["final_system_registry"]["paper_display"] == {
        "schema_version": 1,
        "main_table_roles": ["identity"],
        "ablation_table_roles": ["identity"],
        "rate_quality_roles": ["identity"],
    }
    assert "dmel_codec/tools/freeze_test_gate.py" in gate["source_hashes"]

    dac_clean = write_dev_run(
        "dev-clean",
        adapter="dac",
        system="DAC q=2",
        config_hash="dac-config",
        cut_ids=("a", "different"),
    )
    dac_other = write_dev_run(
        "dev-other",
        adapter="dac",
        system="DAC q=2",
        config_hash="dac-config",
    )
    dac_requirement = {
        **identity_requirement,
        "adapter": "dac",
        "systems": ["DAC q=2"],
        "variant": "dac",
    }
    mismatch_requirements = write_requirements(
        {"identity": identity_requirement, "dac_q2": dac_requirement},
        "mismatch-requirements.yaml",
    )
    monkeypatch.setattr(
        "dmel_codec.tools.freeze_test_gate.DEFAULT_SYSTEM_REQUIREMENTS",
        mismatch_requirements,
    )
    mismatch_registry = write_registry(
        [
            {
                "role": "identity",
                "dev_clean_run": str(clean_run.resolve()),
                "dev_other_run": str(other_run.resolve()),
            },
            {
                "role": "dac_q2",
                "dev_clean_run": str(dac_clean.resolve()),
                "dev_other_run": str(dac_other.resolve()),
            },
        ],
        "mismatch-registry.yaml",
    )
    with pytest.raises(ValueError, match="dev-clean runs use different cut IDs"):
        freeze_test_gate(
            system_registry=mismatch_registry,
            manifest_summary=manifest_summary,
            protocol=protocol,
            output=tmp_path / "cut-mismatch-gate.json",
            metrics=["stoi"],
            repository_root=repository_root,
            allow_dirty=True,
        )

    omitted_registry = write_registry(
        [
            {
                "role": "identity",
                "dev_clean_run": str(clean_run.resolve()),
                "dev_other_run": str(other_run.resolve()),
            }
        ],
        "omitted-registry.yaml",
    )
    with pytest.raises(ValueError, match="role mismatch"):
        freeze_test_gate(
            system_registry=omitted_registry,
            manifest_summary=manifest_summary,
            protocol=protocol,
            output=tmp_path / "single-domain-gate.json",
            metrics=["stoi"],
            repository_root=repository_root,
            allow_dirty=True,
        )

    wrong_role_registry = write_registry(
        [
            {
                "role": "identity",
                "dev_clean_run": str(clean_run.resolve()),
                "dev_other_run": str(other_run.resolve()),
            },
            {
                "role": "dac_q2",
                "dev_clean_run": str(clean_run.resolve()),
                "dev_other_run": str(other_run.resolve()),
            },
        ],
        "wrong-role-registry.yaml",
    )
    with pytest.raises(ValueError, match="dac_q2 has unexpected adapter"):
        freeze_test_gate(
            system_registry=wrong_role_registry,
            manifest_summary=manifest_summary,
            protocol=protocol,
            output=tmp_path / "wrong-role-gate.json",
            metrics=["stoi"],
            repository_root=repository_root,
            allow_dirty=True,
        )

    wrong_domain_registry = write_registry(
        [
            {
                "role": "identity",
                "dev_clean_run": str(clean_run.resolve()),
                "dev_other_run": str(clean_run.resolve()),
            }
        ],
        "wrong-domain-registry.yaml",
    )
    monkeypatch.setattr(
        "dmel_codec.tools.freeze_test_gate.DEFAULT_SYSTEM_REQUIREMENTS",
        requirements_path,
    )
    with pytest.raises(ValueError, match="must be dev-other"):
        freeze_test_gate(
            system_registry=wrong_domain_registry,
            manifest_summary=manifest_summary,
            protocol=protocol,
            output=tmp_path / "single-domain-gate.json",
            metrics=["stoi"],
            repository_root=repository_root,
            allow_dirty=True,
        )

    with pytest.raises(ValueError, match="commit does not match"):
        freeze_test_gate(
            system_registry=registry_path,
            manifest_summary=manifest_summary,
            protocol=protocol,
            output=tmp_path / "invalid-commit-gate.json",
            metrics=["stoi"],
            repository_root=repository_root,
        )

    diagnostic_args = argparse.Namespace(
        split="test-clean",
        max_cuts=None,
        test_gate=gate_path,
        system="identity",
        seed=2026,
    )
    with pytest.raises(ValueError, match="Invalid or incomplete frozen test gate"):
        _validate_test_gate(
            diagnostic_args,
            metrics={"stoi"},
            manifest_hash="clean-hash",
            config_hash="abc",
            checkpoint_hash="checkpoint-abc",
            metric_metadata_hash=gate["metric_metadata_hash"],
            system_name="Identity",
            decode_seed_or_policy="identity",
        )

    formal_gate_path = tmp_path / "formal-test-gate.json"

    def clean_git_output(command, **_):
        if command[1:] == ["rev-parse", "HEAD"]:
            return "deadbeef\n"
        if command[1:] == ["status", "--short"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(
        "dmel_codec.tools.freeze_test_gate.subprocess.check_output",
        clean_git_output,
    )
    formal_gate = freeze_test_gate(
        system_registry=registry_path,
        manifest_summary=manifest_summary,
        protocol=protocol,
        output=formal_gate_path,
        metrics=["stoi"],
        repository_root=repository_root,
    )
    assert formal_gate["status"] == "TEST_GATE_FROZEN"
    assert formal_gate["allow_dirty_dev_runs"] is False
    monkeypatch.setattr(
        "dmel_codec.evaluation.run_codec_eval._git_state",
        lambda _: {"commit": "deadbeef", "dirty": False},
    )
    args = argparse.Namespace(
        split="test-clean",
        max_cuts=None,
        test_gate=formal_gate_path,
        system="identity",
        seed=2026,
    )
    observed_gate_hash = _validate_test_gate(
        args,
        metrics={"stoi"},
        manifest_hash="clean-hash",
        config_hash="abc",
        checkpoint_hash="checkpoint-abc",
        metric_metadata_hash=gate["metric_metadata_hash"],
        system_name="Identity",
        decode_seed_or_policy="identity",
    )
    assert observed_gate_hash == sha256_file(formal_gate_path)
    test_clean_run = write_dev_run("test-clean")
    test_other_run = write_dev_run("test-other")
    for test_run in (test_clean_run, test_other_run):
        resolved_path = test_run / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved["test_gate_sha256"] = observed_gate_hash
        resolved_path.write_text(
            yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
        )
    paper_runs = [
        _load_run(test_clean_run, allow_incomplete=True),
        _load_run(test_other_run, allow_incomplete=True),
    ]
    paper_gate = _validate_test_gate_coverage(paper_runs, formal_gate_path)
    assert paper_gate["status"] == "TEST_GATE_FROZEN"
    assert paper_gate["roles"] == ["identity"]
    assert paper_gate["paper_display"]["main_table_roles"] == ["identity"]
    with pytest.raises(ValueError, match="do not exactly cover"):
        _validate_test_gate_coverage(paper_runs[:1], formal_gate_path)

    mismatched_gate_config = yaml.safe_load(
        (test_other_run / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    mismatched_gate_config["test_gate_sha256"] = "0" * 64
    (test_other_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(mismatched_gate_config, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not bind the frozen test gate"):
        _validate_test_gate_coverage(
            [paper_runs[0], _load_run(test_other_run, allow_incomplete=True)],
            formal_gate_path,
        )
    mismatched_gate_config["test_gate_sha256"] = observed_gate_hash
    (test_other_run / "resolved_config.yaml").write_text(
        yaml.safe_dump(mismatched_gate_config, sort_keys=True), encoding="utf-8"
    )

    (test_other_run / "model_metadata.json").write_text(
        json.dumps({"metrics": {"revision": "tampered"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metric revisions differ"):
        _validate_test_gate_coverage(
            [paper_runs[0], _load_run(test_other_run, allow_incomplete=True)],
            formal_gate_path,
        )
    missing_display_gate = json.loads(formal_gate_path.read_text(encoding="utf-8"))
    del missing_display_gate["final_system_registry"]["paper_display"]
    missing_display_path = tmp_path / "missing-display-gate.json"
    missing_display_path.write_text(json.dumps(missing_display_gate), encoding="utf-8")
    with pytest.raises(ValueError, match="paper-display schema v1"):
        _validate_test_gate(
            argparse.Namespace(
                split="test-clean",
                max_cuts=None,
                test_gate=missing_display_path,
                system="identity",
                seed=2026,
            ),
            metrics={"stoi"},
            manifest_hash="clean-hash",
            config_hash="abc",
            checkpoint_hash="checkpoint-abc",
            metric_metadata_hash=gate["metric_metadata_hash"],
            system_name="Identity",
            decode_seed_or_policy="identity",
        )
    tampered_gate = json.loads(formal_gate_path.read_text(encoding="utf-8"))
    tampered_gate["systems"][0]["roles"] = []
    tampered_gate_path = tmp_path / "tampered-role-gate.json"
    tampered_gate_path.write_text(json.dumps(tampered_gate), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent final-system coverage"):
        _validate_test_gate(
            argparse.Namespace(
                split="test-clean",
                max_cuts=None,
                test_gate=tampered_gate_path,
                system="identity",
                seed=2026,
            ),
            metrics={"stoi"},
            manifest_hash="clean-hash",
            config_hash="abc",
            checkpoint_hash="checkpoint-abc",
            metric_metadata_hash=gate["metric_metadata_hash"],
            system_name="Identity",
            decode_seed_or_policy="identity",
        )
    with pytest.raises(ValueError, match="manifest hash"):
        _validate_test_gate(
            args,
            metrics={"stoi"},
            manifest_hash="wrong-hash",
            config_hash="abc",
            checkpoint_hash="checkpoint-abc",
            metric_metadata_hash=gate["metric_metadata_hash"],
            system_name="Identity",
            decode_seed_or_policy="identity",
        )

    with pytest.raises(ValueError, match="not uniquely registered"):
        _validate_test_gate(
            args,
            metrics={"stoi"},
            manifest_hash="clean-hash",
            config_hash="abc",
            checkpoint_hash="checkpoint-abc",
            metric_metadata_hash=gate["metric_metadata_hash"],
            system_name="Identity",
            decode_seed_or_policy="changed-policy",
        )
