from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

import pytest
import yaml

from dmel_codec.tools import test_queue as test_queue_module
from dmel_codec.evaluation.cache import canonical_hash
from dmel_codec.evaluation.summarize import write_summary
from dmel_codec.tools.test_queue import (
    PRIMARY_BOOTSTRAP_METRICS,
    prepare_test_queue,
    run_test_queue,
    verify_task,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path, content: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        "dmel_codec.tools.test_queue._current_git",
        lambda _: {"commit": "deadbeef", "dirty": False},
    )
    run_root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    manifests = run_root / "manifests/libritts24k"
    manifests.mkdir(parents=True)
    cache_root.mkdir()
    clean_manifest = manifests / "test-clean_eval.jsonl.gz"
    other_manifest = manifests / "test-other_eval.jsonl.gz"
    for manifest in (clean_manifest, other_manifest):
        with gzip.open(manifest, mode="wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": "synthetic-cut"}) + "\n")

    assets_root = tmp_path / "assets"
    shared = {
        "bigvgan_checkpoint": _asset(assets_root / "bigvgan.pt", b"bigvgan"),
        "bigvgan_config": _asset(assets_root / "bigvgan.json", b"{}"),
        "dac_checkpoint": _asset(assets_root / "dac.pth", b"dac"),
        "asr_model_metadata": _asset(assets_root / "asr.json", b"{}"),
        "speaker_model_metadata": _asset(assets_root / "speaker.json", b"{}"),
        "visqol_metadata": _asset(assets_root / "visqol.json", b"{}"),
    }
    ours_checkpoint = _asset(assets_root / "ours.ckpt", b"ours")
    asset_manifest = tmp_path / "FINAL_TEST_ASSETS_RESOLVED.yaml"
    asset_manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "FINAL_TEST_ASSETS_RESOLVED",
                "shared": shared,
                "ours_checkpoints": {
                    "base_m_seed2026": ours_checkpoint,
                    "flagship_m_seed2026": ours_checkpoint,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics = ["pesq_wb", "speaker_cosine", "visqol_speech", "wer"]
    metric_metadata = {"synthetic": "metric-assets"}
    systems = [
        {
            "adapter": "identity",
            "system": "Ground Truth",
            "variant": "identity",
            "seed": 2026,
            "config_hash": "config-identity",
            "checkpoint_hash": "checkpoint-identity",
            "raw_checkpoint_sha256": None,
            "decode_seed_or_policy": "deterministic-v1",
            "num_parallel_streams": 0,
            "roles": ["ground_truth"],
        },
        {
            "adapter": "dac",
            "system": "DAC 24 kHz q=2",
            "variant": "dac",
            "seed": 2026,
            "config_hash": "config-dac",
            "checkpoint_hash": "checkpoint-dac",
            "raw_checkpoint_sha256": shared["dac_checkpoint"]["sha256"],
            "decode_seed_or_policy": "deterministic-v1",
            "num_parallel_streams": 2,
            "roles": ["dac_q2"],
        },
        {
            "adapter": "ours",
            "system": "Ours-M",
            "variant": "ours",
            "seed": 2026,
            "config_hash": "config-ours",
            "checkpoint_hash": "checkpoint-ours",
            "raw_checkpoint_sha256": ours_checkpoint["sha256"],
            "decode_seed_or_policy": "zeros",
            "num_parallel_streams": 10,
            "roles": ["base_m_seed2026", "flagship_m_seed2026"],
        },
    ]
    source_paths = (
        "dmel_codec/tools/test_queue.py",
        "scripts/evaluate.sh",
        "scripts/run_test_queue.sh",
        "scripts/statistics.sh",
    )
    gate = {
        "schema_version": 3,
        "status": "TEST_GATE_FROZEN",
        "allow_dirty_dev_runs": False,
        "git_commit": "deadbeef",
        "metrics": metrics,
        "metric_metadata_hash": canonical_hash(metric_metadata),
        "test_manifests": {
            "test-clean": _sha256(clean_manifest),
            "test-other": _sha256(other_manifest),
        },
        "source_hashes": {
            relative: _sha256(repository / relative) for relative in source_paths
        },
        "final_system_registry": {
            "schema_version": 1,
            "status": "FINAL_SYSTEMS_PREREGISTERED",
            "roles": sorted(role for system in systems for role in system["roles"]),
            "unique_system_identities": len(systems),
        },
        "systems": systems,
    }
    gate_path = tmp_path / "TEST_GATE_FROZEN"
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    queue_path = tmp_path / "FINAL_TEST_QUEUE_FROZEN.json"
    return {
        "repository": repository,
        "run_root": run_root,
        "cache_root": cache_root,
        "asset_manifest": asset_manifest,
        "gate": gate,
        "gate_path": gate_path,
        "queue_path": queue_path,
        "metric_metadata": metric_metadata,
    }


def _prepare(fixture: dict) -> dict:
    return prepare_test_queue(
        test_gate=fixture["gate_path"],
        asset_manifest=fixture["asset_manifest"],
        run_root=fixture["run_root"],
        cache_root=fixture["cache_root"],
        output=fixture["queue_path"],
        repository_root=fixture["repository"],
    )


def test_prepare_queue_deduplicates_roles_and_expands_both_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    queue = _prepare(fixture)
    assert queue["unique_system_identities"] == 3
    assert queue["task_count"] == 6
    assert {task["split"] for task in queue["tasks"]} == {
        "test-clean",
        "test-other",
    }
    ours = [task for task in queue["tasks"] if task["adapter"] == "ours"]
    assert len(ours) == 2
    assert all(
        task["roles"] == ["base_m_seed2026", "flagship_m_seed2026"] for task in ours
    )
    assert len({task["environment"]["OURS_CKPT"] for task in ours}) == 1
    assert _prepare(fixture) == queue

    dry_run = run_test_queue(
        queue_path=fixture["queue_path"],
        gpus=["0", "1", "2", "3"],
        repository_root=fixture["repository"],
        dry_run=True,
    )
    assert dry_run["status"] == "FINAL_TEST_QUEUE_DRY_RUN"
    assert dry_run["completed"] == []
    assert len(dry_run["pending"]) == 6
    assert not (fixture["queue_path"].with_suffix(".state.json")).exists()


def test_prepare_hashes_a_checkpoint_shared_by_roles_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    ours_path = str(
        yaml.safe_load(fixture["asset_manifest"].read_text(encoding="utf-8"))[
            "ours_checkpoints"
        ]["base_m_seed2026"]["path"]
    )
    original = test_queue_module._asset_record
    observed_paths: list[str] = []

    def recording_asset_record(value, *, label):
        observed_paths.append(str(value.get("path", "")))
        return original(value, label=label)

    monkeypatch.setattr(test_queue_module, "_asset_record", recording_asset_record)
    _prepare(fixture)
    assert observed_paths.count(ours_path) == 1


def test_prepare_queue_rejects_nonformal_gate_or_asset_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    gate = json.loads(fixture["gate_path"].read_text(encoding="utf-8"))
    gate["status"] = "TEST_GATE_DIAGNOSTIC"
    fixture["gate_path"].write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="TEST_GATE_FROZEN"):
        _prepare(fixture)

    fixture = _fixture(tmp_path / "drift", monkeypatch)
    assets = yaml.safe_load(fixture["asset_manifest"].read_text(encoding="utf-8"))
    Path(assets["shared"]["dac_checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare(fixture)

    fixture = _fixture(tmp_path / "wrong-dac", monkeypatch)
    gate = json.loads(fixture["gate_path"].read_text(encoding="utf-8"))
    dac = next(system for system in gate["systems"] if system["adapter"] == "dac")
    dac["raw_checkpoint_sha256"] = "f" * 64
    fixture["gate_path"].write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(ValueError, match="dac checkpoint differs"):
        _prepare(fixture)


def test_queue_is_immutable_and_rejects_manual_task_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _prepare(fixture)
    queue = json.loads(fixture["queue_path"].read_text(encoding="utf-8"))
    queue["tasks"][0]["system"] = "Edited after freeze"
    fixture["queue_path"].write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(ValueError, match="does not rebuild"):
        run_test_queue(
            queue_path=fixture["queue_path"],
            gpus=["0"],
            repository_root=fixture["repository"],
            dry_run=True,
        )

    fixture = _fixture(tmp_path / "immutable", monkeypatch)
    _prepare(fixture)
    with fixture["asset_manifest"].open("a", encoding="utf-8") as stream:
        stream.write("# same resolved assets, different provenance\n")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _prepare(fixture)


def _write_completed_identity_task(
    fixture: dict, queue: dict, task: dict, *, resamples: int = 10_000
) -> None:
    run_directory = Path(task["run_directory"])
    run_directory.mkdir(parents=True)
    counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_words": 2,
    }
    record = {
        "run_id": task["task_id"],
        "system": task["system"],
        "variant": task["variant"],
        "seed": task["seed"],
        "split": task["split"],
        "cut_id": "synthetic-cut",
        "speaker_id": "synthetic-speaker",
        "duration": 1.0,
        "nominal_bitrate_kbps": 0.0,
        "serialized_bitrate_kbps": None,
        "token_frame_rate": 0.0,
        "num_parallel_streams": 0,
        "codebook_cardinalities": [],
        "code_histograms": [],
        "metrics": {
            "pesq_wb": 4.0,
            "speaker_cosine": 1.0,
            "visqol_speech": 5.0,
        },
        "asr": {"reference": counts, "reconstruction": counts},
        "metric_failures": [],
        "failed": False,
        "config_hash": task["config_hash"],
        "checkpoint_hash": task["checkpoint_hash"],
        "encoder_seconds": 0.0,
        "decoder_seconds": 0.01,
        "peak_vram_bytes": 0,
        "model_complexity": {},
    }
    records = run_directory / "per_utterance.jsonl"
    records.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(records, run_directory / "summary.json")
    (run_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": task["task_id"],
                "system": task["adapter"],
                "split": task["split"],
                "seed": task["seed"],
                "manifest_sha256": task["manifest"]["sha256"],
                "system_config_hash": task["config_hash"],
                "decode_seed_or_policy": task["decode_seed_or_policy"],
                "test_gate_sha256": queue["test_gate"]["sha256"],
                "max_cuts": None,
                "metrics": queue["metrics"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_directory / "data_manifest.json").write_text(
        json.dumps(
            {
                "sha256": task["manifest"]["sha256"],
                "cuts": task["manifest"]["cuts"],
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "model_metadata.json").write_text(
        json.dumps({"metrics": fixture["metric_metadata"]}), encoding="utf-8"
    )
    (run_directory / "git_state.json").write_text(
        json.dumps({"commit": "deadbeef", "dirty": False}), encoding="utf-8"
    )
    statistics = {
        metric: {
            "metric": metric,
            "point_estimate": 1.0,
            "resamples": resamples,
            "finite_resamples": resamples,
        }
        for metric in PRIMARY_BOOTSTRAP_METRICS
    }
    (run_directory / "statistics.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )


def test_completed_task_requires_rebuilt_summary_metadata_and_10k_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    queue = _prepare(fixture)
    task = next(task for task in queue["tasks"] if task["adapter"] == "identity")
    _write_completed_identity_task(fixture, queue, task)
    assert verify_task(task, queue, fixture["gate"]).statistics_complete

    dry_run = run_test_queue(
        queue_path=fixture["queue_path"],
        gpus=["0"],
        repository_root=fixture["repository"],
        dry_run=True,
    )
    assert task["task_id"] in dry_run["completed"]
    assert task["task_id"] not in dry_run["pending"]

    statistics_path = Path(task["run_directory"]) / "statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics["wer_delta"]["finite_resamples"] = 9_999
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 10,000"):
        verify_task(task, queue, fixture["gate"])

    records_path = Path(task["run_directory"]) / "per_utterance.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["cut_id"] = "wrong-cut"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    write_summary(records_path, Path(task["run_directory"]) / "summary.json")
    statistics["wer_delta"]["finite_resamples"] = 10_000
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")
    with pytest.raises(ValueError, match="output identity mismatch"):
        verify_task(task, queue, fixture["gate"])


class _FailedProcess:
    returncode = 7

    def poll(self) -> int:
        return self.returncode


def test_runner_stops_dispatching_after_the_first_parallel_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text("{}\n", encoding="utf-8")
    tasks = [
        {
            "task_id": f"task-{index}",
            "adapter": "identity",
            "split": "test-clean",
            "seed": 2026,
            "environment": {},
            "run_directory": str(tmp_path / f"run-{index}"),
        }
        for index in range(3)
    ]
    queue = {
        "run_root": str(tmp_path / "run-root"),
        "cache_root": str(tmp_path / "cache-root"),
        "metrics": ["wer"],
        "test_gate": {"path": str(tmp_path / "gate"), "sha256": "gate-sha"},
        "tasks": tasks,
    }
    monkeypatch.setattr(
        "dmel_codec.tools.test_queue._validate_queue", lambda *_: (queue, {})
    )
    monkeypatch.setattr(
        "dmel_codec.tools.test_queue.verify_task",
        lambda *_: type(
            "State", (), {"evaluation_complete": False, "statistics_complete": False}
        )(),
    )
    launched = []

    def fake_popen(command, **kwargs):
        launched.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return _FailedProcess()

    result = run_test_queue(
        queue_path=queue_path,
        gpus=["0", "1"],
        repository_root=Path(__file__).resolve().parents[1],
        popen=fake_popen,
        poll_seconds=0,
    )
    assert result["status"] == "FINAL_TEST_QUEUE_FAILED"
    assert len(launched) == 2
    assert len(result["failures"]) == 2
    assert result["not_started"] == ["task-2"]


def test_formal_test_queue_wrapper_has_no_diagnostic_or_test_bypass() -> None:
    repository = Path(__file__).resolve().parents[1]
    wrapper = (repository / "scripts/run_test_queue.sh").read_text(encoding="utf-8")
    assert "dmel_codec.tools.test_queue run" in wrapper
    assert "TEST_QUEUE" in wrapper
    assert "CUDA_VISIBLE_DEVICES" in wrapper
    assert "--diagnostic" not in wrapper
    assert "--dry-run" not in wrapper
