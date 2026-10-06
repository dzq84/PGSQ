from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from dmel_codec.evaluation.cache import canonical_hash, sha256_file
from dmel_codec.tools import mplus_replication


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkpoint_metadata(
    resolved_config: dict,
    *,
    source_hash: str,
    cudnn_enabled: bool = False,
) -> dict:
    return {
        "training_runtime": {
            "generator_backward_cudnn_enabled": cudnn_enabled,
        },
        "config_sha256": canonical_hash(resolved_config),
        "resolved_config": resolved_config,
        "model": {"initialization": {"sha256": source_hash}},
        "git": {"commit": "a" * 40, "dirty": False},
    }


def _source_run(root: Path, seed: int) -> Path:
    run_dir = root / f"seed_{seed}"
    checkpoint = run_dir / "checkpoints" / "step-00001000.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{seed}".encode())
    _write_json(
        run_dir / "checkpoint_selection.json",
        {
            "monitor": "val_loss",
            "mode": "min",
            "best": {
                "filename": checkpoint.name,
                "global_step": 1000,
                "sha256": sha256_file(checkpoint),
            },
        },
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"seed": seed}), encoding="utf-8"
    )
    _write_json(
        run_dir / "model_metadata.json",
        {
            "codec_definition": {
                "nominal_bitrate_kbps": 1.746377604335702,
                "decoder_input_policy": "zeros",
            }
        },
    )
    _write_json(run_dir / "data_manifest.json", {"manifest": "shared"})
    return run_dir


def _gate_and_receipt(
    root: Path, selected: str = "joint", *, source_hash: str
) -> tuple[Path, Path]:
    def identity(system: str, raw_hash: str) -> dict:
        return {
            "system": system,
            "variant": "ours",
            "seed": 2026,
            "config_hash": f"config-{system}",
            "checkpoint_hash": f"canonical-{raw_hash}",
            "raw_checkpoint_sha256": raw_hash,
            "git_commit": "a" * 40,
        }

    source_identity = identity("Ours-M", source_hash)
    systems = {
        "codec": "Ours-M+ (full codec + frozen vocoder + waveform GAN)",
        "vocoder": "Ours-M+ (frozen codec + full vocoder + waveform GAN)",
        "joint": "Ours-M+ (full codec + full vocoder + waveform GAN)",
    }
    gate = root / "direction_gate.json"
    _write_json(
        gate,
        {
            "schema_version": 1,
            "status": "MPLUS_DIRECTION_GATE",
            "automatic_promotion": False,
            "eligible_before_listening": ["codec", "joint"],
            "disposition": "listening_required",
            "routes": {
                route: {
                    "runs": {
                        split: {
                            "source_identity": source_identity,
                            "candidate_identity": identity(system, (route[0] * 64)),
                        }
                        for split in ("dev-clean", "dev-other")
                    }
                }
                for route, system in systems.items()
            },
        },
    )
    manifest = root / "listening" / "README.md"
    manifest.parent.mkdir()
    manifest.write_text("blind listening index\n", encoding="utf-8")
    receipt = root / "listening_receipt.json"
    _write_json(
        receipt,
        {
            "schema_version": 1,
            "status": "MPLUS_MATCHED_LISTENING_COMPLETE",
            "direction_gate_sha256": sha256_file(gate),
            "selected_direction": selected,
            "evaluated_directions": ["codec", "vocoder", "joint"],
            "listening_manifest": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
            },
        },
    )
    return gate, receipt


def _freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: str = "joint"):
    monkeypatch.setattr(mplus_replication, "_git_state", lambda _: ("a" * 40, False))
    sources = {
        seed: _source_run(tmp_path / "sources", seed)
        for seed in mplus_replication.FORMAL_SEEDS
    }
    source_hash = sha256_file(sources[2026] / "checkpoints" / "step-00001000.ckpt")
    gate, receipt = _gate_and_receipt(tmp_path, selected, source_hash=source_hash)
    output = tmp_path / "MPLUS_DIRECTION_SELECTED.json"
    marker = mplus_replication.freeze_direction_selection(
        direction_gate=gate,
        listening_receipt=receipt,
        source_run_directories=sources,
        output=output,
        repository_root=tmp_path,
    )
    return output, marker


def _freeze_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    statistical_leaders: list[str] | None = None,
):
    current_commit = "a" * 40
    evidence_commit = "b" * 40
    monkeypatch.setattr(
        mplus_replication, "_git_state", lambda _: (current_commit, False)
    )
    sources = {
        seed: _source_run(tmp_path / "sources", seed)
        for seed in mplus_replication.FORMAL_SEEDS
    }
    source_hash = sha256_file(sources[2026] / "checkpoints" / "step-00001000.ckpt")
    gate, _ = _gate_and_receipt(tmp_path, "joint", source_hash=source_hash)
    gate_document = json.loads(gate.read_text(encoding="utf-8"))
    gate_document["statistical_leaders"] = (
        ["joint"] if statistical_leaders is None else statistical_leaders
    )
    for route in mplus_replication.TRAINING_ROUTES:
        for split in ("dev-clean", "dev-other"):
            run = gate_document["routes"][route]["runs"][split]
            run["source_identity"]["git_commit"] = evidence_commit
            run["candidate_identity"]["git_commit"] = evidence_commit
    _write_json(gate, gate_document)
    output = tmp_path / "MPLUS_DIRECTION_SELECTED.json"
    marker = mplus_replication.freeze_direction_selection(
        direction_gate=gate,
        listening_receipt=None,
        objective_only_user_authorized=True,
        authorization_note="User explicitly waived listening before test access.",
        source_run_directories=sources,
        output=output,
        repository_root=tmp_path,
    )
    return output, marker


def test_freeze_direction_selection_binds_all_source_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, marker = _freeze(tmp_path, monkeypatch)
    assert output.is_file()
    assert marker["selected_direction"] == "joint"
    assert marker["disposition"] == "replicate_selected_direction"
    assert set(marker["source_checkpoints"]) == {"2026", "2027", "2028"}
    assert marker["source_checkpoints"]["2028"]["checkpoint"]["sha256"]
    assert marker["selected_seed2026_evaluation_identity"]["system"].startswith(
        "Ours-M+"
    )


def test_objective_selection_uses_only_unique_statistical_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, marker = _freeze_objective(tmp_path, monkeypatch)
    assert marker["schema_version"] == 2
    assert marker["selection_mode"] == "objective_only_user_authorized"
    assert marker["selected_direction"] == "joint"
    assert marker["direction_gate_evaluation_commit"] == "b" * 40
    assert marker["git_commit"] == "a" * 40
    assert marker["selection_evidence"]["matched_listening_waived"] is True
    assert (
        mplus_replication.validate_direction_selection(output, expected_commit="a" * 40)
        == marker
    )

    run_dir = tmp_path / "replications" / "seed_2027"
    preflight = mplus_replication.verify_replication_preflight(
        selection=output,
        seed=2027,
        run_dir=run_dir,
        repository_root=tmp_path,
    )
    assert preflight["direction"] == "joint"


def test_objective_selection_rejects_ambiguous_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="one statistical leader"):
        _freeze_objective(tmp_path, monkeypatch, statistical_leaders=["codec", "joint"])


def test_objective_selection_rejects_missing_authorization_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mplus_replication, "_git_state", lambda _: ("a" * 40, False))
    sources = {
        seed: _source_run(tmp_path / "sources", seed)
        for seed in mplus_replication.FORMAL_SEEDS
    }
    source_hash = sha256_file(sources[2026] / "checkpoints" / "step-00001000.ckpt")
    gate, _ = _gate_and_receipt(tmp_path, "joint", source_hash=source_hash)
    gate_document = json.loads(gate.read_text(encoding="utf-8"))
    gate_document["statistical_leaders"] = ["joint"]
    _write_json(gate, gate_document)
    with pytest.raises(ValueError, match="authorization note"):
        mplus_replication.freeze_direction_selection(
            direction_gate=gate,
            listening_receipt=None,
            objective_only_user_authorized=True,
            source_run_directories=sources,
            output=tmp_path / "selection.json",
            repository_root=tmp_path,
        )


def test_objective_selection_rejects_tampered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = _freeze_objective(tmp_path, monkeypatch)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["selection_evidence"]["statistical_leaders"] = ["codec"]
    _write_json(output, document)
    with pytest.raises(ValueError, match="changed the frozen statistical leader"):
        mplus_replication.validate_direction_selection(output, expected_commit="a" * 40)


def test_freeze_rejects_direction_that_failed_complete_dev_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mplus_replication, "_git_state", lambda _: ("a" * 40, False))
    sources = {
        seed: _source_run(tmp_path / "sources", seed)
        for seed in mplus_replication.FORMAL_SEEDS
    }
    gate, receipt = _gate_and_receipt(
        tmp_path,
        "vocoder",
        source_hash=sha256_file(sources[2026] / "checkpoints" / "step-00001000.ckpt"),
    )
    with pytest.raises(ValueError, match="failed the complete-dev gate"):
        mplus_replication.freeze_direction_selection(
            direction_gate=gate,
            listening_receipt=receipt,
            source_run_directories=sources,
            output=tmp_path / "selection.json",
            repository_root=tmp_path,
        )


def test_replication_preflight_uses_matching_seed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, marker = _freeze(tmp_path, monkeypatch)
    run_dir = tmp_path / "replications" / "seed_2028"
    result = mplus_replication.verify_replication_preflight(
        selection=selection,
        seed=2028,
        run_dir=run_dir,
        repository_root=tmp_path,
    )
    assert result["direction"] == "joint"
    assert result["experiment"].endswith("full_vocoder_gan_100k")
    assert result["run_dir"] == str(run_dir.resolve())
    assert len(result["resolved_config_sha256"]) == 64
    assert (
        result["source_checkpoint"]["sha256"]
        == (marker["source_checkpoints"]["2028"]["checkpoint"]["sha256"])
    )

    preflight = tmp_path / "replication_preflight.json"
    _write_json(preflight, result)
    assert (
        mplus_replication.validate_existing_preflight(
            selection=selection,
            preflight=preflight,
            repository_root=tmp_path,
        )
        == result
    )


def test_replication_preflight_rejects_existing_seed2026_direction_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, _ = _freeze(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="2027, 2028"):
        mplus_replication.verify_replication_preflight(
            selection=selection,
            seed=2026,
            run_dir=tmp_path / "replications" / "seed_2026",
            repository_root=tmp_path,
        )


def test_existing_preflight_rejects_changed_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, _ = _freeze(tmp_path, monkeypatch)
    result = mplus_replication.verify_replication_preflight(
        selection=selection,
        seed=2027,
        run_dir=tmp_path / "replications" / "seed_2027",
        repository_root=tmp_path,
    )
    result["direction"] = "codec"
    preflight = tmp_path / "replication_preflight.json"
    _write_json(preflight, result)
    with pytest.raises(ValueError, match="direction differs"):
        mplus_replication.validate_existing_preflight(
            selection=selection,
            preflight=preflight,
            repository_root=tmp_path,
        )


def test_replication_preflight_refuses_retain_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection, _ = _freeze(tmp_path, monkeypatch, selected="source")
    with pytest.raises(ValueError, match="retains source"):
        mplus_replication.verify_replication_preflight(
            selection=selection,
            seed=2027,
            run_dir=tmp_path / "replications" / "seed_2027",
            repository_root=tmp_path,
        )


def test_verify_replication_segment_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    resolved_config = {"seed": 2027, "model": {"weight_adv": 1.0}}
    config_hash = canonical_hash(resolved_config)
    source_hash = "b" * 64
    preflight = tmp_path / "preflight.json"
    _write_json(
        preflight,
        {
            "status": "MPLUS_REPLICATION_PREFLIGHT",
            "direction": "joint",
            "seed": 2027,
            "run_dir": str(run_dir.resolve()),
            "git_commit": "a" * 40,
            "source_checkpoint": {"sha256": source_hash},
            "generator_backward_cudnn_enabled": False,
            "resolved_config_sha256": config_hash,
        },
    )
    torch.save(
        {
            "global_step": 2000,
            "optimizer_states": [{}, {}],
            "lr_schedulers": [{}, {}],
            "LhotseDataModule": {
                "discarded_prefetched_batches": 1,
                "train_sampler": {
                    "diagnostics": {
                        "current_epoch": 0,
                        "stats_per_epoch": {0: {"kept_batches": 2000}},
                    }
                },
            },
            "codec_metadata": _checkpoint_metadata(
                resolved_config, source_hash=source_hash
            ),
        },
        checkpoint,
    )
    result = mplus_replication.verify_segment_checkpoint(
        checkpoint=checkpoint,
        preflight=preflight,
        target_step=2000,
        start_time="2026-08-22T10:00:00+08:00",
        end_time="2026-08-22T10:30:00+08:00",
        cuda_visible_devices="4",
    )
    assert result["checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["verified"]["sampler_kept_batches"] == 2000
    assert result["elapsed_seconds"] == 1800.0
    assert result["gpu_hours"] == 0.5

    with pytest.raises(ValueError, match="outside the frozen replication run"):
        mplus_replication.verify_segment_checkpoint(
            checkpoint=tmp_path / "last.ckpt",
            preflight=preflight,
            target_step=2000,
        )

    with pytest.raises(ValueError, match="requires start, end, and GPUs"):
        mplus_replication.verify_segment_checkpoint(
            checkpoint=checkpoint,
            preflight=preflight,
            target_step=2000,
            start_time="2026-08-22T10:00:00+08:00",
        )


def test_verify_final_replication_selection(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    best_path = checkpoints / "step-00098000.ckpt"
    runner_up_path = checkpoints / "step-00096000.ckpt"
    last_path = checkpoints / "last.ckpt"
    resolved_config = {"seed": 2028, "model": {"weight_adv": 1.0}}
    config_hash = canonical_hash(resolved_config)
    source_hash = "b" * 64
    for path, step in ((best_path, 98_000), (runner_up_path, 96_000)):
        torch.save(
            {
                "global_step": step,
                "codec_metadata": _checkpoint_metadata(
                    resolved_config, source_hash=source_hash
                ),
            },
            path,
        )
    torch.save(
        {
            "global_step": 100_000,
            "optimizer_states": [{}, {}],
            "lr_schedulers": [{}, {}],
            "LhotseDataModule": {
                "discarded_prefetched_batches": 2,
                "train_sampler": {
                    "diagnostics": {
                        "current_epoch": 0,
                        "stats_per_epoch": {0: {"kept_batches": 100_000}},
                    }
                },
            },
            "codec_metadata": _checkpoint_metadata(
                resolved_config, source_hash=source_hash
            ),
        },
        last_path,
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    best = {
        "filename": best_path.name,
        "global_step": 98000,
        "score": 0.5,
        "sha256": sha256_file(best_path),
    }
    _write_json(
        run_dir / "checkpoint_selection.json",
        {
            "monitor": "val_loss",
            "mode": "min",
            "best": best,
            "candidates": [
                {
                    "filename": runner_up_path.name,
                    "global_step": 96000,
                    "score": 0.6,
                    "sha256": sha256_file(runner_up_path),
                },
                best,
            ],
            "last_checkpoint": {
                "filename": last_path.name,
                "global_step": 100000,
                "sha256": sha256_file(last_path),
            },
        },
    )
    preflight = tmp_path / "preflight.json"
    _write_json(
        preflight,
        {
            "status": "MPLUS_REPLICATION_PREFLIGHT",
            "direction": "joint",
            "seed": 2028,
            "run_dir": str(run_dir.resolve()),
            "git_commit": "a" * 40,
            "source_checkpoint": {"sha256": source_hash},
            "generator_backward_cudnn_enabled": False,
            "resolved_config_sha256": config_hash,
        },
    )
    result = mplus_replication.verify_final_selection(
        run_dir=run_dir, preflight=preflight
    )
    assert result["status"] == "MPLUS_REPLICATION_COMPLETE"
    assert result["best_checkpoint"]["global_step"] == 98000
    assert result["last_checkpoint"]["sha256"] == sha256_file(last_path)
    assert result["verified"]["sampler_kept_batches"] == 100_000

    resolved_config["model"]["weight_adv"] = 0.25
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resolved config differs"):
        mplus_replication.verify_final_selection(run_dir=run_dir, preflight=preflight)
