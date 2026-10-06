from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.tools import ours_075_release


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _selection(root: Path, *, selected: str = "joint") -> Path:
    gate = _write_json(
        root / "direction_gate.json",
        {
            "schema_version": 1,
            "status": "MPLUS_DIRECTION_GATE",
            "automatic_promotion": False,
            "eligible_before_listening": ["joint"],
        },
    )
    manifest = root / "listening_manifest.json"
    manifest.write_text("matched listening\n", encoding="utf-8")
    receipt = _write_json(
        root / "listening_receipt.json",
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
    return _write_json(
        root / "MPLUS_DIRECTION_SELECTED.json",
        {
            "schema_version": 1,
            "status": "MPLUS_DIRECTION_SELECTED",
            "selected_direction": selected,
            "disposition": (
                "retain_source"
                if selected == "source"
                else "replicate_selected_direction"
            ),
            "git_commit": "a" * 40,
            "direction_gate": {"path": str(gate), "sha256": sha256_file(gate)},
            "listening_receipt": {
                "path": str(receipt),
                "sha256": sha256_file(receipt),
            },
        },
    )


def _formal_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "runs"
    manifest_root = run_root / "manifests" / "libritts24k"
    manifest_root.mkdir(parents=True)
    (manifest_root / "train-clean-100_codec.jsonl.gz").write_bytes(b"train")
    (manifest_root / "dev-clean_eval.jsonl.gz").write_bytes(b"dev")
    bigvgan_checkpoint = tmp_path / "bigvgan.pt"
    bigvgan_checkpoint.write_bytes(b"bigvgan")
    bigvgan_config = tmp_path / "bigvgan.json"
    bigvgan_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DMEL_RUN_ROOT", str(run_root))
    monkeypatch.setenv("BIGVGAN_CKPT", str(bigvgan_checkpoint))
    monkeypatch.setenv("BIGVGAN_CONFIG", str(bigvgan_config))


@pytest.mark.parametrize("selected", ["source", "joint"])
def test_release_binds_complete_mplus_decision_and_lowrate_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: str
) -> None:
    _formal_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(ours_075_release, "_git_state", lambda _: ("a" * 40, False))
    selection = _selection(tmp_path, selected=selected)
    output = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    release = ours_075_release.create_release(
        selection=selection, repository_root=tmp_path, output=output
    )
    assert release["status"] == "OURS_075_FORMAL_RELEASE"
    assert release["mplus_direction_selection"]["selected_direction"] == selected
    assert release["formal_contract"]["fsq_levels"] == [3, 3]
    assert release["formal_contract"]["nominal_bitrate_kbps"] == pytest.approx(
        0.7429511719
    )
    assert release["formal_contract"]["max_steps"] == 100000
    assert len(release["formal_contract"]["resolved_config_sha256"]) == 64
    assert (
        ours_075_release.verify_release(release=output, repository_root=tmp_path)
        == release
    )


def test_release_rejects_changed_direction_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _formal_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(ours_075_release, "_git_state", lambda _: ("a" * 40, False))
    selection = _selection(tmp_path)
    output = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    ours_075_release.create_release(
        selection=selection, repository_root=tmp_path, output=output
    )
    document = json.loads(selection.read_text())
    document["selected_direction"] = "codec"
    selection.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ours_075_release.verify_release(release=output, repository_root=tmp_path)


def test_release_rejects_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ours_075_release, "_git_state", lambda _: ("a" * 40, True))
    with pytest.raises(ValueError, match="clean worktree"):
        ours_075_release.create_release(
            selection=_selection(tmp_path),
            repository_root=tmp_path,
            output=tmp_path / "release.json",
        )


def test_verify_final_selection_binds_100k_checkpoint_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _formal_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(ours_075_release, "_git_state", lambda _: ("a" * 40, False))
    release = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    ours_075_release.create_release(
        selection=_selection(tmp_path),
        repository_root=tmp_path,
        output=release,
    )
    release_document = json.loads(release.read_text(encoding="utf-8"))
    run_dir = Path(release_document["formal_contract"]["run_dir"])
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    best_path = checkpoints / "step-00098000.ckpt"
    runner_up_path = checkpoints / "step-00096000.ckpt"
    last_path = checkpoints / "last.ckpt"
    best_path.write_bytes(b"best")
    runner_up_path.write_bytes(b"runner-up")
    torch.save(
        {
            "global_step": 100_000,
            "optimizer_states": [{}, {}],
            "lr_schedulers": [{}, {}],
            "LhotseDataModule": {
                "train_sampler": {
                    "diagnostics": {
                        "current_epoch": 0,
                        "stats_per_epoch": {0: {"kept_batches": 100_000}},
                    }
                }
            },
            "codec_metadata": {
                "codec_definition": {
                    "decoder_input_policy": "zeros",
                    "nominal_bitrate_kbps": 0.7429511719,
                }
            },
        },
        last_path,
    )
    best = {
        "filename": best_path.name,
        "global_step": 98_000,
        "score": 0.4,
        "sha256": sha256_file(best_path),
    }
    runner_up = {
        "filename": runner_up_path.name,
        "global_step": 96_000,
        "score": 0.5,
        "sha256": sha256_file(runner_up_path),
    }
    _write_json(
        run_dir / "checkpoint_selection.json",
        {
            "monitor": "val_loss",
            "mode": "min",
            "candidates": [best, runner_up],
            "best": best,
            "last_checkpoint": {
                "filename": last_path.name,
                "global_step": 100_000,
                "sha256": sha256_file(last_path),
            },
        },
    )
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config = ours_075_release._formal_resolved_config()
    resolved_config_path.write_text(
        yaml.safe_dump(resolved_config, sort_keys=True),
        encoding="utf-8",
    )

    receipt = ours_075_release.verify_final_selection(
        run_dir=run_dir, release=release, repository_root=tmp_path
    )

    assert receipt["status"] == "OURS_075_TRAINING_COMPLETE"
    assert receipt["best_checkpoint"]["sha256"] == sha256_file(best_path)
    assert receipt["last_checkpoint"]["sha256"] == sha256_file(last_path)
    assert receipt["verified"]["sampler_kept_batches"] == 100_000

    resolved_config["codec"]["model"]["weight_adv"] = 0.25
    resolved_config_path.write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="complete resolved config"):
        ours_075_release.verify_final_selection(
            run_dir=run_dir, release=release, repository_root=tmp_path
        )


def test_verify_final_selection_rejects_unreleased_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _formal_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(ours_075_release, "_git_state", lambda _: ("a" * 40, False))
    release = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    ours_075_release.create_release(
        selection=_selection(tmp_path),
        repository_root=tmp_path,
        output=release,
    )

    with pytest.raises(ValueError, match="run directory"):
        ours_075_release.verify_final_selection(
            run_dir=tmp_path / "unreleased-run",
            release=release,
            repository_root=tmp_path,
        )
