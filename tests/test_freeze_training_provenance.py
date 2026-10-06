from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.tools import freeze_test_gate


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def test_freeze_provenance_binds_source_retained_flagship(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_path = _write_json(tmp_path / "direction-gate.json", {"gate": True})
    source_hashes = {}
    source_records = {}
    for seed, value in zip((2026, 2027, 2028), (b"a", b"b", b"c")):
        checkpoint = tmp_path / f"source-{seed}.ckpt"
        checkpoint.write_bytes(value)
        source_hashes[seed] = sha256_file(checkpoint)
        source_records[str(seed)] = {"checkpoint": _record(checkpoint)}
    identity = {
        "system": "Ours-M",
        "variant": "ours",
        "seed": 2026,
        "config_hash": "config",
        "checkpoint_hash": "canonical",
        "raw_checkpoint_sha256": source_hashes[2026],
        "git_commit": "a" * 40,
    }
    selection = {
        "selected_direction": "source",
        "direction_gate": _record(gate_path),
        "selected_seed2026_evaluation_identity": identity,
        "source_checkpoints": source_records,
    }
    selection_path = _write_json(tmp_path / "selection.json", selection)
    monkeypatch.setattr(
        freeze_test_gate.ours_075_release,
        "validate_direction_selection",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        freeze_test_gate.mplus_replication,
        "selected_seed2026_identity",
        lambda *_args, **_kwargs: identity,
    )
    entries = {
        "base_m_seed2026": {
            "system": "Ours-M",
            "raw_checkpoint_sha256": source_hashes[2026],
        },
        **{
            f"flagship_m_seed{seed}": {
                "system": "Ours-M",
                "raw_checkpoint_sha256": source_hashes[seed],
            }
            for seed in (2026, 2027, 2028)
        },
    }

    result = freeze_test_gate._validate_mplus_provenance(
        selection_path=selection_path,
        replication_completion_paths=[],
        entry_by_role=entries,
        dev_commit="a" * 40,
        repository_root=tmp_path,
    )

    assert result["selected_direction"] == "source"
    assert result["replication_completions"] == {}
    entries["flagship_m_seed2028"]["raw_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="uses another checkpoint"):
        freeze_test_gate._validate_mplus_provenance(
            selection_path=selection_path,
            replication_completion_paths=[],
            entry_by_role=entries,
            dev_commit="a" * 40,
            repository_root=tmp_path,
        )


def test_freeze_provenance_binds_ours_075_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path = _write_json(tmp_path / "selection.json", {"selected": "joint"})
    release_path = _write_json(tmp_path / "release.json", {"release": True})
    best_hash = "d" * 64
    completion = {
        "schema_version": 1,
        "status": "OURS_075_TRAINING_COMPLETE",
        "seed": 2026,
        "final_global_step": 100_000,
        "release": _record(release_path),
        "run_dir": str(tmp_path / "run"),
        "best_checkpoint": {"sha256": best_hash},
    }
    completion_path = _write_json(tmp_path / "075-complete.json", completion)
    release = {
        "mplus_direction_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        }
    }
    monkeypatch.setattr(
        freeze_test_gate.ours_075_release,
        "verify_release",
        lambda **_kwargs: release,
    )
    monkeypatch.setattr(
        freeze_test_gate.ours_075_release,
        "verify_final_selection",
        lambda **_kwargs: completion,
    )
    entries = {
        "ours_075_seed2026": {"raw_checkpoint_sha256": best_hash},
    }

    result = freeze_test_gate._validate_ours_075_provenance(
        release_path=release_path,
        completion_path=completion_path,
        mplus_selection_path=selection_path,
        entry_by_role=entries,
        repository_root=tmp_path,
    )

    assert result["best_checkpoint_sha256"] == best_hash
