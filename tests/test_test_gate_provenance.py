from __future__ import annotations

from pathlib import Path

import pytest

from dmel_codec.evaluation.adapters.base import sha256_file
from dmel_codec.evaluation.run_codec_eval import _validate_training_provenance


def _artifact(path: Path, content: bytes) -> dict[str, str]:
    path.write_bytes(content)
    return {"path": str(path), "sha256": sha256_file(path)}


def _systems(selected_system: str, hashes: dict[str, str]) -> list[dict]:
    return [
        {
            "roles": ["base_m_seed2026"],
            "system": "Ours-M",
            "raw_checkpoint_sha256": hashes["base"],
        },
        *[
            {
                "roles": [f"flagship_m_seed{seed}"],
                "system": selected_system,
                "raw_checkpoint_sha256": hashes[str(seed)],
            }
            for seed in (2026, 2027, 2028)
        ],
        {
            "roles": ["ours_075_seed2026"],
            "system": "Ours-0.75",
            "raw_checkpoint_sha256": hashes["075"],
        },
    ]


def test_source_retained_training_provenance_is_self_consistent(tmp_path: Path) -> None:
    hashes = {
        "base": "a" * 64,
        "2026": "a" * 64,
        "2027": "b" * 64,
        "2028": "c" * 64,
        "075": "d" * 64,
    }
    gate = {
        "systems": _systems("Ours-M", hashes),
        "training_provenance": {
            "mplus": {
                "direction_selection": _artifact(
                    tmp_path / "selection.json", b"selection"
                ),
                "selected_direction": "source",
                "selected_system": "Ours-M",
                "selected_seed2026_checkpoint_sha256": hashes["2026"],
                "source_checkpoint_sha256_by_seed": {
                    "2026": hashes["2026"],
                    "2027": hashes["2027"],
                    "2028": hashes["2028"],
                },
                "replication_completions": {},
            },
            "ours_075": {
                "formal_release": _artifact(tmp_path / "release.json", b"release"),
                "training_completion": _artifact(
                    tmp_path / "075-complete.json", b"complete"
                ),
                "best_checkpoint_sha256": hashes["075"],
            },
        },
    }

    _validate_training_provenance(gate)

    gate["systems"][2]["raw_checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="flagship_m_seed2027"):
        _validate_training_provenance(gate)


def test_promoted_training_provenance_requires_both_replications(
    tmp_path: Path,
) -> None:
    hashes = {
        "base": "a" * 64,
        "2026": "j" * 64,
        "2027": "k" * 64,
        "2028": "l" * 64,
        "075": "d" * 64,
    }
    selected_system = "Ours-M+ (full codec + full vocoder + waveform GAN)"
    replications = {
        seed: {
            **_artifact(tmp_path / f"mplus-{seed}.json", seed.encode()),
            "best_checkpoint_sha256": hashes[seed],
        }
        for seed in ("2027", "2028")
    }
    gate = {
        "systems": _systems(selected_system, hashes),
        "training_provenance": {
            "mplus": {
                "direction_selection": _artifact(
                    tmp_path / "selection.json", b"selection"
                ),
                "selected_direction": "joint",
                "selected_system": selected_system,
                "selected_seed2026_checkpoint_sha256": hashes["2026"],
                "source_checkpoint_sha256_by_seed": {
                    "2026": hashes["base"],
                    "2027": "b" * 64,
                    "2028": "c" * 64,
                },
                "replication_completions": replications,
            },
            "ours_075": {
                "formal_release": _artifact(tmp_path / "release.json", b"release"),
                "training_completion": _artifact(
                    tmp_path / "075-complete.json", b"complete"
                ),
                "best_checkpoint_sha256": hashes["075"],
            },
        },
    }

    _validate_training_provenance(gate)

    del replications["2028"]
    with pytest.raises(ValueError, match="lacks both replications"):
        _validate_training_provenance(gate)
