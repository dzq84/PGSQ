from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.tools.complete_mplus_listening import complete_mplus_listening
from dmel_codec.tools.make_listening_bundle import ROUTES, SPLITS


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _documents(
    root: Path,
    *,
    eligible: list[str],
    route_scores: dict[str, int],
) -> tuple[Path, Path, Path, Path]:
    gate = _write_json(
        root / "direction_gate.json",
        {
            "schema_version": 1,
            "status": "MPLUS_DIRECTION_GATE",
            "automatic_promotion": False,
            "eligible_before_listening": eligible,
            "disposition": "listening_required" if eligible else "retain_source",
        },
    )
    items = []
    key_items = []
    for split in SPLITS:
        cut_id = f"{split}_cut"
        conditions = [
            {"blind_label": label, "file": f"{split}_{label}.wav"}
            for label in ("A", "B", "C", "D")
        ]
        key_conditions = [
            {"blind_label": label, "route": route, "file": f"{split}_{label}.wav"}
            for label, route in zip(("A", "B", "C", "D"), ROUTES, strict=True)
        ]
        items.append(
            {
                "split": split,
                "cut_id": cut_id,
                "reference_file": f"{split}_reference.wav",
                "conditions": conditions,
            }
        )
        key_items.append(
            {"split": split, "cut_id": cut_id, "conditions": key_conditions}
        )
    manifest = _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": "MPLUS_MATCHED_LISTENING_BUNDLE",
            "randomization_seed": 2027,
            "items": items,
        },
    )
    key = _write_json(
        root / "listening_key.json",
        {
            "schema_version": 1,
            "status": "MPLUS_MATCHED_LISTENING_KEY",
            "randomization_seed": 2027,
            "items": key_items,
        },
    )
    ratings = _write_json(
        root / "ratings.json",
        {
            "schema_version": 1,
            "status": "MPLUS_MATCHED_LISTENING_RATINGS",
            "listening_manifest_sha256": sha256_file(manifest),
            "created_at": "2026-08-22T00:00:00Z",
            "items": [
                {
                    "split": item["split"],
                    "cut_id": item["cut_id"],
                    "scores": {
                        label: route_scores[route]
                        for label, route in zip(
                            ("A", "B", "C", "D"), ROUTES, strict=True
                        )
                    },
                }
                for item in items
            ],
        },
    )
    return gate, manifest, key, ratings


def test_complete_mplus_listening_selects_unique_eligible_winner(
    tmp_path: Path,
) -> None:
    gate, manifest, key, ratings = _documents(
        tmp_path,
        eligible=["codec", "joint"],
        route_scores={"source": 3, "codec": 4, "vocoder": 2, "joint": 5},
    )
    output = tmp_path / "receipt.json"
    receipt = complete_mplus_listening(
        direction_gate=gate,
        listening_manifest=manifest,
        listening_key=key,
        ratings=ratings,
        output=output,
    )
    assert receipt["selected_direction"] == "joint"
    assert receipt["evaluated_directions"] == ["codec", "vocoder", "joint"]
    assert receipt["route_scores"]["joint"]["mean"] == 5.0
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        complete_mplus_listening(
            direction_gate=gate,
            listening_manifest=manifest,
            listening_key=key,
            ratings=ratings,
            output=output,
        )


def test_complete_mplus_listening_retains_source_on_top_tie(tmp_path: Path) -> None:
    gate, manifest, key, ratings = _documents(
        tmp_path,
        eligible=["codec", "joint"],
        route_scores={"source": 3, "codec": 5, "vocoder": 2, "joint": 5},
    )
    receipt = complete_mplus_listening(
        direction_gate=gate,
        listening_manifest=manifest,
        listening_key=key,
        ratings=ratings,
        output=tmp_path / "receipt.json",
    )
    assert receipt["selected_direction"] == "source"
    assert receipt["selection_reason"] == (
        "top eligible listening means tied; retained source"
    )


def test_complete_mplus_listening_rejects_unbound_ratings(tmp_path: Path) -> None:
    gate, manifest, key, ratings = _documents(
        tmp_path,
        eligible=["joint"],
        route_scores={"source": 3, "codec": 4, "vocoder": 2, "joint": 5},
    )
    document = json.loads(ratings.read_text())
    document["listening_manifest_sha256"] = "0" * 64
    _write_json(ratings, document)
    with pytest.raises(ValueError, match="do not bind"):
        complete_mplus_listening(
            direction_gate=gate,
            listening_manifest=manifest,
            listening_key=key,
            ratings=ratings,
            output=tmp_path / "receipt.json",
        )
