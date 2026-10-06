from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from dmel_codec.tools.make_listening_bundle import (
    ROUTES,
    SPLITS,
    make_listening_bundle,
)


def _write_wav(path: Path, frames: int, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24_000)
        stream.writeframes(value.to_bytes(2, "little", signed=True) * frames)


def _run(root: Path, route: str, split: str, *, mismatched: bool = False) -> Path:
    run_dir = root / route / split
    (run_dir / "audio_samples").mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": f"{route}-{split}",
                "split": split,
                "failed_utterances": 0,
                "checkpoint_hash": f"checkpoint-{route}",
                "config_hash": f"config-{route}",
            }
        ),
        encoding="utf-8",
    )
    for index, cut_id in enumerate(("cut_a", "cut_b")):
        frames = 100 + index
        _write_wav(run_dir / "audio_samples" / f"{cut_id}_reference.wav", frames, 0)
        reconstruction_frames = frames + int(mismatched and cut_id == "cut_b")
        _write_wav(
            run_dir / "audio_samples" / f"{cut_id}_reconstruction.wav",
            reconstruction_frames,
            index + sum(route.encode("ascii")) % 100 + 1,
        )
    return run_dir


def _inputs(root: Path, *, mismatched_route: str | None = None):
    return {
        route: {
            split: _run(
                root,
                route,
                split,
                mismatched=route == mismatched_route and split == "dev-other",
            )
            for split in SPLITS
        }
        for route in ROUTES
    }


def test_make_listening_bundle_is_blind_matched_and_idempotent(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path / "runs")
    output = tmp_path / "listening"
    first = make_listening_bundle(
        inputs,
        output_dir=output,
        randomization_seed=2027,
        expected_cuts_per_split=2,
    )
    second = make_listening_bundle(
        inputs,
        output_dir=output,
        randomization_seed=2027,
        expected_cuts_per_split=2,
    )
    assert first == second
    assert len(first["items"]) == 4
    assert len(list(output.glob("*.wav"))) == 20
    assert all(path.is_symlink() for path in output.glob("*.wav"))
    index = (output / "index.html").read_text()
    assert index.count("<audio controls") == 20
    assert index.count("<select aria-label") == 16
    assert "Download ratings" in index
    assert "\x00" not in index
    key = json.loads((output / "listening_key.json").read_text())
    assert all(
        {condition["route"] for condition in item["conditions"]} == set(ROUTES)
        for item in key["items"]
    )
    orders = [
        tuple(condition["route"] for condition in item["conditions"])
        for item in key["items"]
    ]
    assert len(set(orders)) > 1


def test_make_listening_bundle_rejects_length_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "runs", mismatched_route="joint")
    with pytest.raises(ValueError, match="Paired WAV format/length mismatch"):
        make_listening_bundle(
            inputs,
            output_dir=tmp_path / "listening",
            expected_cuts_per_split=2,
        )


def test_make_listening_bundle_rejects_incomplete_route_set(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path / "runs")
    inputs.pop("vocoder")
    with pytest.raises(ValueError, match="cover exactly"):
        make_listening_bundle(inputs, output_dir=tmp_path / "listening")


def test_make_listening_bundle_supports_paired_dmel_routes(tmp_path: Path) -> None:
    routes = ("direct", "decoder")
    inputs = {
        route: {split: _run(tmp_path / "runs", route, split) for split in SPLITS}
        for route in routes
    }
    output = tmp_path / "listening"
    result = make_listening_bundle(
        inputs,
        output_dir=output,
        randomization_seed=2027,
        expected_cuts_per_split=2,
        routes=routes,
        title="dMel decoder matched listening",
        status_prefix="DMEL_DECODER_MATCHED",
        ratings_filename="dmel_decoder_matched_listening_ratings.json",
    )
    assert result["routes"] == list(routes)
    assert result["status"] == "DMEL_DECODER_MATCHED_LISTENING_BUNDLE"
    assert len(list(output.glob("*.wav"))) == 12
    index = (output / "index.html").read_text()
    assert index.count("<audio controls") == 12
    assert index.count("<select aria-label") == 8
    assert "dMel decoder matched listening" in index
    assert "DMEL_DECODER_MATCHED_LISTENING_RATINGS" in index
    assert "dmel_decoder_matched_listening_ratings.json" in index
    key = json.loads((output / "listening_key.json").read_text())
    assert all(
        {condition["route"] for condition in item["conditions"]} == set(routes)
        for item in key["items"]
    )
