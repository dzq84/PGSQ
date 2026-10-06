from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.tools import prepare_mplus_final_eval


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(root: Path, route: str, *, selected_step: int = 98000) -> Path:
    run_dir = root / route
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    selected = checkpoint_dir / f"step-{selected_step:08d}.ckpt"
    selected.write_bytes(f"selected-{route}".encode())
    last = checkpoint_dir / "last.ckpt"
    last.write_bytes(f"last-{route}".encode())
    _write_json(
        run_dir / "checkpoint_selection.json",
        {
            "monitor": "val_loss",
            "mode": "min",
            "best": {
                "filename": selected.name,
                "global_step": selected_step,
                "sha256": sha256_file(selected),
            },
            "last_checkpoint": {
                "filename": last.name,
                "global_step": 100000,
                "sha256": sha256_file(last),
            },
            "candidates": [{}, {}],
        },
    )
    return run_dir


def _inputs(root: Path):
    runs = {
        route: _run(root / "runs", route) for route in prepare_mplus_final_eval.ROUTES
    }
    receipts = {}
    for route in prepare_mplus_final_eval.TRAINING_ROUTES:
        selection = runs[route] / "checkpoint_selection.json"
        selected = prepare_mplus_final_eval._selected_checkpoint(runs[route])
        receipt = root / "receipts" / f"{route}.json"
        _write_json(
            receipt,
            {
                "direction": route,
                "final_global_step": 100000,
                "final_checkpoint_sha256": sha256_file(
                    runs[route] / "checkpoints" / "last.ckpt"
                ),
                "selection_sha256": sha256_file(selection),
                "best_filename": Path(selected["path"]).name,
                "best_sha256": selected["sha256"],
            },
        )
        receipts[route] = receipt
    route_script = root / "run_final_route.sh"
    postprocess_script = root / "postprocess_final_dev.sh"
    route_script.write_text("route\n", encoding="utf-8")
    postprocess_script.write_text("postprocess\n", encoding="utf-8")
    return runs, receipts, route_script, postprocess_script


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, dirty: bool = False):
    commit = "a" * 40
    monkeypatch.setattr(
        prepare_mplus_final_eval, "_git_state", lambda _: (commit, dirty)
    )
    runs, receipts, route_script, postprocess_script = _inputs(tmp_path)
    output = tmp_path / "FINAL_EVAL_READY.json"
    marker = prepare_mplus_final_eval.prepare_final_eval_marker(
        repository_root=tmp_path,
        expected_git_commit=commit,
        evaluation_tag="deterministic-v1-a1b2c3d",
        run_directories=runs,
        selection_receipts=receipts,
        route_script=route_script,
        postprocess_script=postprocess_script,
        output=output,
    )
    return output, marker, runs, receipts


def test_prepare_final_eval_marker_binds_all_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, marker, _, _ = _prepare(tmp_path, monkeypatch)
    assert output.is_file()
    assert marker["status"] == "FINAL_EVAL_READY"
    assert marker["test_access"] is False
    assert marker["encoding_policy"]["name"] == (
        "utterance-independent-deterministic-v1"
    )
    assert marker["metrics"] == [
        "estoi",
        "f0_correlation",
        "mr_stft",
        "ms_log_mel",
        "pesq_wb",
        "si_sdr",
        "speaker_cosine",
        "stoi",
        "visqol_speech",
        "wer",
    ]
    assert set(marker["checkpoints"]) == set(prepare_mplus_final_eval.ROUTES)
    assert marker["checkpoints"]["joint"]["selection_receipt"]["sha256"]
    assert set(marker["queue_scripts"]) == {"route", "postprocess"}


def test_prepare_final_eval_marker_rejects_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="clean worktree"):
        _prepare(tmp_path, monkeypatch, dirty=True)


def test_prepare_final_eval_marker_rejects_receipt_checkpoint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        prepare_mplus_final_eval, "_git_state", lambda _: (commit, False)
    )
    runs, receipts, route_script, postprocess_script = _inputs(tmp_path)
    receipt = json.loads(receipts["joint"].read_text(encoding="utf-8"))
    receipt["best_sha256"] = "0" * 64
    _write_json(receipts["joint"], receipt)
    with pytest.raises(ValueError, match="selected checkpoint"):
        prepare_mplus_final_eval.prepare_final_eval_marker(
            repository_root=tmp_path,
            expected_git_commit=commit,
            evaluation_tag="deterministic-v1-a1b2c3d",
            run_directories=runs,
            selection_receipts=receipts,
            route_script=route_script,
            postprocess_script=postprocess_script,
            output=tmp_path / "FINAL_EVAL_READY.json",
        )


def test_prepare_final_eval_marker_rejects_modified_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(
        prepare_mplus_final_eval, "_git_state", lambda _: (commit, False)
    )
    runs, receipts, route_script, postprocess_script = _inputs(tmp_path)
    (runs["codec"] / "checkpoints" / "last.ckpt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Final checkpoint SHA-256 mismatch"):
        prepare_mplus_final_eval.prepare_final_eval_marker(
            repository_root=tmp_path,
            expected_git_commit=commit,
            evaluation_tag="deterministic-v1-a1b2c3d",
            run_directories=runs,
            selection_receipts=receipts,
            route_script=route_script,
            postprocess_script=postprocess_script,
            output=tmp_path / "FINAL_EVAL_READY.json",
        )


def test_prepare_final_eval_marker_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, runs, receipts = _prepare(tmp_path, monkeypatch)
    route_script = tmp_path / "run_final_route.sh"
    postprocess_script = tmp_path / "postprocess_final_dev.sh"
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_mplus_final_eval.prepare_final_eval_marker(
            repository_root=tmp_path,
            expected_git_commit="a" * 40,
            evaluation_tag="deterministic-v1-a1b2c3d",
            run_directories=runs,
            selection_receipts=receipts,
            route_script=route_script,
            postprocess_script=postprocess_script,
            output=output,
        )
