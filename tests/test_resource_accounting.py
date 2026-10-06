from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmel_codec.tools.resource_accounting import (
    record_job_receipt,
    summarize_resource_manifest,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipt(
    root: Path,
    name: str,
    *,
    start: str,
    end: str,
    devices: str,
    verified: bool = False,
) -> Path:
    value: dict[str, object] = {
        "start_time": start,
        "end_time": end,
        "cuda_visible_devices": devices,
    }
    if verified:
        value["verified"] = {"global_step": 2000}
    return _write_json(root / name, value)


def test_resource_accounting_sums_concurrent_gpu_jobs(tmp_path: Path) -> None:
    _receipt(
        tmp_path,
        "train-a.json",
        start="2026-08-22T10:00:00+08:00",
        end="2026-08-22T12:00:00+08:00",
        devices="2",
        verified=True,
    )
    _receipt(
        tmp_path,
        "train-b.json",
        start="2026-08-22T10:30:00+08:00",
        end="2026-08-22T11:30:00+08:00",
        devices="4",
        verified=True,
    )
    _receipt(
        tmp_path,
        "eval.json",
        start="2026-08-22T12:00:00+08:00",
        end="2026-08-22T12:30:00+08:00",
        devices="0,1",
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "jobs": [
                {
                    "label": "train-a",
                    "category": "training",
                    "outcome": "verified",
                    "receipt": "train-a.json",
                },
                {
                    "label": "train-b",
                    "category": "training",
                    "outcome": "verified",
                    "receipt": "train-b.json",
                },
                {
                    "label": "eval",
                    "category": "evaluation",
                    "outcome": "completed",
                    "receipt": "eval.json",
                },
            ],
        },
    )

    result = summarize_resource_manifest(manifest)

    assert result["totals"]["jobs"] == 3
    assert result["totals"]["wall_hours"] == pytest.approx(3.5)
    assert result["totals"]["gpu_hours"] == pytest.approx(4.0)
    assert result["totals"]["calendar_union_hours"] == pytest.approx(2.5)
    assert result["totals"]["by_category"]["training"]["gpu_hours"] == 3.0
    assert result["totals"]["by_category"]["evaluation"]["gpu_hours"] == 1.0
    assert len(result["jobs"][0]["receipt"]["sha256"]) == 64


def test_resource_accounting_separates_aborted_cost(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path,
        "abort.json",
        start="2026-08-22T10:00:00+08:00",
        end="2026-08-22T10:15:00+08:00",
        devices="3",
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "jobs": [
                {
                    "label": "failed-smoke",
                    "category": "diagnostic",
                    "outcome": "aborted",
                    "receipt": str(receipt),
                }
            ],
        },
    )

    output = tmp_path / "summary" / "resources.json"
    result = summarize_resource_manifest(manifest, output=output)

    assert result["totals"]["gpu_hours"] == pytest.approx(0.25)
    assert result["totals"]["by_outcome"]["aborted"]["gpu_hours"] == 0.25
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_record_job_receipt_is_immutable_and_aggregates(tmp_path: Path) -> None:
    receipt_path = tmp_path / "completed.json"
    receipt = record_job_receipt(
        label="ours-075-invocation",
        category="training",
        outcome="completed",
        start_time="2026-08-22T10:00:00+08:00",
        end_time="2026-08-22T10:30:00+08:00",
        cuda_visible_devices=" 2 ",
        exit_code=0,
        output=receipt_path,
    )
    assert receipt["cuda_visible_devices"] == "2"
    assert receipt["elapsed_seconds"] == 1800.0
    assert receipt["gpu_hours"] == 0.5
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        record_job_receipt(
            label="duplicate",
            category="training",
            outcome="completed",
            start_time="2026-08-22T10:00:00+08:00",
            end_time="2026-08-22T10:30:00+08:00",
            cuda_visible_devices="2",
            exit_code=0,
            output=receipt_path,
        )

    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "jobs": [
                {
                    "label": "ours-075-invocation",
                    "category": "training",
                    "outcome": "completed",
                    "receipt": str(receipt_path),
                }
            ],
        },
    )
    assert summarize_resource_manifest(manifest)["totals"]["gpu_hours"] == 0.5

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["jobs"][0]["outcome"] = "failed"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt outcome=.*disagrees with manifest"):
        summarize_resource_manifest(manifest)


def test_resource_accounting_rejects_inconsistent_derived_cost(tmp_path: Path) -> None:
    receipt = _receipt(
        tmp_path,
        "receipt.json",
        start="2026-08-22T10:00:00+08:00",
        end="2026-08-22T11:00:00+08:00",
        devices="2",
    )
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["gpu_hours"] = 99.0
    receipt.write_text(json.dumps(document), encoding="utf-8")
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "jobs": [
                {
                    "label": "job",
                    "category": "training",
                    "outcome": "completed",
                    "receipt": str(receipt),
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="gpu_hours disagrees"):
        summarize_resource_manifest(manifest)


def test_record_job_receipt_accepts_iso_decimal_comma(tmp_path: Path) -> None:
    receipt = record_job_receipt(
        label="fast-failure",
        category="training",
        outcome="failed",
        start_time="2026-08-22T10:00:00,001000000+08:00",
        end_time="2026-08-22T10:00:00,006000000+08:00",
        cuda_visible_devices="2",
        exit_code=7,
        output=tmp_path / "receipt.json",
    )
    assert receipt["elapsed_seconds"] == pytest.approx(0.005)
    assert receipt["start_time"] == "2026-08-22T10:00:00.001000+08:00"


@pytest.mark.parametrize(
    ("outcome", "exit_code", "match"),
    [
        ("completed", 1, "completed job receipt"),
        ("failed", 0, "needs a nonzero exit_code"),
        ("verified", 0, "completed, aborted, or failed"),
    ],
)
def test_record_job_receipt_rejects_inconsistent_outcome(
    tmp_path: Path, outcome: str, exit_code: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        record_job_receipt(
            label="job",
            category="training",
            outcome=outcome,
            start_time="2026-08-22T10:00:00+08:00",
            end_time="2026-08-22T10:30:00+08:00",
            cuda_visible_devices="2",
            exit_code=exit_code,
            output=tmp_path / "receipt.json",
        )


@pytest.mark.parametrize(
    ("receipt_update", "match"),
    [
        ({"start_time": "2026-08-22T10:00:00"}, "UTC offset"),
        ({"end_time": "2026-08-22T09:00:00+08:00"}, "after start_time"),
        ({"cuda_visible_devices": "2,2"}, "invalid cuda_visible_devices"),
    ],
)
def test_resource_accounting_rejects_invalid_receipt(
    tmp_path: Path, receipt_update: dict[str, str], match: str
) -> None:
    receipt = {
        "start_time": "2026-08-22T10:00:00+08:00",
        "end_time": "2026-08-22T11:00:00+08:00",
        "cuda_visible_devices": "2",
        "verified": {"global_step": 2000},
    }
    receipt.update(receipt_update)
    _write_json(tmp_path / "receipt.json", receipt)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "jobs": [
                {
                    "label": "job",
                    "category": "training",
                    "outcome": "verified",
                    "receipt": "receipt.json",
                }
            ],
        },
    )

    with pytest.raises(ValueError, match=match):
        summarize_resource_manifest(manifest)


def test_resource_accounting_rejects_duplicate_receipt(tmp_path: Path) -> None:
    _receipt(
        tmp_path,
        "receipt.json",
        start="2026-08-22T10:00:00+08:00",
        end="2026-08-22T11:00:00+08:00",
        devices="2",
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "jobs": [
                {
                    "label": label,
                    "category": "diagnostic",
                    "outcome": "completed",
                    "receipt": "receipt.json",
                }
                for label in ("first", "second")
            ],
        },
    )

    with pytest.raises(ValueError, match="duplicate receipt path"):
        summarize_resource_manifest(manifest)
