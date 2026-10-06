from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_CATEGORIES = {"training", "evaluation", "diagnostic"}
_OUTCOMES = {"completed", "verified", "aborted", "failed"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _parse_timestamp(value: Any, *, field: str, receipt: Path) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{receipt}: {field} must be an ISO-8601 string")
    normalized = value
    fractional = re.fullmatch(
        r"(?P<prefix>.+?)[.,](?P<fraction>[0-9]+)"
        r"(?P<zone>Z|[+-][0-9]{2}:[0-9]{2})?",
        value,
    )
    if fractional is not None:
        digits = (fractional.group("fraction") + "000000")[:6]
        normalized = (
            f"{fractional.group('prefix')}.{digits}{fractional.group('zone') or ''}"
        )
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{receipt}: invalid {field}: {value!r}") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{receipt}: {field} must include a UTC offset")
    return timestamp


def _visible_gpu_count(value: Any, *, receipt: Path) -> int:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{receipt}: cuda_visible_devices must be a non-empty string")
    devices = [device.strip() for device in value.split(",")]
    if any(not device for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"{receipt}: invalid cuda_visible_devices={value!r}")
    return len(devices)


def execution_metadata(
    *, start_time: str, end_time: str, cuda_visible_devices: str
) -> dict[str, Any]:
    """Validate and normalize the resource fields shared by job receipts."""

    label = Path("<execution-metadata>")
    start = _parse_timestamp(start_time, field="start_time", receipt=label)
    end = _parse_timestamp(end_time, field="end_time", receipt=label)
    elapsed_seconds = (end - start).total_seconds()
    if elapsed_seconds <= 0:
        raise ValueError("end_time must be after start_time")
    gpu_count = _visible_gpu_count(cuda_visible_devices, receipt=label)
    devices = ",".join(device.strip() for device in cuda_visible_devices.split(","))
    return {
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "cuda_visible_devices": devices,
        "elapsed_seconds": elapsed_seconds,
        "gpu_count": gpu_count,
        "gpu_hours": elapsed_seconds * gpu_count / 3600.0,
    }


def record_job_receipt(
    *,
    label: str,
    category: str,
    outcome: str,
    start_time: str,
    end_time: str,
    cuda_visible_devices: str,
    exit_code: int,
    output: str | Path,
) -> dict[str, Any]:
    """Write an immutable resource receipt for a completed job invocation."""

    if not label:
        raise ValueError("job label must be non-empty")
    if category not in _CATEGORIES:
        raise ValueError(f"category must be one of {sorted(_CATEGORIES)}")
    if outcome not in {"completed", "aborted", "failed"}:
        raise ValueError("job receipt outcome must be completed, aborted, or failed")
    if exit_code < 0:
        raise ValueError("exit_code must be non-negative")
    if outcome == "completed" and exit_code != 0:
        raise ValueError("a completed job receipt must have exit_code 0")
    if outcome != "completed" and exit_code == 0:
        raise ValueError("an aborted or failed job receipt needs a nonzero exit_code")
    result = {
        "schema_version": 1,
        "status": "RESOURCE_JOB_RECEIPT",
        "label": label,
        "category": category,
        "outcome": outcome,
        "exit_code": exit_code,
        **execution_metadata(
            start_time=start_time,
            end_time=end_time,
            cuda_visible_devices=cuda_visible_devices,
        ),
    }
    _atomic_json(Path(output).expanduser().resolve(), result)
    return result


def _union_seconds(intervals: list[tuple[datetime, datetime]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda interval: interval[0])
    current_start, current_end = ordered[0]
    total = 0.0
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += (current_end - current_start).total_seconds()
        current_start, current_end = start, end
    return total + (current_end - current_start).total_seconds()


def _empty_totals() -> dict[str, float | int]:
    return {"jobs": 0, "wall_hours": 0.0, "gpu_hours": 0.0}


def _add_totals(
    totals: dict[str, float | int], *, elapsed_seconds: float, gpu_count: int
) -> None:
    totals["jobs"] = int(totals["jobs"]) + 1
    totals["wall_hours"] = float(totals["wall_hours"]) + elapsed_seconds / 3600.0
    totals["gpu_hours"] = (
        float(totals["gpu_hours"]) + elapsed_seconds * gpu_count / 3600.0
    )


def summarize_resource_manifest(
    manifest: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize explicitly listed job receipts without crawling a run root."""

    manifest_path = Path(manifest).resolve()
    document = _load_mapping(manifest_path)
    if document.get("schema_version") != 1:
        raise ValueError("resource manifest schema_version must be 1")
    entries = document.get("jobs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("resource manifest jobs must be a non-empty list")

    labels: set[str] = set()
    receipt_paths: set[Path] = set()
    intervals: list[tuple[datetime, datetime]] = []
    jobs: list[dict[str, Any]] = []
    total = _empty_totals()
    by_category = {category: _empty_totals() for category in sorted(_CATEGORIES)}
    by_outcome = {outcome: _empty_totals() for outcome in sorted(_OUTCOMES)}

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        label = entry.get("label")
        category = entry.get("category")
        outcome = entry.get("outcome")
        receipt_text = entry.get("receipt")
        if not isinstance(label, str) or not label:
            raise ValueError(f"jobs[{index}].label must be a non-empty string")
        if label in labels:
            raise ValueError(f"duplicate job label: {label}")
        if category not in _CATEGORIES:
            raise ValueError(f"{label}: category must be one of {sorted(_CATEGORIES)}")
        if outcome not in _OUTCOMES:
            raise ValueError(f"{label}: outcome must be one of {sorted(_OUTCOMES)}")
        if not isinstance(receipt_text, str) or not receipt_text:
            raise ValueError(f"{label}: receipt must be a non-empty path")

        receipt_path = Path(receipt_text).expanduser()
        if not receipt_path.is_absolute():
            receipt_path = manifest_path.parent / receipt_path
        receipt_path = receipt_path.resolve()
        if receipt_path in receipt_paths:
            raise ValueError(f"duplicate receipt path: {receipt_path}")
        receipt = _load_mapping(receipt_path)
        for field, expected in (
            ("label", label),
            ("category", category),
            ("outcome", outcome),
        ):
            observed = receipt.get(field)
            if observed is not None and observed != expected:
                raise ValueError(
                    f"{receipt_path}: receipt {field}={observed!r} "
                    f"disagrees with manifest {expected!r}"
                )
        start = _parse_timestamp(
            receipt.get("start_time"), field="start_time", receipt=receipt_path
        )
        end = _parse_timestamp(
            receipt.get("end_time"), field="end_time", receipt=receipt_path
        )
        elapsed_seconds = (end - start).total_seconds()
        if elapsed_seconds <= 0:
            raise ValueError(f"{receipt_path}: end_time must be after start_time")
        gpu_count = _visible_gpu_count(
            receipt.get("cuda_visible_devices"), receipt=receipt_path
        )
        derived_fields = {
            "elapsed_seconds": elapsed_seconds,
            "gpu_count": gpu_count,
            "gpu_hours": elapsed_seconds * gpu_count / 3600.0,
        }
        for field, expected in derived_fields.items():
            observed = receipt.get(field)
            if observed is not None and (
                not isinstance(observed, (int, float))
                or not math.isclose(
                    float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ValueError(
                    f"{receipt_path}: receipt {field} disagrees with timestamps/GPUs"
                )
        if outcome == "verified" and not isinstance(receipt.get("verified"), dict):
            raise ValueError(f"{receipt_path}: verified outcome needs receipt.verified")

        labels.add(label)
        receipt_paths.add(receipt_path)
        intervals.append((start, end))
        _add_totals(total, elapsed_seconds=elapsed_seconds, gpu_count=gpu_count)
        _add_totals(
            by_category[category],
            elapsed_seconds=elapsed_seconds,
            gpu_count=gpu_count,
        )
        _add_totals(
            by_outcome[outcome],
            elapsed_seconds=elapsed_seconds,
            gpu_count=gpu_count,
        )
        jobs.append(
            {
                "label": label,
                "category": category,
                "outcome": outcome,
                "receipt": {
                    "path": str(receipt_path),
                    "bytes": receipt_path.stat().st_size,
                    "sha256": _sha256_file(receipt_path),
                },
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "gpu_count": gpu_count,
                "gpu_hours": elapsed_seconds * gpu_count / 3600.0,
            }
        )

    result = {
        "schema_version": 1,
        "accounting_rule": (
            "GPU-hours are the sum of each listed job's elapsed wall time times "
            "its visible GPU count; concurrent jobs are not collapsed."
        ),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
        },
        "jobs": jobs,
        "totals": {
            **total,
            "calendar_union_hours": _union_seconds(intervals) / 3600.0,
            "by_category": by_category,
            "by_outcome": by_outcome,
        },
    }
    if output is not None:
        output_path = Path(output).resolve()
        _atomic_json(output_path, result, replace=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sum GPU-hours from an explicit manifest of timestamped receipts"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--record-job", type=Path, metavar="OUTPUT")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--category", choices=sorted(_CATEGORIES))
    parser.add_argument("--outcome", choices=("completed", "aborted", "failed"))
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    if args.manifest is not None:
        result = summarize_resource_manifest(args.manifest, output=args.output)
    else:
        required = {
            "label": args.label,
            "category": args.category,
            "outcome": args.outcome,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "cuda_visible_devices": args.cuda_visible_devices,
            "exit_code": args.exit_code,
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            parser.error("--record-job requires " + ", ".join(missing))
        result = record_job_receipt(
            **required,
            output=args.record_job,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
