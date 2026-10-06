from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median

from dmel_codec.evaluation.bitrate import CodeStatisticsAccumulator
from dmel_codec.evaluation.cache import sha256_file
from dmel_codec.evaluation.statistics import corpus_wer, load_jsonl


def summarize_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty record set")
    cut_ids = [record["cut_id"] for record in records]
    duplicates = [key for key, count in Counter(cut_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate cut IDs in per-utterance records: {duplicates[:3]}")
    invariant_fields = (
        "run_id",
        "system",
        "variant",
        "seed",
        "split",
        "config_hash",
        "checkpoint_hash",
    )
    invariants = {}
    for field in invariant_fields:
        values = {record.get(field) for record in records}
        if len(values) != 1:
            raise ValueError(f"Records disagree on {field}: {values}")
        invariants[field] = values.pop()

    metric_names = sorted(
        {name for record in records for name in record.get("metrics", {})}
    )
    metrics = {}
    for name in metric_names:
        values = [
            float(record["metrics"][name])
            for record in records
            if record.get("metrics", {}).get(name) is not None
            and math.isfinite(float(record["metrics"][name]))
        ]
        metrics[name] = {
            "count": len(values),
            "failures": len(records) - len(values),
            "mean": float(mean(values)) if values else None,
            "median": float(median(values)) if values else None,
        }
    failure_reasons = Counter(
        failure["metric"] + ":" + failure["error_type"]
        for record in records
        for failure in record.get("metric_failures", [])
    )
    result = {
        "schema_version": 1,
        **invariants,
        "utterances": len(records),
        "failed_utterances": sum(bool(record.get("failed")) for record in records),
        "metric_failures": sum(failure_reasons.values()),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "duration_seconds": float(sum(record["duration"] for record in records)),
        "corpus_wer": {
            "reference": corpus_wer(records, "reference"),
            "reconstruction": corpus_wer(records, "reconstruction"),
        },
        "metrics": metrics,
    }
    cardinality_values = {
        tuple(record.get("codebook_cardinalities", [])) for record in records
    }
    frame_rate_values = {float(record.get("token_frame_rate", 0)) for record in records}
    if len(cardinality_values) != 1 or len(frame_rate_values) != 1:
        raise ValueError("Records disagree on codec stream definition")
    cardinalities = cardinality_values.pop()
    token_frame_rate = frame_rate_values.pop()
    stream_count_values = {
        int(record.get("num_parallel_streams", len(cardinalities)))
        for record in records
    }
    if stream_count_values != {len(cardinalities)}:
        raise ValueError("Records disagree on the number of codec streams")
    result["token_frame_rate"] = token_frame_rate
    result["num_parallel_streams"] = len(cardinalities)
    result["codebook_cardinalities"] = list(cardinalities)
    code_statistics = CodeStatisticsAccumulator(cardinalities)
    for record in records:
        code_statistics.update_histograms(record.get("code_histograms", []))
    result["code_statistics"] = code_statistics.as_dict(token_frame_rate)
    result["bitrate_kbps"] = {
        "nominal": records[0].get("nominal_bitrate_kbps"),
        "entropy": result["code_statistics"]["empirical_bitrate_kbps"],
        "serialized": records[0].get("serialized_bitrate_kbps"),
    }
    total_duration = result["duration_seconds"]
    result["rtf"] = {
        "encoder": sum(float(record.get("encoder_seconds", 0)) for record in records)
        / total_duration,
        "decoder": sum(float(record.get("decoder_seconds", 0)) for record in records)
        / total_duration,
    }
    result["peak_vram_bytes"] = max(
        int(record.get("peak_vram_bytes", 0)) for record in records
    )
    complexity_values = {
        json.dumps(record.get("model_complexity", {}), sort_keys=True)
        for record in records
    }
    if len(complexity_values) != 1:
        raise ValueError("Records disagree on model complexity metadata")
    result["model_complexity"] = json.loads(complexity_values.pop())
    reference_wer = result["corpus_wer"]["reference"]
    reconstruction_wer = result["corpus_wer"]["reconstruction"]
    result["corpus_wer"]["delta"] = (
        reconstruction_wer - reference_wer
        if reference_wer is not None and reconstruction_wer is not None
        else None
    )
    return result


def write_summary(records_path: str | Path, output: str | Path) -> dict:
    records_path = Path(records_path)
    result = summarize_records(load_jsonl(records_path))
    result["per_utterance_sha256"] = sha256_file(records_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild summary.json from JSONL")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_summary(args.records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
