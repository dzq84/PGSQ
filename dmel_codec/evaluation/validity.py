from __future__ import annotations

import math
from typing import Any, Iterable


PRIMARY_METRICS = ("visqol_speech", "pesq_wb", "speaker_cosine")
ALLOWED_PRIMARY_DOMAIN_ERRORS = {
    "pesq_wb": {"NoUtterancesError"},
    # Legacy M6 processes loaded the evaluator before the explicit domain
    # check existed and therefore retained the native binding's IndexError.
    "visqol_speech": {"IndexError", "MetricDomainError"},
}


def validate_primary_metric_domain(
    records: list[dict[str, Any]],
    *,
    required_metrics: Iterable[str] = PRIMARY_METRICS,
) -> dict[str, list[str]]:
    required = tuple(sorted(set(required_metrics)))
    unsupported = sorted(set(required) - set(PRIMARY_METRICS))
    if unsupported:
        raise ValueError(f"Unsupported primary metrics: {unsupported}")
    exclusions: dict[str, list[str]] = {}
    for record in records:
        cut_id = str(record["cut_id"])
        if bool(record.get("failed")):
            raise ValueError(f"Reconstruction failed for {cut_id}")
        failures_by_metric: dict[str, list[dict[str, Any]]] = {}
        for failure in record.get("metric_failures", []):
            metric = str(failure.get("metric"))
            if metric in required:
                failures_by_metric.setdefault(metric, []).append(failure)
        for metric in required:
            failures = failures_by_metric.get(metric, [])
            if len(failures) > 1:
                raise ValueError(f"Duplicate {metric} failures for {cut_id}")
            value = record.get("metrics", {}).get(metric)
            if not failures:
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"Missing finite {metric} value for {cut_id}")
                continue
            if value is not None:
                raise ValueError(f"{metric} has both a value and a failure for {cut_id}")
            error_type = str(failures[0].get("error_type"))
            if error_type not in ALLOWED_PRIMARY_DOMAIN_ERRORS.get(metric, set()):
                raise ValueError(
                    f"Non-domain primary metric failure for {cut_id}: {metric}:{error_type}"
                )
            if metric == "visqol_speech" and float(record["duration"]) >= 0.64:
                raise ValueError(f"ViSQOL failed inside its duration domain: {cut_id}")
            exclusions.setdefault(metric, []).append(cut_id)
    return {metric: sorted(cut_ids) for metric, cut_ids in sorted(exclusions.items())}
