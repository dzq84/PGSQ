from __future__ import annotations

import pytest

from dmel_codec.tools.mplus_direction_gate import (
    METRIC_DIRECTIONS,
    SPLITS,
    _validate_paired_document,
    assess_route,
)


def _metric(point: float, lower: float, upper: float) -> dict:
    return {
        "point_estimate": point,
        "ci95": [lower, upper],
        "contrast": "left_minus_right",
        "resamples": 10_000,
        "finite_resamples": 10_000,
        "paired_cuts": 8,
    }


def _passing_split() -> dict[str, dict]:
    return {
        "wer_reconstruction": _metric(-0.01, -0.02, -0.001),
        "wer_delta": _metric(-0.01, -0.02, -0.001),
        "visqol_speech": _metric(0.05, 0.02, 0.08),
        "pesq_wb": _metric(0.04, 0.01, 0.07),
        "speaker_cosine": _metric(0.001, -0.002, 0.004),
    }


def test_mplus_direction_gate_requires_all_metrics_on_both_domains() -> None:
    paired = {split: _passing_split() for split in SPLITS}

    result = assess_route(paired)

    assert result["eligible_before_listening"] is True
    assert result["beneficial_significant_metrics"] == 6
    assert result["harmful_significant_metrics"] == 0


@pytest.mark.parametrize(
    ("metric", "point", "lower", "upper"),
    [
        ("wer_reconstruction", 0.001, -0.001, 0.003),
        ("wer_delta", 0.001, -0.001, 0.003),
        ("visqol_speech", -0.001, -0.003, 0.001),
        ("pesq_wb", -0.001, -0.003, 0.001),
        ("speaker_cosine", -0.001, -0.003, 0.001),
    ],
)
def test_mplus_direction_gate_rejects_any_wrong_point_direction(
    metric: str, point: float, lower: float, upper: float
) -> None:
    paired = {split: _passing_split() for split in SPLITS}
    paired["dev-other"][metric] = _metric(point, lower, upper)

    result = assess_route(paired)

    assert result["eligible_before_listening"] is False


def test_mplus_direction_gate_validates_bootstrap_and_summary_difference() -> None:
    document = {"paired": _passing_split()}
    expected = {
        metric: float(document["paired"][metric]["point_estimate"])
        for metric in METRIC_DIRECTIONS
    }
    assert _validate_paired_document(
        document, expected_differences=expected, expected_cuts=8
    ) == document["paired"]

    document["paired"]["pesq_wb"]["finite_resamples"] = 9_999
    with pytest.raises(ValueError, match="Non-finite"):
        _validate_paired_document(
            document, expected_differences=expected, expected_cuts=8
        )
