"""Compatibility exports; new code should import :mod:`evaluation.metrics`."""

from dmel_codec.evaluation.metrics import (
    estoi_score,
    f0_correlation,
    multi_resolution_stft_distance,
    multi_scale_log_mel_distance,
    normalize_text,
    pesq_wb_score,
    si_sdr_db,
    stoi_score,
    word_error_counts,
)

__all__ = [
    "estoi_score",
    "f0_correlation",
    "multi_resolution_stft_distance",
    "multi_scale_log_mel_distance",
    "normalize_text",
    "pesq_wb_score",
    "si_sdr_db",
    "stoi_score",
    "word_error_counts",
]
