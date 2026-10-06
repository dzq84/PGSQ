from __future__ import annotations

import pytest
import torch

from dmel_codec.evaluation.metrics import (
    MetricDomainError,
    VisqolSpeechMetric,
    estoi_score,
    f0_correlation,
    metric_metadata,
    multi_resolution_stft_distance,
    multi_scale_log_mel_distance,
    normalize_text,
    pesq_wb_score,
    si_sdr_db,
    stoi_score,
    whisper_english_normalizer,
    word_error_counts,
)


def _signal() -> torch.Tensor:
    generator = torch.Generator().manual_seed(2027)
    noise = torch.randn(1, 72_000, generator=generator) * 0.03
    kernel = torch.ones(1, 1, 31) / 31
    return torch.nn.functional.conv1d(
        torch.nn.functional.pad(noise.unsqueeze(0), (15, 15), mode="reflect"),
        kernel,
    ).squeeze(0)


def test_text_normalization_and_edit_counts_are_explicit() -> None:
    assert normalize_text("Hello,  WORLD!") == "hello world"
    counts = word_error_counts("ONE TWO THREE", "one four three extra")
    assert counts == {
        "reference": "one two three",
        "hypothesis": "one four three extra",
        "hits": 2,
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
        "reference_words": 3,
    }


def test_whisper_english_normalizer_preserves_spoken_bracket_content() -> None:
    normalizer = whisper_english_normalizer({})
    assert normalizer("Well--thereabouts") == "well thereabouts"
    assert normalizer("[Illustration: CRUMPETS.]") == "illustration crumpets"
    counts = word_error_counts(
        "Twenty-two crumpets",
        "22 crumpets",
        normalizer=normalizer,
        include_raw=True,
    )
    assert counts["reference"] == "22 crumpets"
    assert counts["hypothesis"] == "22 crumpets"
    assert counts["raw_reference"] == "Twenty-two crumpets"
    assert counts["raw_hypothesis"] == "22 crumpets"

    with pytest.raises(MetricDomainError, match="empty after text normalization"):
        word_error_counts("Hmm.", "hmm", normalizer=normalizer)


def test_identity_intrusive_metrics_reach_their_expected_ceiling() -> None:
    signal = _signal()
    assert stoi_score(signal, signal, 24_000) == pytest.approx(1.0, abs=1e-6)
    assert estoi_score(signal, signal, 24_000) == pytest.approx(1.0, abs=1e-6)
    assert pesq_wb_score(signal, signal, 24_000) > 4.5
    assert multi_resolution_stft_distance(signal, signal) == pytest.approx(0.0)
    assert multi_scale_log_mel_distance(signal, signal, 24_000) == pytest.approx(0.0)
    assert si_sdr_db(signal, signal) > 80


def test_metric_metadata_pins_models_and_sampling_policy() -> None:
    metadata = metric_metadata()
    assert metadata["asr"] == {
        "model": "openai/whisper-large-v3-turbo",
        "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "decoding_parameters": {
            "do_sample": False,
            "num_beams": 1,
            "language": "en",
            "task": "transcribe",
        },
        "return_attention_mask": True,
        "chunk_length_seconds": 30,
        "stride_length_seconds": 5,
        "normalizer": [
            "transformers Whisper EnglishTextNormalizer",
            "tokenizer English spelling map",
            "preserve LibriTTS bracketed content",
        ],
        "store_raw_reference_and_hypothesis": True,
    }
    assert len(metadata["speaker"]["revision"]) == 40
    assert metadata["sample_rate_policy"]["pesq_wb"] == 16_000
    assert metadata["visqol"]["mode"] == "speech"
    assert metadata["metric_domain_policy"]["visqol_speech"] == {
        "minimum_samples_16k": 10_240,
        "minimum_duration_seconds": 0.64,
        "excluded_value": None,
    }
    assert metadata["metric_domain_policy"]["wer"] == {
        "empty_normalized_reference": "excluded_value",
        "excluded_value": None,
    }
    assert metadata["metric_domain_policy"]["comparison"] == (
        "identical_per_metric_cut_sets"
    )


def test_identity_f0_correlation_reaches_one_for_varying_pitch() -> None:
    sample_rate = 24_000
    time = torch.arange(sample_rate * 3) / sample_rate
    phase = 2 * torch.pi * (120 * time + 20 * time.square())
    signal = (0.1 * torch.sin(phase)).unsqueeze(0)
    assert f0_correlation(signal, signal, sample_rate) == pytest.approx(1.0)


def test_visqol_rejects_audio_below_its_measured_minimum_before_native_call() -> None:
    metric = object.__new__(VisqolSpeechMetric)
    signal = torch.zeros(1, 10_239)
    with pytest.raises(MetricDomainError, match="0.64 s"):
        metric(signal, signal, 16_000)
