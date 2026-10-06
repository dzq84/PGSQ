from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import jiwer
import numpy as np
import torch
import torchaudio
from pesq import PesqError, pesq
from pystoi import stoi


ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
ASR_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SPEAKER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
VISQOL_SOURCE = "https://github.com/google/visqol"
VISQOL_REVISION = "38d0b0163e441047d4429bf07ad09e5b9031d02c"
VISQOL_MIN_SAMPLES_16K = 10_240


class MetricDomainError(ValueError):
    """The metric is undefined for an otherwise valid evaluation item."""


TEXT_NORMALIZER = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def normalize_text(text: str) -> str:
    return TEXT_NORMALIZER(text)


def whisper_english_normalizer(
    spelling_normalizer: Mapping[str, str],
) -> Callable[[str], str]:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer(dict(spelling_normalizer))
    bracket_translation = str.maketrans(
        {character: " " for character in "[](){}"}
    )

    def normalize(text: str) -> str:
        # LibriTTS uses brackets for some spoken editorial text, e.g. image captions.
        return normalizer(text.translate(bracket_translation))

    return normalize


def word_error_counts(
    reference: str,
    hypothesis: str,
    *,
    normalizer: Callable[[str], str] = normalize_text,
    include_raw: bool = False,
) -> dict[str, int | str]:
    normalized_reference = normalizer(reference)
    normalized_hypothesis = normalizer(hypothesis)
    if not normalized_reference:
        raise MetricDomainError("ASR reference is empty after text normalization")
    output = jiwer.process_words(normalized_reference, normalized_hypothesis)
    result: dict[str, int | str] = {
        "reference": normalized_reference,
        "hypothesis": normalized_hypothesis,
        "hits": int(output.hits),
        "substitutions": int(output.substitutions),
        "deletions": int(output.deletions),
        "insertions": int(output.insertions),
        "reference_words": int(output.hits + output.substitutions + output.deletions),
    }
    if include_raw:
        result["raw_reference"] = reference
        result["raw_hypothesis"] = hypothesis
    return result


def _mono_numpy(waveform: torch.Tensor) -> np.ndarray:
    waveform = waveform.detach().float().cpu().squeeze()
    if waveform.ndim != 1:
        raise ValueError(f"Expected one mono waveform, got {tuple(waveform.shape)}")
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform contains NaN or Inf")
    return waveform.numpy().astype(np.float64, copy=False)


def resample_mono(
    waveform: torch.Tensor, source_rate: int, target_rate: int = 16_000
) -> torch.Tensor:
    waveform = waveform.detach().float().cpu().reshape(1, -1)
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return waveform.squeeze(0)


def stoi_score(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int) -> float:
    reference_16k = _mono_numpy(resample_mono(reference, sample_rate))
    degraded_16k = _mono_numpy(resample_mono(degraded, sample_rate))
    return float(stoi(reference_16k, degraded_16k, 16_000, extended=False))


def estoi_score(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int) -> float:
    reference_16k = _mono_numpy(resample_mono(reference, sample_rate))
    degraded_16k = _mono_numpy(resample_mono(degraded, sample_rate))
    return float(stoi(reference_16k, degraded_16k, 16_000, extended=True))


def pesq_wb_score(
    reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int
) -> float:
    reference_16k = _mono_numpy(resample_mono(reference, sample_rate))
    degraded_16k = _mono_numpy(resample_mono(degraded, sample_rate))
    value = pesq(
        16_000,
        reference_16k,
        degraded_16k,
        mode="wb",
        on_error=PesqError.RAISE_EXCEPTION,
    )
    return float(value)


def si_sdr_db(reference: torch.Tensor, degraded: torch.Tensor, eps: float = 1e-8) -> float:
    reference = reference.detach().double().flatten()
    degraded = degraded.detach().double().flatten()
    reference = reference - reference.mean()
    degraded = degraded - degraded.mean()
    scale = torch.dot(degraded, reference) / (torch.dot(reference, reference) + eps)
    target = scale * reference
    noise = degraded - target
    ratio = torch.dot(target, target) / (torch.dot(noise, noise) + eps)
    return float((10 * torch.log10(ratio + eps)).item())


def multi_resolution_stft_distance(
    reference: torch.Tensor,
    degraded: torch.Tensor,
    resolutions: Sequence[tuple[int, int, int]] = (
        (512, 120, 512),
        (1024, 240, 1024),
        (2048, 480, 2048),
    ),
) -> float:
    reference = reference.detach().float().flatten()
    degraded = degraded.detach().float().flatten()
    distances = []
    for n_fft, hop, window_length in resolutions:
        window = torch.hann_window(window_length, device=reference.device)
        reference_stft = torch.stft(
            reference,
            n_fft=n_fft,
            hop_length=hop,
            win_length=window_length,
            window=window,
            return_complex=True,
        ).abs()
        degraded_stft = torch.stft(
            degraded,
            n_fft=n_fft,
            hop_length=hop,
            win_length=window_length,
            window=window,
            return_complex=True,
        ).abs()
        spectral_convergence = torch.linalg.vector_norm(reference_stft - degraded_stft) / (
            torch.linalg.vector_norm(reference_stft) + 1e-7
        )
        log_distance = (torch.log(reference_stft + 1e-7) - torch.log(degraded_stft + 1e-7)).abs().mean()
        distances.append(spectral_convergence + log_distance)
    return float(torch.stack(distances).mean().item())


def multi_scale_log_mel_distance(
    reference: torch.Tensor,
    degraded: torch.Tensor,
    sample_rate: int,
    fft_sizes: Sequence[int] = (512, 1024, 2048),
) -> float:
    reference = reference.detach().float().reshape(1, -1)
    degraded = degraded.detach().float().reshape(1, -1)
    distances = []
    for n_fft in fft_sizes:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=n_fft // 4,
            n_mels=80,
            f_min=0,
            f_max=sample_rate / 2,
            power=1.0,
        )
        reference_mel = transform(reference)
        degraded_mel = transform(degraded)
        distance = (
            torch.log(reference_mel + 1e-5) - torch.log(degraded_mel + 1e-5)
        ).abs().mean()
        distances.append(distance)
    return float(torch.stack(distances).mean().item())


def f0_correlation(
    reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int
) -> float:
    reference = reference.detach().float().cpu().reshape(1, -1)
    degraded = degraded.detach().float().cpu().reshape(1, -1)
    reference_f0 = torchaudio.functional.detect_pitch_frequency(
        reference, sample_rate=sample_rate
    ).flatten()
    degraded_f0 = torchaudio.functional.detect_pitch_frequency(
        degraded, sample_rate=sample_rate
    ).flatten()
    frames = min(reference_f0.numel(), degraded_f0.numel())
    reference_f0 = reference_f0[:frames]
    degraded_f0 = degraded_f0[:frames]
    valid = (reference_f0 > 0) & (degraded_f0 > 0)
    if int(valid.sum()) < 2:
        raise ValueError("F0 correlation requires at least two jointly voiced frames")
    reference_f0 = reference_f0[valid]
    degraded_f0 = degraded_f0[valid]
    if reference_f0.std() == 0 or degraded_f0.std() == 0:
        raise ValueError("F0 correlation is undefined for a constant track")
    return float(torch.corrcoef(torch.stack((reference_f0, degraded_f0)))[0, 1])


class VisqolSpeechMetric:
    def __init__(self) -> None:
        try:
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2
        except ImportError as exc:
            raise RuntimeError(
                "Official ViSQOL Python bindings are not installed; build pinned v3.3.3"
            ) from exc
        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = 16_000
        config.options.use_speech_scoring = True
        model = (
            Path(visqol_lib_py.__file__).parent
            / "model"
            / "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        )
        if not model.is_file():
            raise FileNotFoundError(f"ViSQOL speech model is absent: {model}")
        config.options.svr_model_path = str(model)
        self.api = visqol_lib_py.VisqolApi()
        self.api.Create(config)

    def __call__(
        self, reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int
    ) -> float:
        reference_16k = _mono_numpy(resample_mono(reference, sample_rate))
        degraded_16k = _mono_numpy(resample_mono(degraded, sample_rate))
        shortest = min(reference_16k.size, degraded_16k.size)
        if shortest < VISQOL_MIN_SAMPLES_16K:
            raise MetricDomainError(
                "ViSQOL speech requires at least 10240 samples at 16 kHz "
                f"(0.64 s); got {shortest}"
            )
        result = self.api.Measure(reference_16k, degraded_16k)
        return float(result.moslqo)


class SpeakerSimilarityMetric:
    model_id = SPEAKER_MODEL_ID
    revision = SPEAKER_MODEL_REVISION

    def __init__(self, cache_dir: str | Path, device: str | torch.device) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        fetch_config = FetchConfig(
            allow_updates=False,
            revision=self.revision,
            huggingface_cache_dir=Path(cache_dir) / "huggingface",
        )
        self.model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=Path(cache_dir) / "speechbrain" / self.revision,
            run_opts={"device": str(self.device)},
            local_strategy=LocalStrategy.COPY,
            fetch_config=fetch_config,
        )
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self, reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int
    ) -> float:
        reference = resample_mono(reference, sample_rate).to(self.device).unsqueeze(0)
        degraded = resample_mono(degraded, sample_rate).to(self.device).unsqueeze(0)
        reference_embedding = self.model.encode_batch(reference).flatten(1)
        degraded_embedding = self.model.encode_batch(degraded).flatten(1)
        return float(
            torch.nn.functional.cosine_similarity(
                reference_embedding, degraded_embedding, dim=1
            ).item()
        )


class WhisperASR:
    model_id = ASR_MODEL_ID
    revision = ASR_MODEL_REVISION
    sample_rate = 16_000
    decoding_parameters = {
        "do_sample": False,
        "num_beams": 1,
        "language": "en",
        "task": "transcribe",
    }

    def __init__(self, cache_dir: str | Path, device: str | torch.device) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = torch.device(device)
        cache_dir = Path(cache_dir) / "huggingface"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=cache_dir
        )
        self.processor.feature_extractor.return_attention_mask = True
        self.normalize_text = whisper_english_normalizer(
            self.processor.tokenizer.english_spelling_normalizer
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        ).to(self.device).eval()
        device_argument = self.device.index if self.device.type == "cuda" else -1
        if self.device.type == "cuda" and device_argument is None:
            device_argument = 0
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=dtype,
            device=device_argument,
            chunk_length_s=30,
            stride_length_s=5,
        )

    def transcribe(
        self, waveforms: Iterable[torch.Tensor], source_rate: int, batch_size: int = 4
    ) -> list[str]:
        inputs = [
            _mono_numpy(resample_mono(waveform, source_rate, self.sample_rate)).astype(
                np.float32
            )
            for waveform in waveforms
        ]
        outputs = self.pipeline(
            inputs,
            batch_size=batch_size,
            generate_kwargs=dict(self.decoding_parameters),
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [str(output["text"]) for output in outputs]


def metric_metadata() -> dict[str, Any]:
    return {
        "sample_rate_policy": {
            "visqol_speech": 16_000,
            "pesq_wb": 16_000,
            "stoi": 16_000,
            "estoi": 16_000,
            "speaker": 16_000,
            "asr": 16_000,
        },
        "metric_domain_policy": {
            "visqol_speech": {
                "minimum_samples_16k": VISQOL_MIN_SAMPLES_16K,
                "minimum_duration_seconds": VISQOL_MIN_SAMPLES_16K / 16_000,
                "excluded_value": None,
            },
            "pesq_wb": {
                "no_utterances_error": "excluded_value",
                "excluded_value": None,
            },
            "wer": {
                "empty_normalized_reference": "excluded_value",
                "excluded_value": None,
            },
            "comparison": "identical_per_metric_cut_sets",
        },
        "asr": {
            "model": ASR_MODEL_ID,
            "revision": ASR_MODEL_REVISION,
            "decoding_parameters": dict(WhisperASR.decoding_parameters),
            "return_attention_mask": True,
            "chunk_length_seconds": 30,
            "stride_length_seconds": 5,
            "normalizer": [
                "transformers Whisper EnglishTextNormalizer",
                "tokenizer English spelling map",
                "preserve LibriTTS bracketed content",
            ],
            "store_raw_reference_and_hypothesis": True,
        },
        "speaker": {
            "model": SPEAKER_MODEL_ID,
            "revision": SPEAKER_MODEL_REVISION,
        },
        "visqol": {
            "source": VISQOL_SOURCE,
            "revision": VISQOL_REVISION,
            "mode": "speech",
            "scaled_mos": True,
        },
    }
