from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torchaudio
import yaml
from lhotse import CutSet
from torch.nn import functional as F

from dmel_codec.evaluation.adapters import (
    DACAdapter,
    DmelAdaptedDecoderAdapter,
    DmelSameFrontendAdapter,
    GroundTruthControl,
    MelOracleControl,
    OursAdapter,
)
from dmel_codec.evaluation.adapters.base import CodecAdapter, Reconstruction
from dmel_codec.evaluation.bitrate import CodeStatisticsAccumulator
from dmel_codec.evaluation.cache import (
    JsonCache,
    ReconstructionCache,
    canonical_hash,
    reconstruction_cache_key,
    sha256_file,
)
from dmel_codec.evaluation.dependency import CodeDependencyAccumulator
from dmel_codec.evaluation.metrics import (
    SpeakerSimilarityMetric,
    VisqolSpeechMetric,
    WhisperASR,
    estoi_score,
    f0_correlation,
    metric_metadata,
    multi_resolution_stft_distance,
    multi_scale_log_mel_distance,
    pesq_wb_score,
    si_sdr_db,
    stoi_score,
    word_error_counts,
)
from dmel_codec.evaluation.summarize import write_summary


SUPPORTED_METRICS = {
    "stoi",
    "estoi",
    "pesq_wb",
    "visqol_speech",
    "speaker_cosine",
    "si_sdr",
    "mr_stft",
    "ms_log_mel",
    "wer",
    "f0_correlation",
}
DEFAULT_METRICS = ",".join(sorted(SUPPORTED_METRICS))


@dataclass
class BatchItem:
    cut_id: str
    speaker_id: str
    text: str
    waveform: torch.Tensor
    audio_length: int
    duration: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _iter_batches(cuts: Sequence, max_duration: float) -> Iterable[list]:
    batch = []
    duration = 0.0
    for cut in cuts:
        if batch and duration + cut.duration > max_duration:
            yield batch
            batch = []
            duration = 0.0
        batch.append(cut)
        duration += cut.duration
    if batch:
        yield batch


def _load_batch(cuts: Sequence) -> list[BatchItem]:
    result = []
    for cut in cuts:
        waveform = torch.from_numpy(cut.load_audio()).float()
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(f"Expected mono audio for {cut.id}")
        supervision = cut.supervisions[0] if cut.supervisions else None
        if supervision is None or not supervision.text or not supervision.speaker:
            raise ValueError(f"Evaluation cut {cut.id} lacks text or speaker")
        result.append(
            BatchItem(
                cut_id=cut.id,
                speaker_id=str(supervision.speaker),
                text=str(supervision.text),
                waveform=waveform,
                audio_length=waveform.shape[-1],
                duration=waveform.shape[-1] / 24_000,
            )
        )
    return result


def _collate(
    items: Sequence[BatchItem],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    maximum = max(item.audio_length for item in items)
    waveform = torch.stack(
        [F.pad(item.waveform, (0, maximum - item.audio_length)) for item in items]
    )
    lengths = torch.tensor([item.audio_length for item in items], dtype=torch.long)
    return waveform, lengths, [item.cut_id for item in items]


def _timed_reconstruct(
    system,
    waveform: torch.Tensor,
    lengths: torch.Tensor,
    cut_ids: list[str],
    device: torch.device,
) -> tuple[Reconstruction, float, float, int]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if isinstance(system, CodecAdapter):
        _synchronize(device)
        start = time.perf_counter()
        codes, code_lengths = system.encode(waveform, lengths)
        _synchronize(device)
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reconstructed = system.decode(codes, lengths, cut_ids)
        _synchronize(device)
        decode_seconds = time.perf_counter() - start
        system.validate_reconstruction(reconstructed, waveform, lengths)
        result = Reconstruction(reconstructed, codes, code_lengths)
    else:
        _synchronize(device)
        start = time.perf_counter()
        result = system.reconstruct(waveform, lengths, cut_ids)
        _synchronize(device)
        decode_seconds = time.perf_counter() - start
        encode_seconds = 0.0
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return result, encode_seconds, decode_seconds, peak_vram_bytes


def _split_reconstruction(
    reconstruction: Reconstruction,
    lengths: torch.Tensor,
    encode_seconds: float,
    decode_seconds: float,
    peak_vram_bytes: int,
) -> list[dict]:
    total_samples = int(lengths.sum().item())
    result = []
    for index, audio_length in enumerate(lengths.tolist()):
        code_length = int(reconstruction.code_lengths[index].item())
        weight = audio_length / total_samples
        result.append(
            {
                "waveform": reconstruction.waveform[index, :, :audio_length]
                .detach()
                .float()
                .cpu(),
                "codes": reconstruction.codes[index, :, :code_length]
                .detach()
                .long()
                .cpu(),
                "code_length": code_length,
                "audio_length": audio_length,
                "encoder_seconds": encode_seconds * weight,
                "decoder_seconds": decode_seconds * weight,
                "peak_vram_bytes": peak_vram_bytes,
            }
        )
    return result


def _reconstruct_with_fallback(
    system,
    items: Sequence[BatchItem],
    device: torch.device,
) -> list[dict | Exception]:
    waveform, lengths, cut_ids = _collate(items)
    try:
        reconstruction, encode_seconds, decode_seconds, peak_vram_bytes = (
            _timed_reconstruct(system, waveform, lengths, cut_ids, device)
        )
        return _split_reconstruction(
            reconstruction, lengths, encode_seconds, decode_seconds, peak_vram_bytes
        )
    except Exception as batch_error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(items) == 1:
            return [batch_error]
        results: list[dict | Exception] = []
        for item in items:
            try:
                one_waveform, one_lengths, one_ids = _collate([item])
                reconstruction, encode_seconds, decode_seconds, peak_vram_bytes = (
                    _timed_reconstruct(
                        system, one_waveform, one_lengths, one_ids, device
                    )
                )
                results.extend(
                    _split_reconstruction(
                        reconstruction,
                        one_lengths,
                        encode_seconds,
                        decode_seconds,
                        peak_vram_bytes,
                    )
                )
            except Exception as item_error:
                results.append(item_error)
        return results


def _error(metric: str, exc: Exception) -> dict[str, str]:
    return {
        "metric": metric,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _metric_cache_key(
    *,
    metric: str,
    metric_metadata_hash: str,
    metrics_source_hash: str,
    manifest_hash: str,
    cut_id: str,
    reconstruction_key: str,
) -> str:
    return canonical_hash(
        {
            "schema": 1,
            "metric": metric,
            "metric_metadata_hash": metric_metadata_hash,
            "metrics_source_hash": metrics_source_hash,
            "manifest_hash": manifest_hash,
            "cut_id": cut_id,
            "reconstruction_cache_key": reconstruction_key,
        }
    )


def _cached_metric_value(
    cache: JsonCache, key: str, compute: Callable[[], float]
) -> float:
    cached = cache.load(key)
    if cached is not None:
        return float(cached["value"])
    value = float(compute())
    if not math.isfinite(value):
        raise ValueError(f"Metric produced a non-finite value: {value}")
    cache.save(key, {"value": value})
    return value


def _code_histograms(
    codes: torch.Tensor, cardinalities: Sequence[int]
) -> list[list[int]]:
    if codes.shape[0] != len(cardinalities):
        raise ValueError("Code stream count disagrees with metadata")
    return [
        torch.bincount(codes[stream], minlength=cardinality).tolist()
        for stream, cardinality in enumerate(cardinalities)
    ]


def _transcribe_with_fallback(
    asr: WhisperASR, waveforms: list[torch.Tensor], sample_rate: int
) -> list[str | Exception]:
    if not waveforms:
        return []
    try:
        return asr.transcribe(waveforms, sample_rate)
    except Exception:
        results: list[str | Exception] = []
        for waveform in waveforms:
            try:
                results.append(asr.transcribe([waveform], sample_rate, batch_size=1)[0])
            except Exception as exc:
                results.append(exc)
        return results


def _checkpoint_hash(metadata: dict) -> str | None:
    checkpoint = metadata.get("checkpoint_revision_and_hash", {})
    if not checkpoint:
        return None
    if checkpoint == {"control": "identity", "checkpoint": None}:
        return None
    return canonical_hash(checkpoint)


def _decode_policy(system) -> str:
    embedded = getattr(system, "embedded_metadata", {})
    definition = embedded.get("codec_definition", {})
    return str(definition.get("decoder_input_policy", "deterministic-v1"))


def _git_state(repository: Path) -> dict:
    def command(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository, text=True
        ).strip()

    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repository)
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_short": command("status", "--short").splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty": bool(command("status", "--short")),
    }


def _environment() -> dict:
    packages = {}
    for name in (
        "torch",
        "torchaudio",
        "jiwer",
        "pesq",
        "pystoi",
        "speechbrain",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "packages": packages,
    }


def _asset_manifest(path: Path | None, *, revision: str) -> dict:
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"Pinned metric asset metadata is required for revision {revision}"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("revision") != revision:
        raise ValueError(
            f"Asset manifest revision {document.get('revision')} does not match {revision}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "metadata": document,
    }


def _make_system(args: argparse.Namespace, device: torch.device):
    if args.system == "identity":
        return GroundTruthControl()
    if args.system == "mel_oracle":
        return MelOracleControl(
            bigvgan_checkpoint=args.bigvgan_checkpoint,
            bigvgan_config=args.bigvgan_config,
            device=device,
        )
    if args.system == "dmel":
        return DmelSameFrontendAdapter(
            range_metadata=args.dmel_range,
            bigvgan_checkpoint=args.bigvgan_checkpoint,
            bigvgan_config=args.bigvgan_config,
            device=device,
        )
    if args.system == "dmel_decoder":
        return DmelAdaptedDecoderAdapter(
            decoder_checkpoint=args.dmel_decoder_checkpoint,
            range_metadata=args.dmel_range,
            bigvgan_checkpoint=args.bigvgan_checkpoint,
            bigvgan_config=args.bigvgan_config,
            device=device,
        )
    if args.system == "dac":
        return DACAdapter(
            checkpoint=args.dac_checkpoint,
            n_quantizers=args.n_quantizers,
            device=device,
        )
    if args.system == "ours":
        return OursAdapter(
            checkpoint=args.ours_checkpoint,
            bigvgan_checkpoint=args.bigvgan_checkpoint,
            bigvgan_config=args.bigvgan_config,
            device=device,
        )
    raise ValueError(args.system)


def _validate_frozen_paper_display(registry: dict[str, Any]) -> None:
    display = registry.get("paper_display")
    known_roles = registry.get("roles")
    if not isinstance(display, dict) or display.get("schema_version") != 1:
        raise ValueError("Frozen test gate lacks paper-display schema v1")
    if not isinstance(known_roles, list):
        raise ValueError("Frozen test gate lacks final-system roles")
    for field in (
        "main_table_roles",
        "ablation_table_roles",
        "rate_quality_roles",
    ):
        values = display.get(field)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or not set(values) <= set(known_roles)
        ):
            raise ValueError(f"Frozen test gate has invalid paper_display.{field}")


def _validate_test_gate(
    args: argparse.Namespace,
    *,
    metrics: set[str] | None = None,
    manifest_hash: str | None = None,
    config_hash: str | None = None,
    checkpoint_hash: str | None = None,
    metric_metadata_hash: str | None = None,
    system_name: str | None = None,
    system_seed: int | None = None,
    decode_seed_or_policy: str | None = None,
) -> str | None:
    if not args.split.startswith("test"):
        return None
    if args.max_cuts is not None:
        raise ValueError("Test evaluation cannot use --max-cuts")
    if args.test_gate is None or not args.test_gate.is_file():
        raise FileNotFoundError("Frozen test evaluation requires --test-gate")
    if config_hash is None:
        return sha256_file(args.test_gate)
    try:
        gate_bytes = args.test_gate.read_bytes()
        gate = json.loads(gate_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("Frozen test gate must be a JSON object") from exc
    gate_sha256 = hashlib.sha256(gate_bytes).hexdigest()
    if gate.get("schema_version") != 3 or gate.get("status") != "TEST_GATE_FROZEN":
        raise ValueError("Invalid or incomplete frozen test gate")
    registry = gate.get("final_system_registry")
    if (
        not isinstance(registry, dict)
        or registry.get("status") != "FINAL_SYSTEMS_PREREGISTERED"
        or not registry.get("sha256")
        or not registry.get("requirements_sha256")
        or not registry.get("roles")
    ):
        raise ValueError("Frozen test gate lacks a complete final-system registry")
    registered_roles = registry["roles"]
    system_roles = [
        role for entry in gate.get("systems", []) for role in entry.get("roles", [])
    ]
    if (
        len(system_roles) != len(set(system_roles))
        or sorted(system_roles) != registered_roles
        or registry.get("unique_system_identities") != len(gate.get("systems", []))
    ):
        raise ValueError("Frozen test gate has inconsistent final-system coverage")
    _validate_frozen_paper_display(registry)
    _validate_training_provenance(gate)
    expected_manifest = gate.get("test_manifests", {}).get(args.split)
    if expected_manifest != manifest_hash:
        raise ValueError(
            f"Test manifest hash is not frozen for {args.split}: "
            f"{manifest_hash} != {expected_manifest}"
        )
    if sorted(metrics or ()) != gate.get("metrics"):
        raise ValueError("Requested metric set does not match the frozen test gate")
    if metric_metadata_hash != gate.get("metric_metadata_hash"):
        raise ValueError("Metric revisions/assets do not match the frozen test gate")
    matches = [
        entry
        for entry in gate.get("systems", [])
        if entry.get("adapter") == args.system
        and entry.get("system") == system_name
        and int(entry.get("seed"))
        == int(args.seed if system_seed is None else system_seed)
        and entry.get("config_hash") == config_hash
        and entry.get("checkpoint_hash") == checkpoint_hash
        and (
            decode_seed_or_policy is None
            or entry.get("decode_seed_or_policy") == decode_seed_or_policy
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "System/config/checkpoint/seed is not uniquely registered in the test gate"
        )
    repository_root = Path(__file__).resolve().parents[2]
    if not gate.get("allow_dirty_dev_runs", False):
        current_git = _git_state(repository_root)
        if current_git.get("commit") != gate.get("git_commit"):
            raise ValueError("Current git commit does not match the frozen test gate")
        if current_git.get("dirty"):
            raise ValueError("Formal test evaluation requires a clean frozen worktree")
    for relative_path, expected_hash in gate.get("source_hashes", {}).items():
        path = repository_root / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen source hash mismatch: {relative_path}")
    return gate_sha256


def _validate_training_provenance(gate: dict[str, Any]) -> None:
    systems = gate.get("systems", [])
    by_role = {role: entry for entry in systems for role in entry.get("roles", [])}
    required_roles = {
        "base_m_seed2026",
        "flagship_m_seed2026",
        "flagship_m_seed2027",
        "flagship_m_seed2028",
        "ours_075_seed2026",
    }
    if not required_roles <= set(by_role):
        return
    provenance = gate.get("training_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Frozen test gate lacks final training provenance")
    mplus = provenance.get("mplus")
    ours_075 = provenance.get("ours_075")
    if not isinstance(mplus, dict) or not isinstance(ours_075, dict):
        raise ValueError("Frozen test gate has incomplete training provenance")

    def verify_artifact(record: Any, label: str) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"Frozen {label} provenance is invalid")
        raw_path = record.get("path")
        expected = record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ValueError(f"Frozen {label} provenance is incomplete")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen {label} provenance hash mismatch")

    verify_artifact(mplus.get("direction_selection"), "M+ direction selection")
    selected = mplus.get("selected_direction")
    selected_system = mplus.get("selected_system")
    if selected not in {"source", "codec", "vocoder", "joint"}:
        raise ValueError("Frozen M+ provenance has an invalid selected direction")
    flagship_roles = [f"flagship_m_seed{seed}" for seed in (2026, 2027, 2028)]
    if any(by_role[role].get("system") != selected_system for role in flagship_roles):
        raise ValueError("Frozen flagship systems disagree with M+ provenance")
    source_hashes = mplus.get("source_checkpoint_sha256_by_seed")
    if not isinstance(source_hashes, dict) or set(source_hashes) != {
        "2026",
        "2027",
        "2028",
    }:
        raise ValueError("Frozen M+ provenance lacks all source checkpoints")
    if by_role["base_m_seed2026"].get("raw_checkpoint_sha256") != source_hashes["2026"]:
        raise ValueError("Frozen Base-M checkpoint disagrees with M+ provenance")
    replications = mplus.get("replication_completions")
    if not isinstance(replications, dict):
        raise ValueError("Frozen M+ replication provenance is invalid")
    if selected == "source":
        if replications:
            raise ValueError("Source-retained M+ provenance contains replications")
        expected_flagship_hashes = source_hashes
    else:
        if set(replications) != {"2027", "2028"}:
            raise ValueError("Promoted M+ provenance lacks both replications")
        for seed, record in replications.items():
            verify_artifact(record, f"M+ seed-{seed} completion")
        expected_flagship_hashes = {
            "2026": mplus.get("selected_seed2026_checkpoint_sha256"),
            "2027": replications["2027"].get("best_checkpoint_sha256"),
            "2028": replications["2028"].get("best_checkpoint_sha256"),
        }
    for seed in (2026, 2027, 2028):
        role = f"flagship_m_seed{seed}"
        if (
            by_role[role].get("raw_checkpoint_sha256")
            != expected_flagship_hashes[str(seed)]
        ):
            raise ValueError(f"Frozen {role} checkpoint disagrees with provenance")

    verify_artifact(ours_075.get("formal_release"), "Ours-0.75 release")
    verify_artifact(
        ours_075.get("training_completion"), "Ours-0.75 training completion"
    )
    if by_role["ours_075_seed2026"].get("raw_checkpoint_sha256") != ours_075.get(
        "best_checkpoint_sha256"
    ):
        raise ValueError("Frozen Ours-0.75 checkpoint disagrees with provenance")


def _ours_model_complexity(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    discriminator_prefixes = ("discriminator.", "waveform_discriminator.")
    inference_prefixes = ("vocoder.", *discriminator_prefixes)
    codec_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith(inference_prefixes)
    )
    trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith(inference_prefixes) and parameter.requires_grad
    )
    training_only = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(discriminator_prefixes)
    )
    external = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("vocoder.")
    )
    return {
        "codec_parameters": codec_parameters,
        "codec_trainable_parameters": trainable,
        "training_only_parameters": training_only,
        "external_pretrained_parameters": external,
        "training_checkpoint_size_bytes": checkpoint.stat().st_size,
        "training_checkpoint_includes_optimizer_state": True,
    }


def _model_complexity(system) -> dict[str, Any]:
    def count(module: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    if isinstance(system, OursAdapter):
        return _ours_model_complexity(system.model, system.checkpoint)
    if isinstance(system, DACAdapter):
        return {
            "codec_parameters": 0,
            "codec_trainable_parameters": 0,
            "training_only_parameters": 0,
            "external_pretrained_parameters": count(system.model),
            "training_checkpoint_size_bytes": None,
            "pretrained_checkpoint_size_bytes": system.checkpoint.stat().st_size,
        }
    if isinstance(system, DmelAdaptedDecoderAdapter):
        return {
            "codec_parameters": count(system.mel_decoder),
            "codec_trainable_parameters": count(system.mel_decoder),
            "training_only_parameters": 0,
            "external_pretrained_parameters": count(system.vocoder),
            "training_checkpoint_size_bytes": system.decoder_checkpoint.stat().st_size,
        }
    if isinstance(system, DmelSameFrontendAdapter):
        return {
            "codec_parameters": 0,
            "codec_trainable_parameters": 0,
            "training_only_parameters": 0,
            "external_pretrained_parameters": count(system.vocoder),
            "training_checkpoint_size_bytes": None,
        }
    if isinstance(system, MelOracleControl):
        return {
            "codec_parameters": 0,
            "codec_trainable_parameters": 0,
            "training_only_parameters": 0,
            "external_pretrained_parameters": count(system.vocoder),
            "training_checkpoint_size_bytes": None,
        }
    return {
        "codec_parameters": 0,
        "codec_trainable_parameters": 0,
        "training_only_parameters": 0,
        "external_pretrained_parameters": 0,
        "training_checkpoint_size_bytes": None,
    }


def _result_seed(system: CodecAdapter, analysis_seed: int) -> int:
    if isinstance(system, OursAdapter):
        return int(system.config.seed)
    return int(analysis_seed)


def evaluate(args: argparse.Namespace) -> dict:
    metrics = {name.strip() for name in args.metrics.split(",") if name.strip()}
    unknown = metrics - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
    metric_cpu_threads = int(getattr(args, "metric_cpu_threads", 8))
    if metric_cpu_threads <= 0:
        raise ValueError("metric_cpu_threads must be positive")
    torch.set_num_threads(metric_cpu_threads)
    _validate_test_gate(args)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    device = torch.device(args.device)
    system = _make_system(args, device)
    metadata = system.metadata()
    model_complexity = _model_complexity(system)
    config_hash = canonical_hash(metadata)
    checkpoint_hash = _checkpoint_hash(metadata)
    decode_policy = _decode_policy(system)
    result_seed = _result_seed(system, args.seed)
    manifest_hash = sha256_file(args.manifest)
    metrics_metadata = metric_metadata()
    if "wer" in metrics:
        metrics_metadata["asr"]["asset_manifest"] = _asset_manifest(
            args.asr_model_metadata,
            revision=metrics_metadata["asr"]["revision"],
        )
    if "speaker_cosine" in metrics:
        metrics_metadata["speaker"]["asset_manifest"] = _asset_manifest(
            args.speaker_model_metadata,
            revision=metrics_metadata["speaker"]["revision"],
        )
    if "visqol_speech" in metrics:
        metrics_metadata["visqol"]["asset_manifest"] = _asset_manifest(
            args.visqol_metadata,
            revision=metrics_metadata["visqol"]["revision"],
        )
    metric_metadata_hash = canonical_hash(metrics_metadata)
    metrics_source_hash = sha256_file(Path(__file__).with_name("metrics.py"))
    test_gate_sha256 = _validate_test_gate(
        args,
        metrics=metrics,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        metric_metadata_hash=metric_metadata_hash,
        system_name=system.system_name,
        system_seed=result_seed,
        decode_seed_or_policy=decode_policy,
    )
    run_id = args.run_id or f"{args.split}-{args.system}-{config_hash[:12]}"
    run_dir = args.output_dir / run_id
    for directory in (
        run_dir,
        run_dir / "checkpoints",
        run_dir / "logs",
        run_dir / "audio_samples",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    cuts = list(CutSet.from_file(args.manifest))
    if args.max_cuts is not None:
        cuts = cuts[: args.max_cuts]
    run_config = {
        "schema_version": 1,
        "run_id": run_id,
        "system": args.system,
        "split": args.split,
        "seed": result_seed,
        "analysis_seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "max_duration": args.max_duration,
        "max_cuts": args.max_cuts,
        "warmup_cuts": args.warmup_cuts,
        "remeasure_rtf": args.remeasure_rtf,
        "metrics": sorted(metrics),
        "metric_cpu_threads": metric_cpu_threads,
        "device": str(device),
        "system_config_hash": config_hash,
        "decode_seed_or_policy": decode_policy,
        "test_gate_sha256": test_gate_sha256,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(run_config, sort_keys=True), encoding="utf-8"
    )
    _atomic_json(run_dir / "environment.json", _environment())
    _atomic_json(
        run_dir / "git_state.json", _git_state(Path(__file__).resolve().parents[2])
    )
    _atomic_json(
        run_dir / "data_manifest.json",
        {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_hash,
            "cuts": len(cuts),
            "split": args.split,
        },
    )
    _atomic_json(
        run_dir / "model_metadata.json",
        {
            "system": metadata,
            "model_complexity": model_complexity,
            "metrics": metrics_metadata,
        },
    )

    if args.warmup_cuts:
        warmup_items = _load_batch(cuts[: args.warmup_cuts])
        warmup_results = _reconstruct_with_fallback(system, warmup_items, device)
        failures_during_warmup = [
            result for result in warmup_results if isinstance(result, Exception)
        ]
        if failures_during_warmup:
            raise RuntimeError(
                f"Reconstruction warm-up failed: {failures_during_warmup[0]}"
            )

    reconstruction_cache = ReconstructionCache(args.cache_root / "reconstructions")
    asr_cache = JsonCache(args.cache_root / "asr")
    metric_cache = JsonCache(args.cache_root / "metrics")
    speaker_metric = (
        SpeakerSimilarityMetric(args.model_cache_root, device)
        if "speaker_cosine" in metrics
        else None
    )
    visqol_metric = VisqolSpeechMetric() if "visqol_speech" in metrics else None
    asr = WhisperASR(args.model_cache_root, device) if "wer" in metrics else None
    code_statistics = CodeStatisticsAccumulator(system.codebook_cardinalities)
    code_dependency = (
        CodeDependencyAccumulator(
            system.codebook_cardinalities,
            sample_limit=args.dependency_sample_limit,
            seed=args.seed,
        )
        if args.system in {"ours", "dac"} and system.num_parallel_streams >= 2
        else None
    )
    records = []
    failures = []
    samples_written = 0

    for cut_batch in _iter_batches(cuts, args.max_duration):
        items = _load_batch(cut_batch)
        payloads: list[dict | Exception | None] = [None] * len(items)
        cache_keys = []
        missing_items = []
        missing_indices = []
        for index, item in enumerate(items):
            key = reconstruction_cache_key(
                system_config_hash=config_hash,
                checkpoint_hash=checkpoint_hash,
                cut_id=item.cut_id,
                decode_seed_or_policy=decode_policy,
            )
            cache_keys.append(key)
            cached = reconstruction_cache.load(key)
            if cached is None or args.remeasure_rtf:
                missing_items.append(item)
                missing_indices.append(index)
            else:
                if cached["audio_length"] != item.audio_length:
                    raise ValueError(f"Cached length mismatch for {item.cut_id}")
                payloads[index] = cached
        if missing_items:
            reconstructed = _reconstruct_with_fallback(system, missing_items, device)
            for index, result in zip(missing_indices, reconstructed):
                payloads[index] = result
                if not isinstance(result, Exception):
                    reconstruction_cache.save(cache_keys[index], result)

        reference_asr: list[dict | Exception | None] = [None] * len(items)
        reconstructed_asr: list[dict | Exception | None] = [None] * len(items)
        if asr is not None:
            for index, item in enumerate(items):
                reference_key = canonical_hash(
                    {
                        "metric": metrics_metadata["asr"],
                        "manifest_sha256": manifest_hash,
                        "cut_id": item.cut_id,
                        "kind": "reference",
                    }
                )
                cached = asr_cache.load(reference_key)
                if cached is not None:
                    reference_asr[index] = cached["result"]
                payload = payloads[index]
                if not isinstance(payload, (Exception, type(None))):
                    reconstruction_key = canonical_hash(
                        {
                            "metric": metrics_metadata["asr"],
                            "reconstruction_cache_key": cache_keys[index],
                            "manifest_sha256": manifest_hash,
                            "kind": "reconstruction",
                        }
                    )
                    cached = asr_cache.load(reconstruction_key)
                    if cached is not None:
                        reconstructed_asr[index] = cached["result"]

            reference_indices = [
                index for index, value in enumerate(reference_asr) if value is None
            ]
            outputs = _transcribe_with_fallback(
                asr, [items[index].waveform for index in reference_indices], 24_000
            )
            for index, output in zip(reference_indices, outputs):
                if isinstance(output, Exception):
                    reference_asr[index] = output
                else:
                    try:
                        result = word_error_counts(
                            items[index].text,
                            output,
                            normalizer=asr.normalize_text,
                            include_raw=True,
                        )
                    except Exception as error:
                        reference_asr[index] = error
                        continue
                    reference_asr[index] = result
                    key = canonical_hash(
                        {
                            "metric": metrics_metadata["asr"],
                            "manifest_sha256": manifest_hash,
                            "cut_id": items[index].cut_id,
                            "kind": "reference",
                        }
                    )
                    asr_cache.save(key, {"result": result})
            reconstruction_indices = [
                index
                for index, value in enumerate(reconstructed_asr)
                if value is None and isinstance(payloads[index], dict)
            ]
            outputs = _transcribe_with_fallback(
                asr,
                [payloads[index]["waveform"] for index in reconstruction_indices],
                24_000,
            )
            for index, output in zip(reconstruction_indices, outputs):
                if isinstance(output, Exception):
                    reconstructed_asr[index] = output
                else:
                    try:
                        result = word_error_counts(
                            items[index].text,
                            output,
                            normalizer=asr.normalize_text,
                            include_raw=True,
                        )
                    except Exception as error:
                        reconstructed_asr[index] = error
                        continue
                    reconstructed_asr[index] = result
                    key = canonical_hash(
                        {
                            "metric": metrics_metadata["asr"],
                            "reconstruction_cache_key": cache_keys[index],
                            "manifest_sha256": manifest_hash,
                            "kind": "reconstruction",
                        }
                    )
                    asr_cache.save(key, {"result": result})

        for index, (item, payload) in enumerate(zip(items, payloads)):
            metric_values: dict[str, float | None] = {}
            item_failures = []
            if isinstance(payload, Exception) or payload is None:
                exc = (
                    payload
                    if isinstance(payload, Exception)
                    else RuntimeError("missing payload")
                )
                item_failures.append(_error("reconstruction", exc))
                reconstructed_waveform = None
                code_histograms = [
                    [0] * cardinality for cardinality in system.codebook_cardinalities
                ]
                code_length = 0
                encoder_seconds = 0.0
                decoder_seconds = 0.0
                peak_vram_bytes = 0
            else:
                reconstructed_waveform = payload["waveform"]
                code_length = int(payload["code_length"])
                encoder_seconds = float(payload["encoder_seconds"])
                decoder_seconds = float(payload["decoder_seconds"])
                peak_vram_bytes = int(payload.get("peak_vram_bytes", 0))
                code_histograms = _code_histograms(
                    payload["codes"], system.codebook_cardinalities
                )
                code_statistics.update_histograms(code_histograms)
                if code_dependency is not None:
                    code_dependency.update(payload["codes"], item.cut_id)

            metric_functions = {
                "stoi": stoi_score,
                "estoi": estoi_score,
                "pesq_wb": pesq_wb_score,
                "si_sdr": lambda reference, degraded, _: si_sdr_db(reference, degraded),
                "mr_stft": lambda reference, degraded, _: (
                    multi_resolution_stft_distance(reference, degraded)
                ),
                "ms_log_mel": multi_scale_log_mel_distance,
                "f0_correlation": f0_correlation,
            }
            for name, function in metric_functions.items():
                if name not in metrics:
                    continue
                if reconstructed_waveform is None:
                    metric_values[name] = None
                    continue
                try:
                    key = _metric_cache_key(
                        metric=name,
                        metric_metadata_hash=metric_metadata_hash,
                        metrics_source_hash=metrics_source_hash,
                        manifest_hash=manifest_hash,
                        cut_id=item.cut_id,
                        reconstruction_key=cache_keys[index],
                    )
                    metric_values[name] = _cached_metric_value(
                        metric_cache,
                        key,
                        lambda: function(item.waveform, reconstructed_waveform, 24_000),
                    )
                except Exception as exc:
                    metric_values[name] = None
                    item_failures.append(_error(name, exc))
            if "visqol_speech" in metrics:
                try:
                    if reconstructed_waveform is None:
                        raise RuntimeError("reconstruction unavailable")
                    key = _metric_cache_key(
                        metric="visqol_speech",
                        metric_metadata_hash=metric_metadata_hash,
                        metrics_source_hash=metrics_source_hash,
                        manifest_hash=manifest_hash,
                        cut_id=item.cut_id,
                        reconstruction_key=cache_keys[index],
                    )
                    metric_values["visqol_speech"] = _cached_metric_value(
                        metric_cache,
                        key,
                        lambda: visqol_metric(
                            item.waveform, reconstructed_waveform, 24_000
                        ),
                    )
                except Exception as exc:
                    metric_values["visqol_speech"] = None
                    item_failures.append(_error("visqol_speech", exc))
            if "speaker_cosine" in metrics:
                try:
                    if reconstructed_waveform is None:
                        raise RuntimeError("reconstruction unavailable")
                    key = _metric_cache_key(
                        metric="speaker_cosine",
                        metric_metadata_hash=metric_metadata_hash,
                        metrics_source_hash=metrics_source_hash,
                        manifest_hash=manifest_hash,
                        cut_id=item.cut_id,
                        reconstruction_key=cache_keys[index],
                    )
                    metric_values["speaker_cosine"] = _cached_metric_value(
                        metric_cache,
                        key,
                        lambda: speaker_metric(
                            item.waveform, reconstructed_waveform, 24_000
                        ),
                    )
                except Exception as exc:
                    metric_values["speaker_cosine"] = None
                    item_failures.append(_error("speaker_cosine", exc))

            asr_result = {"reference": None, "reconstruction": None}
            if asr is not None:
                for kind, value in (
                    ("reference", reference_asr[index]),
                    ("reconstruction", reconstructed_asr[index]),
                ):
                    if isinstance(value, Exception):
                        item_failures.append(_error(f"wer_{kind}", value))
                    elif value is not None:
                        asr_result[kind] = value

            record = {
                "schema_version": 1,
                "run_id": run_id,
                "system": system.system_name,
                "variant": args.system,
                "seed": result_seed,
                "split": args.split,
                "cut_id": item.cut_id,
                "speaker_id": item.speaker_id,
                "duration": item.duration,
                "nominal_bitrate_kbps": system.nominal_bitrate,
                "entropy_bitrate_kbps": None,
                "serialized_bitrate_kbps": None,
                "token_frame_rate": system.token_frame_rate,
                "num_parallel_streams": system.num_parallel_streams,
                "codebook_cardinalities": list(system.codebook_cardinalities),
                "code_histograms": code_histograms,
                "token_frames": code_length,
                "audio_length": item.audio_length,
                "metrics": metric_values,
                "asr": asr_result,
                "failed": any(
                    failure["metric"] == "reconstruction" for failure in item_failures
                ),
                "metric_failures": item_failures,
                "checkpoint_hash": checkpoint_hash,
                "config_hash": config_hash,
                "reconstruction_cache_key": cache_keys[index],
                "encoder_seconds": encoder_seconds,
                "decoder_seconds": decoder_seconds,
                "peak_vram_bytes": peak_vram_bytes,
                "model_complexity": model_complexity,
            }
            records.append(record)
            for failure in item_failures:
                failures.append({"run_id": run_id, "cut_id": item.cut_id, **failure})
            if (
                reconstructed_waveform is not None
                and samples_written < args.audio_samples
            ):
                stem = item.cut_id.replace("/", "_")
                torchaudio.save(
                    run_dir / "audio_samples" / f"{stem}_reference.wav",
                    item.waveform,
                    24_000,
                )
                torchaudio.save(
                    run_dir / "audio_samples" / f"{stem}_reconstruction.wav",
                    reconstructed_waveform,
                    24_000,
                )
                samples_written += 1

    entropy_bitrate = code_statistics.empirical_bitrate_kbps(system.token_frame_rate)
    for record in records:
        record["entropy_bitrate_kbps"] = entropy_bitrate
    records_path = run_dir / "per_utterance.jsonl"
    _atomic_jsonl(records_path, records)
    _atomic_jsonl(run_dir / "failures.jsonl", failures)
    if code_dependency is not None:
        _atomic_json(run_dir / "code_dependency.json", code_dependency.as_dict())
    summary = write_summary(records_path, run_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traceable codec reconstruction evaluation"
    )
    parser.add_argument(
        "--system",
        choices=("identity", "mel_oracle", "dmel", "dmel_decoder", "dac", "ours"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-cache-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-cuts", type=int)
    parser.add_argument("--warmup-cuts", type=int, default=2)
    parser.add_argument("--remeasure-rtf", action="store_true")
    parser.add_argument("--audio-samples", type=int, default=4)
    parser.add_argument("--dependency-sample-limit", type=int, default=50_000)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument(
        "--metric-cpu-threads",
        type=int,
        default=8,
        help="PyTorch intra-op threads used by CPU waveform metrics",
    )
    parser.add_argument("--asr-model-metadata", type=Path)
    parser.add_argument("--speaker-model-metadata", type=Path)
    parser.add_argument("--visqol-metadata", type=Path)
    parser.add_argument("--test-gate", type=Path)
    parser.add_argument("--bigvgan-checkpoint", type=Path)
    parser.add_argument("--bigvgan-config", type=Path)
    parser.add_argument("--dmel-range", type=Path)
    parser.add_argument("--dmel-decoder-checkpoint", type=Path)
    parser.add_argument("--dac-checkpoint", type=Path)
    parser.add_argument("--n-quantizers", type=int, default=2)
    parser.add_argument("--ours-checkpoint", type=Path)
    args = parser.parse_args()
    if (
        args.max_duration <= 0
        or args.audio_samples < 0
        or args.warmup_cuts < 0
        or args.dependency_sample_limit < 2
        or args.metric_cpu_threads <= 0
    ):
        parser.error(
            "--max-duration must be positive; --audio-samples/--warmup-cuts "
            "non-negative; --dependency-sample-limit >= 2; "
            "--metric-cpu-threads positive"
        )
    if args.system == "dmel_decoder" and args.dmel_decoder_checkpoint is None:
        parser.error("--system dmel_decoder requires --dmel-decoder-checkpoint")
    return args


def main() -> int:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
