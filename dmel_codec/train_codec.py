from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import hydra
import lightning.pytorch as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from dmel_codec.utils.logger import RankedLogger
from dmel_codec.utils.print_config import print_config_tree
from dmel_codec.utils.callbacks import StopAfterValidationCheckpoint


logger = RankedLogger(__name__, rank_zero_only=True)
CONFIG_DIR = Path(__file__).resolve().parent / "config"
STOP_AFTER_VALIDATION_ENV = "DMEL_STOP_AFTER_VALIDATION_STEP"


def get_config(overrides: Iterable[str] | None = None) -> DictConfig:
    """Compose the codec config without relying on a private user config."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(CONFIG_DIR.resolve())
    ):
        return compose(config_name="config", overrides=list(overrides or []))


def _required_file(value: str | None, label: str) -> Path:
    if not value or value == "null":
        raise FileNotFoundError(
            f"{label} is not configured. Set the corresponding environment variable."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _require_external_run_dir(value: str) -> Path:
    run_dir = Path(value).expanduser()
    if not run_dir.is_absolute():
        raise ValueError(
            "Formal runs require an absolute DMEL_RUN_ROOT/run_dir outside git"
        )
    repository = CONFIG_DIR.parent.parent.resolve()
    resolved = run_dir.resolve()
    if resolved == repository or repository in resolved.parents:
        raise ValueError(f"Formal run_dir must be outside the git checkout: {resolved}")
    return run_dir


def validate_runtime_config(config: DictConfig) -> None:
    """Fail before model allocation when a run cannot satisfy its contract."""
    OmegaConf.resolve(config)
    refinement_mode = str(config.codec.model.get("refinement_mode", ""))
    if refinement_mode == "waveform_joint_vocoder_tail":
        raise ValueError(
            "Joint decoder/vocoder training requires a waveform GAN; use "
            "experiment=refinement_waveform_joint_gan. Historical non-GAN "
            "checkpoints remain loadable for audit and evaluation."
        )
    init_from = config.get("init_from")
    resume_from = config.get("resume_from")
    if init_from and resume_from:
        raise ValueError("init_from and resume_from are mutually exclusive")

    if init_from:
        _required_file(str(init_from), "init_from")
    if resume_from:
        _required_file(str(resume_from), "resume_from")

    if config.experiment.get("require_vocoder", False):
        vocoder = config.codec.model.get("vocoder")
        if not vocoder:
            raise ValueError("This experiment requires a BigVGAN vocoder config")
        _required_file(vocoder.get("ckpt_path"), "BIGVGAN_CKPT")
        _required_file(vocoder.get("h_path"), "BIGVGAN_CONFIG")
        _require_external_run_dir(str(config.experiment.run_dir))
        for key in ("train_cuts_path", "val_cuts_path"):
            _required_file(str(config.data.module.get(key)), f"data.module.{key}")

    accelerator = str(config.experiment.trainer.get("accelerator", "auto"))
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "The selected experiment requires CUDA, but torch.cuda.is_available() is false"
        )
    stop_step = os.environ.get(STOP_AFTER_VALIDATION_ENV)
    if stop_step is not None:
        try:
            parsed_stop_step = int(stop_step)
        except ValueError as error:
            raise ValueError(
                f"{STOP_AFTER_VALIDATION_ENV} must be a positive integer"
            ) from error
        if parsed_stop_step <= 0:
            raise ValueError(
                f"{STOP_AFTER_VALIDATION_ENV} must be a positive integer"
            )


def load_initial_weights(
    model: torch.nn.Module, checkpoint_path: str | Path, *, strict: bool = False
):
    """Load model weights only; optimizer, scheduler, and global step are ignored."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("init_from checkpoint does not contain a state dict")
    if hasattr(model, "on_load_checkpoint"):
        model.on_load_checkpoint(checkpoint)
    return model.load_state_dict(state_dict, strict=strict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_state() -> dict[str, Any]:
    repository = CONFIG_DIR.parent.parent

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = git("status", "--short").splitlines()
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def _data_manifest_metadata(config: DictConfig) -> dict[str, Any]:
    paths = {}
    for key in ("train_cuts_path", "val_cuts_path", "test_cuts_path"):
        value = config.data.module.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if path.is_file():
                metadata = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                name = path.name
                for suffix in (".jsonl.gz", ".jsonl"):
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                derivation = path.with_name(f"{name}.metadata.json")
                if derivation.is_file():
                    metadata["derivation"] = {
                        "path": str(derivation),
                        "bytes": derivation.stat().st_size,
                        "sha256": _sha256_file(derivation),
                    }
                paths[key] = metadata
    summary_path = Path(str(config.data.manifest_dir)) / "manifest_summary.json"
    summary = (
        {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": _sha256_file(summary_path),
        }
        if summary_path.is_file()
        else None
    )
    return {"manifests": paths, "manifest_summary": summary}


def _environment_metadata() -> dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": gpu_names,
    }
    stop_step = os.environ.get(STOP_AFTER_VALIDATION_ENV)
    if stop_step is not None:
        metadata["runtime_stop_after_validation_step"] = int(stop_step)
    return metadata


def _scientific_configuration(config: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved codec configuration must be a mapping")
    # The recovery location is execution provenance, not a scientific
    # hyperparameter. Normalizing it keeps a resumed run's identity stable.
    resolved["resume_from"] = None
    return resolved


def _validated_scientific_configuration(config: DictConfig) -> dict[str, Any]:
    """Return the original scientific identity for fresh or resumed runs."""
    current_resolved = _scientific_configuration(config)
    if not config.get("resume_from"):
        return current_resolved
    run_dir = Path(str(config.experiment.run_dir)).expanduser()
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Resume run lacks resolved_config.yaml: {resolved_path}"
    )
    previous = OmegaConf.load(resolved_path)
    previous_resolved = _scientific_configuration(previous)
    # init_from is applied exactly once. A resumed invocation must omit it so
    # Lightning can restore the complete Trainer state, while retaining the
    # original initialization checkpoint in the run's scientific identity.
    if current_resolved.get("init_from") is None:
        current_resolved["init_from"] = previous_resolved.get("init_from")
    current_experiment = current_resolved.get("experiment", {})
    previous_experiment = previous_resolved.get("experiment", {})
    if (
        isinstance(current_experiment, dict)
        and isinstance(previous_experiment, dict)
        and "init_from" in previous_experiment
        and current_experiment.get("init_from") is None
    ):
        current_experiment["init_from"] = previous_experiment.get("init_from")
    if previous_resolved != current_resolved:
        raise ValueError("Resume configuration differs from the original run")
    return previous_resolved


def _validate_resume_configuration(config: DictConfig) -> None:
    _validated_scientific_configuration(config)


def prepare_run_metadata(config: DictConfig, model: torch.nn.Module) -> dict[str, Any]:
    """Create traceable run files and the metadata embedded in every checkpoint."""
    resolved = _validated_scientific_configuration(config)
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    git_state = _git_state()
    data_manifest = _data_manifest_metadata(config)
    model_metadata = {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "codec_definition": (
            model.codec_definition() if hasattr(model, "codec_definition") else None
        ),
    }
    init_from = resolved.get("init_from")
    if init_from:
        init_path = Path(str(init_from)).expanduser().resolve()
        model_metadata["initialization"] = {
            "method": "weights_only",
            "path": str(init_path),
            "bytes": init_path.stat().st_size,
            "sha256": _sha256_file(init_path),
        }
    checkpoint_metadata = {
        "config_sha256": config_hash,
        "resolved_config": resolved,
        "git": git_state,
        "data": data_manifest,
        "model": model_metadata,
    }
    if hasattr(model, "set_checkpoint_metadata"):
        model.set_checkpoint_metadata(checkpoint_metadata)

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        run_dir = Path(str(config.experiment.run_dir)).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
        resume_from = config.get("resume_from")
        if resume_from:
            resume_path = Path(str(resume_from)).expanduser().resolve()
            history_path = run_dir / "resume_history.jsonl"
            history = (
                history_path.read_text(encoding="utf-8")
                if history_path.is_file()
                else ""
            )
            event = {
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(resume_path),
                "checkpoint_bytes": resume_path.stat().st_size,
                "checkpoint_sha256": _sha256_file(resume_path),
                "git": git_state,
                "environment": _environment_metadata(),
            }
            _atomic_write(
                history_path,
                history + json.dumps(event, sort_keys=True) + "\n",
            )
        else:
            _atomic_write(
                run_dir / "resolved_config.yaml",
                OmegaConf.to_yaml(
                    OmegaConf.create(resolved), resolve=True, sort_keys=True
                ),
            )
            _write_json(run_dir / "git_state.json", git_state)
            _write_json(run_dir / "data_manifest.json", data_manifest)
            _write_json(run_dir / "model_metadata.json", model_metadata)
            _write_json(run_dir / "environment.json", _environment_metadata())
    return checkpoint_metadata


def _instantiate_callbacks(config: DictConfig) -> list:
    callbacks = []
    for callback_config in config.experiment.get("callbacks", {}).values():
        if callback_config and callback_config.get("_target_"):
            callbacks.append(hydra.utils.instantiate(callback_config))
    stop_step = os.environ.get(STOP_AFTER_VALIDATION_ENV)
    if stop_step is not None:
        callbacks.append(StopAfterValidationCheckpoint(int(stop_step)))
    return callbacks


def main(config: DictConfig) -> None:
    validate_runtime_config(config)
    _validate_resume_configuration(config)
    print_config_tree(
        config,
        print_order=("data", "codec", "experiment", "project", "seed"),
        resolve=True,
    )
    pl.seed_everything(config.seed, workers=True)

    logger.info(f"Instantiating datamodule <{config.data.module._target_}>")
    datamodule = hydra.utils.instantiate(config.data.module, _convert_="partial")

    logger.info(f"Instantiating model <{config.codec.model._target_}>")
    model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
    prepare_run_metadata(config, model)
    if config.get("init_from"):
        result = load_initial_weights(model, config.init_from)
        if hasattr(model, "validate_initial_weights"):
            model.validate_initial_weights(result)
        logger.info(
            f"Loaded init_from weights: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )

    callbacks = _instantiate_callbacks(config)
    logger_config = config.experiment.get("logger", False)
    experiment_logger = (
        hydra.utils.instantiate(logger_config, _convert_="partial")
        if logger_config and logger_config.get("_target_")
        else False
    )
    trainer = hydra.utils.instantiate(
        config.experiment.trainer,
        callbacks=callbacks,
        logger=experiment_logger,
        use_distributed_sampler=False,
        _convert_="partial",
    )

    resume_from = config.get("resume_from") or None
    logger.info(f"Starting training with resume_from={resume_from}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def cli(config: DictConfig) -> None:
    main(config)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    cli()
