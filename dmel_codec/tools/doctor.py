from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from dmel_codec.train_codec import CONFIG_DIR, get_config


REQUIRED_SPLITS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)
CORE_IMPORTS = (
    "einops",
    "hydra",
    "librosa",
    "lightning",
    "lhotse",
    "numpy",
    "omegaconf",
    "scipy",
    "soundfile",
    "torch",
    "torchaudio",
    "vector_quantize_pytorch",
)
FORBIDDEN_CONFIG_TEXT = ("???", "/home/", "/data0/", "/sdb/")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _check_dependencies() -> Check:
    missing = [name for name in CORE_IMPORTS if importlib.util.find_spec(name) is None]
    return Check(
        "dependencies",
        not missing,
        "all core imports available" if not missing else f"missing: {', '.join(missing)}",
    )


def _check_config_sources() -> Check:
    failures = []
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        if "config/lm" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        found = [token for token in FORBIDDEN_CONFIG_TEXT if token in text]
        if found:
            failures.append(f"{path.relative_to(CONFIG_DIR)}: {found}")
    return Check(
        "config_sources",
        not failures,
        "no missing markers or private paths" if not failures else "; ".join(failures),
    )


def _check_config_composition() -> Check:
    failures = []
    combinations = [
        (codec, experiment)
        for codec in ("ours_075", "ours_l", "ours_m", "ours_h")
        for experiment in ("smoke", "pilot_clean100", "full_libritts")
    ] + [
        ("ours_m", "pilot_clean100_repair2_100k"),
        ("ours_m", "ablation_random_groups"),
        ("ours_m", "ablation_joint_fullband"),
        ("ours_m", "refinement_mel_control"),
        ("ours_m", "refinement_waveform_frozen"),
        ("ours_m", "refinement_waveform_joint_tail"),
        ("ours_m", "refinement_waveform_joint_gan"),
        ("ours_m", "refinement_waveform_codec_gan"),
        ("ours_m", "refinement_waveform_vocoder_gan"),
        ("ours_m", "refinement_waveform_full_codec_gan"),
        ("ours_m", "refinement_waveform_codec_gan_100k"),
        ("ours_m", "refinement_waveform_vocoder_full_gan_100k"),
        ("ours_m", "refinement_waveform_full_codec_full_vocoder_gan_100k"),
    ]
    for codec, experiment in combinations:
        label = f"codec={codec},experiment={experiment}"
        try:
            config = get_config([f"codec={codec}", f"experiment={experiment}"])
            OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        except Exception as exc:  # pragma: no cover - detail is surfaced by CLI
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    return Check(
        "config_composition",
        not failures,
        f"{len(combinations)} codec/experiment combinations resolved"
        if not failures
        else " | ".join(failures),
    )


def _check_smoke_instantiation() -> Check:
    try:
        config = get_config(["codec=ours_m", "experiment=smoke"])
        model = hydra.utils.instantiate(config.codec.model, _convert_="partial")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        del model
        return Check("cpu_smoke", True, f"model instantiated ({parameter_count:,} parameters)")
    except Exception as exc:
        return Check("cpu_smoke", False, f"{type(exc).__name__}: {exc}")


def _check_data_root(data_root: Path) -> Check:
    missing = [split for split in REQUIRED_SPLITS if not (data_root / split).is_dir()]
    readable = data_root.is_dir() and os.access(data_root, os.R_OK)
    writable = data_root.exists() and os.access(data_root, os.W_OK)
    ok = readable and not writable and not missing
    detail = f"readable={readable} writable={writable}"
    if missing:
        detail += f" missing={','.join(missing)}"
    return Check("libritts", ok, detail)


def _check_writable_path(path: Path | None, name: str) -> Check:
    if path is None:
        return Check(name, False, f"not configured; set {name.upper()}")
    resolved = path.expanduser().resolve()
    ok = resolved.is_dir() and os.access(resolved, os.W_OK)
    return Check(name, ok, f"{resolved} writable={ok}")


def _check_bigvgan(checkpoint: Path | None, config_path: Path | None) -> Check:
    if checkpoint is None or config_path is None:
        return Check(
            "bigvgan",
            False,
            "set BIGVGAN_CKPT and BIGVGAN_CONFIG or pass both CLI options",
        )
    checkpoint = checkpoint.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if not checkpoint.is_file() or not config_path.is_file():
        return Check(
            "bigvgan",
            False,
            f"checkpoint_exists={checkpoint.is_file()} config_exists={config_path.is_file()}",
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Check("bigvgan", False, f"invalid config: {exc}")
    expected = {
        "sampling_rate": 24000,
        "num_mels": 100,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 1024,
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    return Check(
        "bigvgan",
        not mismatch,
        f"checkpoint_bytes={checkpoint.stat().st_size} frontend={expected}"
        if not mismatch
        else f"frontend mismatch: {mismatch}",
    )


def _path_arg(value: str | None) -> Path | None:
    return Path(value) if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the dMel codec runtime")
    parser.add_argument(
        "--data-root", default=os.environ.get("LIBRITTS_ROOT", "/datasets/LibriTTS")
    )
    parser.add_argument("--run-root", default=os.environ.get("DMEL_RUN_ROOT"))
    parser.add_argument("--cache-root", default=os.environ.get("DMEL_CACHE_ROOT"))
    parser.add_argument("--bigvgan-ckpt", default=os.environ.get("BIGVGAN_CKPT"))
    parser.add_argument("--bigvgan-config", default=os.environ.get("BIGVGAN_CONFIG"))
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def run_checks(args: argparse.Namespace) -> list[Check]:
    return [
        Check("python", sys.version_info[:2] == (3, 10), platform.python_version()),
        _check_dependencies(),
        _check_config_sources(),
        _check_config_composition(),
        _check_smoke_instantiation(),
        _check_data_root(Path(args.data_root).resolve()),
        _check_writable_path(_path_arg(args.run_root), "dmel_run_root"),
        _check_writable_path(_path_arg(args.cache_root), "dmel_cache_root"),
        _check_bigvgan(
            _path_arg(args.bigvgan_ckpt), _path_arg(args.bigvgan_config)
        ),
        Check(
            "gpu",
            torch.cuda.is_available(),
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()} "
            f"torch={torch.__version__} cuda={torch.version.cuda}",
        ),
    ]


def main() -> int:
    args = parse_args()
    checks = run_checks(args)
    if args.json_output:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            status = "PASS" if check.ok else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
