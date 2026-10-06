from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_ours_evaluation_pins_cublas_before_python_starts(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'child_workspace=%s\\n' \"${CUBLAS_WORKSPACE_CONFIG:-missing}\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    repository = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DMEL_RUN_ROOT": str(tmp_path / "runs"),
        "DMEL_CACHE_ROOT": str(tmp_path / "cache"),
        "OURS_CKPT": str(tmp_path / "ours.ckpt"),
        "BIGVGAN_CKPT": str(tmp_path / "bigvgan.pt"),
        "BIGVGAN_CONFIG": str(tmp_path / "config.json"),
        "CUDA_VISIBLE_DEVICES": "6",
    }
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    command = [
        "bash",
        "scripts/evaluate.sh",
        "ours",
        "dev-clean",
        "wrapper-test",
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "child_workspace=:4096:8" in completed.stdout

    environment["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    rejected = subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "requires CUBLAS_WORKSPACE_CONFIG=:4096:8" in rejected.stderr
    assert "child_workspace" not in rejected.stdout


def test_formal_ours_075_wrapper_verifies_release_before_training(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'torchrun cublas=%s %s\\n' "
        '"${CUBLAS_WORKSPACE_CONFIG:-missing}" "$*"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    release = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    release.write_text("{}\n", encoding="utf-8")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LIBRITTS_ROOT": "/datasets/LibriTTS",
        "DMEL_RUN_ROOT": str(tmp_path / "runs"),
        "OURS_075_RELEASE": str(release),
        "BIGVGAN_CKPT": str(tmp_path / "bigvgan.pt"),
        "BIGVGAN_CONFIG": str(tmp_path / "config.json"),
        "CUDA_VISIBLE_DEVICES": "6",
        "NPROC": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", "scripts/train_ours_075.sh"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "experiment=pilot_clean100_repair2_100k" in completed.stdout
    assert "codec=ours_075" in completed.stdout
    assert "seed=2026" in completed.stdout
    assert "torchrun cublas=missing" in completed.stdout


@pytest.mark.parametrize(
    "script",
    ["scripts/train_selected_mplus.sh", "scripts/train_selected_mplus_segment.sh"],
)
def test_selected_mplus_wrappers_reject_duplicate_seed2026(script: str) -> None:
    repository = Path(__file__).resolve().parents[1]
    command = ["bash", script, "2026"]
    if script.endswith("_segment.sh"):
        command.append("2000")
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "seed 2026 is the selected direction run" in completed.stderr


def test_selected_mplus_uses_failure_driven_resume_instead_of_2k_restarts() -> None:
    repository = Path(__file__).resolve().parents[1]
    bulk = (repository / "scripts/train_selected_mplus.sh").read_text(encoding="utf-8")
    segment = (repository / "scripts/train_selected_mplus_segment.sh").read_text(
        encoding="utf-8"
    )

    assert "target_step=100000" in bulk
    assert "seq 2000 2000 100000" not in bulk
    assert 'if [[ ! -f "${preflight}" ]]' in segment
    assert 'if [[ -f "${resume_checkpoint}" ]]' in segment
    assert "DMEL_STOP_AFTER_VALIDATION_STEP=${target_step}" in segment


def test_selected_mplus_runs_one_100k_target_and_resumes_after_failure(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'command_text="$*"\n'
        'printf "python %s\\n" "$command_text" >>"$CALLS"\n'
        'output=""\n'
        "while (($#)); do\n"
        '  case "$1" in\n'
        "    --output) output=$2; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'if [[ "$command_text" == *"mplus_replication preflight"* ]]; then\n'
        '  mkdir -p "$(dirname "$output")"\n'
        "  printf '%s\\n' "
        '\'{"experiment":"refinement_waveform_codec_gan_100k",'
        '"source_checkpoint":{"path":"/tmp/source.ckpt"}}\' '
        '>"$output"\n'
        'elif [[ "$command_text" == *"mplus_replication verify-segment"* ]]; then\n'
        '  mkdir -p "$(dirname "$output")"\n'
        "  printf '%s\\n' "
        '\'{"status":"MPLUS_REPLICATION_SEGMENT_VERIFIED",'
        '"seed":2027,"target_step":100000,"direction":"codec"}\' '
        '>"$output"\n'
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        'printf "torchrun target=%s init=%s cublas=%s args=%s\\n" '
        '"${DMEL_STOP_AFTER_VALIDATION_STEP}" '
        '"${OURS_REFINEMENT_INIT:-missing}" '
        '"${CUBLAS_WORKSPACE_CONFIG:-missing}" "$*" >>"$CALLS"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)

    selection = tmp_path / "MPLUS_DIRECTION_SELECTED.json"
    selection.write_text(
        '{"selected_direction":"codec","git_commit":"abcdef1234567890"}\n',
        encoding="utf-8",
    )
    replication_root = tmp_path / "replications"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALLS": str(calls),
        "LIBRITTS_ROOT": "/datasets/LibriTTS",
        "DMEL_RUN_ROOT": str(tmp_path / "runs"),
        "DMEL_CACHE_ROOT": str(tmp_path / "cache"),
        "BIGVGAN_CKPT": str(tmp_path / "bigvgan.pt"),
        "BIGVGAN_CONFIG": str(tmp_path / "config.json"),
        "CUDA_VISIBLE_DEVICES": "6",
        "MPLUS_DIRECTION_SELECTION": str(selection),
        "MPLUS_REPLICATION_ROOT": str(replication_root),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    repository = Path(__file__).resolve().parents[1]
    command = ["bash", "scripts/train_selected_mplus.sh", "2027"]

    subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = replication_root / "seed_2027"
    receipt = output / "logs" / "segment_step100000.json"
    first_calls = calls.read_text(encoding="utf-8")
    assert first_calls.count("torchrun target=100000") == 1
    assert "init=/tmp/source.ckpt" in first_calls
    assert "cublas=missing" in first_calls
    assert "resume_from=" not in first_calls
    assert receipt.is_file()

    receipt.unlink()
    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    subprocess.run(
        command,
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    all_calls = calls.read_text(encoding="utf-8")
    assert all_calls.count("torchrun target=100000") == 2
    assert all_calls.count("cublas=missing") == 2
    assert f"resume_from={checkpoint}" in all_calls
    invocation_logs = list(output.glob("logs/segment_step100000_invocation_*.log"))
    assert len(invocation_logs) == 2


@pytest.mark.parametrize(
    "script",
    ["scripts/train_selected_mplus_segment.sh", "scripts/train_ours_075.sh"],
)
def test_formal_training_wrappers_record_resource_intervals(script: str) -> None:
    source = (Path(__file__).resolve().parents[1] / script).read_text(encoding="utf-8")
    assert "--record-job" in source
    assert "--start-time" in source
    assert "--end-time" in source
    assert "--cuda-visible-devices" in source
    assert "trap" in source


def test_readme_does_not_bypass_formal_ours_075_release() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "scripts/train.sh pilot_clean100_repair2_100k ours_075" not in readme
    assert "scripts/train_ours_075.sh" in readme


def test_evaluation_wrapper_records_fast_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *dmel_codec.tools.resource_accounting* ]]; then\n'
        f'  exec {sys.executable} "$@"\n'
        "fi\n"
        "exit 7\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    run_root = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DMEL_RUN_ROOT": str(run_root),
        "DMEL_CACHE_ROOT": str(tmp_path / "cache"),
        "CUDA_VISIBLE_DEVICES": "6",
    }
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["bash", "scripts/evaluate.sh", "identity", "dev-clean", "fast-failure"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    receipts = list(
        (run_root / "evaluations" / "fast-failure" / "logs").glob(
            "evaluation_invocation_*.resource.json"
        )
    )
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["category"] == "evaluation"
    assert receipt["outcome"] == "failed"
    assert receipt["exit_code"] == 7
    assert receipt["cuda_visible_devices"] == "6"
    assert receipt["elapsed_seconds"] > 0


def test_ours_075_wrapper_records_fast_training_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *dmel_codec.tools.resource_accounting* ]]; then\n'
        f'  exec {sys.executable} "$@"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)
    release = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    release.write_text("{}\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LIBRITTS_ROOT": "/datasets/LibriTTS",
        "DMEL_RUN_ROOT": str(run_root),
        "OURS_075_RELEASE": str(release),
        "BIGVGAN_CKPT": str(tmp_path / "bigvgan.pt"),
        "BIGVGAN_CONFIG": str(tmp_path / "config.json"),
        "CUDA_VISIBLE_DEVICES": "6",
        "NPROC": "1",
    }
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["bash", "scripts/train_ours_075.sh"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    receipts = list(
        (
            run_root / "pilot_clean100_repair2_100k" / "ours_075" / "seed_2026" / "logs"
        ).glob("training_invocation_*.resource.json")
    )
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["outcome"] == "failed"
    assert receipt["exit_code"] == 7
    assert receipt["cuda_visible_devices"] == "6"
    assert receipt["elapsed_seconds"] > 0


def test_ours_075_wrapper_does_not_hide_post_training_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *dmel_codec.tools.resource_accounting* ]]; then\n'
        f'  exec {sys.executable} "$@"\n'
        "fi\n"
        'if [[ "$*" == *dmel_codec.tools.select_checkpoint* ]]; then\n'
        "  exit 9\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_torchrun.chmod(0o755)
    release = tmp_path / "OURS_075_FORMAL_RELEASE.json"
    release.write_text("{}\n", encoding="utf-8")
    run_root = tmp_path / "runs"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LIBRITTS_ROOT": "/datasets/LibriTTS",
        "DMEL_RUN_ROOT": str(run_root),
        "OURS_075_RELEASE": str(release),
        "BIGVGAN_CKPT": str(tmp_path / "bigvgan.pt"),
        "BIGVGAN_CONFIG": str(tmp_path / "config.json"),
        "CUDA_VISIBLE_DEVICES": "6",
        "NPROC": "1",
    }
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["bash", "scripts/train_ours_075.sh"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 9
    receipts = list(
        (
            run_root / "pilot_clean100_repair2_100k" / "ours_075" / "seed_2026" / "logs"
        ).glob("training_invocation_*.resource.json")
    )
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["outcome"] == "completed"
    assert receipt["exit_code"] == 0
