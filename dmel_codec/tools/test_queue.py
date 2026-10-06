from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from dmel_codec.evaluation.cache import canonical_hash, sha256_file
from dmel_codec.evaluation.statistics import load_jsonl
from dmel_codec.evaluation.summarize import summarize_records


ASSET_FIELDS_BY_ADAPTER = {
    "identity": (),
    "mel_oracle": ("bigvgan_checkpoint", "bigvgan_config"),
    "dmel": ("bigvgan_checkpoint", "bigvgan_config", "dmel_range"),
    "dmel_decoder": (
        "bigvgan_checkpoint",
        "bigvgan_config",
        "dmel_range",
        "dmel_decoder_checkpoint",
    ),
    "dac": ("dac_checkpoint",),
    "ours": ("bigvgan_checkpoint", "bigvgan_config"),
}
METRIC_ASSET_FIELDS = {
    "wer": "asr_model_metadata",
    "speaker_cosine": "speaker_model_metadata",
    "visqol_speech": "visqol_metadata",
}
ENV_BY_ASSET = {
    "asr_model_metadata": "ASR_MODEL_METADATA",
    "bigvgan_checkpoint": "BIGVGAN_CKPT",
    "bigvgan_config": "BIGVGAN_CONFIG",
    "dac_checkpoint": "DAC_CKPT",
    "dmel_decoder_checkpoint": "DMEL_DECODER_CKPT",
    "dmel_range": "DMEL_RANGE",
    "speaker_model_metadata": "SPEAKER_MODEL_METADATA",
    "visqol_metadata": "VISQOL_METADATA",
}
REQUIRED_TASK_FILES = (
    "per_utterance.jsonl",
    "summary.json",
    "resolved_config.yaml",
    "model_metadata.json",
    "data_manifest.json",
    "git_state.json",
)
PRIMARY_BOOTSTRAP_METRICS = (
    "visqol_speech",
    "pesq_wb",
    "speaker_cosine",
    "wer_reconstruction",
    "wer_delta",
)


@dataclass(frozen=True)
class TaskState:
    evaluation_complete: bool
    statistics_complete: bool


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read YAML object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return value


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    content = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"Refusing to replace different artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _current_git(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=repository_root, text=True
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot inspect git repository {repository_root}"
        ) from error
    return {"commit": commit, "dirty": dirty}


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_external_root(path: Path, repository_root: Path, label: str) -> None:
    if _under(path, Path("/datasets/LibriTTS")):
        raise ValueError(f"{label} cannot be under /datasets/LibriTTS")
    if _under(path, repository_root):
        raise ValueError(f"{label} must be outside the git repository")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")


def _validate_gate(path: Path, repository_root: Path) -> tuple[dict[str, Any], str]:
    gate = _load_json(path)
    if (
        gate.get("schema_version") != 3
        or gate.get("status") != "TEST_GATE_FROZEN"
        or gate.get("allow_dirty_dev_runs") is not False
    ):
        raise ValueError("Final test queue requires schema-v3 TEST_GATE_FROZEN")
    systems = gate.get("systems")
    registry = gate.get("final_system_registry")
    metrics = gate.get("metrics")
    if (
        not isinstance(systems, list)
        or not systems
        or not isinstance(registry, dict)
        or registry.get("schema_version") != 1
        or registry.get("status") != "FINAL_SYSTEMS_PREREGISTERED"
        or not isinstance(metrics, list)
        or not metrics
        or not all(isinstance(metric, str) for metric in metrics)
        or len(metrics) != len(set(metrics))
        or set(gate.get("test_manifests", {})) != {"test-clean", "test-other"}
    ):
        raise ValueError("Frozen test gate lacks systems or registry")
    roles = [role for system in systems for role in system.get("roles", [])]
    if (
        len(roles) != len(set(roles))
        or sorted(roles) != registry.get("roles")
        or registry.get("unique_system_identities") != len(systems)
    ):
        raise ValueError("Frozen test gate has inconsistent role coverage")
    git_state = _current_git(repository_root)
    if git_state["dirty"] or git_state["commit"] != gate.get("git_commit"):
        raise ValueError("Final test queue requires the clean frozen gate commit")
    sources = gate.get("source_hashes")
    if not isinstance(sources, dict):
        raise ValueError("Frozen test gate lacks source hashes")
    required_sources = {
        "dmel_codec/tools/test_queue.py",
        "scripts/evaluate.sh",
        "scripts/run_test_queue.sh",
        "scripts/statistics.sh",
    }
    missing = sorted(required_sources - set(sources))
    if missing:
        raise ValueError(f"Frozen test gate does not bind queue sources: {missing}")
    for relative, expected in sources.items():
        source = repository_root / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
    from dmel_codec.evaluation.run_codec_eval import _validate_training_provenance

    _validate_training_provenance(gate)
    return gate, sha256_file(path)


def _asset_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Asset {label} must contain path and sha256")
    raw_path = value.get("path")
    expected = value.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"Asset {label} must contain path and sha256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Asset {label} SHA-256 mismatch")
    return {"path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}


def _resolve_assets(
    asset_manifest: Path, gate: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    value = _load_yaml(asset_manifest)
    if (
        value.get("schema_version") != 1
        or value.get("status") != "FINAL_TEST_ASSETS_RESOLVED"
    ):
        raise ValueError("Invalid final-test asset manifest")
    shared_raw = value.get("shared")
    ours_raw = value.get("ours_checkpoints")
    if not isinstance(shared_raw, dict) or not isinstance(ours_raw, dict):
        raise ValueError("Final-test assets require shared and ours_checkpoints maps")
    required_shared = {
        field
        for system in gate["systems"]
        for field in ASSET_FIELDS_BY_ADAPTER.get(str(system.get("adapter")), ())
        if field != "dmel_decoder_checkpoint"
    }
    required_shared.update(
        field
        for metric, field in METRIC_ASSET_FIELDS.items()
        if metric in gate["metrics"]
    )
    if any(system.get("adapter") == "dmel_decoder" for system in gate["systems"]):
        required_shared.add("dmel_decoder_checkpoint")
    if set(shared_raw) != required_shared:
        missing = sorted(required_shared - set(shared_raw))
        extra = sorted(set(shared_raw) - required_shared)
        raise ValueError(
            f"Shared test asset mismatch; missing={missing}, extra={extra}"
        )

    resolved_assets: dict[tuple[str, str], dict[str, Any]] = {}

    def resolve_asset(record: Any, *, label: str) -> dict[str, Any]:
        if not isinstance(record, dict):
            return _asset_record(record, label=label)
        path = str(Path(str(record.get("path", ""))).expanduser().resolve())
        expected = str(record.get("sha256", ""))
        key = (path, expected)
        if key not in resolved_assets:
            resolved_assets[key] = _asset_record(record, label=label)
        return resolved_assets[key]

    shared = {
        field: resolve_asset(shared_raw.get(field), label=f"shared.{field}")
        for field in sorted(required_shared)
    }

    ours: dict[str, dict[str, Any]] = {}
    ours_roles = {
        role
        for system in gate["systems"]
        if system.get("adapter") == "ours"
        for role in system["roles"]
    }
    if set(ours_raw) != ours_roles:
        missing = sorted(ours_roles - set(ours_raw))
        extra = sorted(set(ours_raw) - ours_roles)
        raise ValueError(
            f"Ours checkpoint role mismatch; missing={missing}, extra={extra}"
        )
    for role in sorted(ours_roles):
        ours[role] = resolve_asset(ours_raw[role], label=f"ours_checkpoints.{role}")
    for system in gate["systems"]:
        adapter = str(system.get("adapter"))
        expected = system.get("raw_checkpoint_sha256")
        if adapter == "ours":
            records = [ours[role] for role in system["roles"]]
            if len({record["path"] for record in records}) != 1:
                raise ValueError(
                    "Roles sharing one Ours identity must use one checkpoint"
                )
            observed = records[0]["sha256"]
        elif adapter == "dac":
            observed = shared["dac_checkpoint"]["sha256"]
        elif adapter == "dmel_decoder":
            observed = shared["dmel_decoder_checkpoint"]["sha256"]
        else:
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(
                f"Frozen {adapter} identity lacks a raw checkpoint SHA-256"
            )
        if observed != expected:
            raise ValueError(
                f"{adapter} checkpoint differs from frozen role {system['roles'][0]}"
            )
    return shared, ours


def _task_assets(
    system: dict[str, Any],
    shared: dict[str, dict[str, Any]],
    ours: dict[str, dict[str, Any]],
    metrics: Iterable[str],
) -> dict[str, dict[str, Any]]:
    adapter = str(system["adapter"])
    if adapter not in ASSET_FIELDS_BY_ADAPTER:
        raise ValueError(f"Unsupported frozen adapter: {adapter}")
    fields = set(ASSET_FIELDS_BY_ADAPTER[adapter])
    fields.update(
        field for metric, field in METRIC_ASSET_FIELDS.items() if metric in metrics
    )
    result = {field: shared[field] for field in sorted(fields)}
    if adapter == "ours":
        result["ours_checkpoint"] = ours[system["roles"][0]]
    return result


def _run_id(system: dict[str, Any], split: str, gate_sha256: str) -> str:
    role = sorted(system["roles"])[0].replace("_", "-")
    return f"final-{role}-{split}-{gate_sha256[:8]}"


def _manifest_identity(path: Path) -> dict[str, Any]:
    cut_ids = []
    try:
        with gzip.open(path, mode="rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                cut_id = value.get("id") if isinstance(value, dict) else None
                if not isinstance(cut_id, str) or not cut_id:
                    raise ValueError(f"Manifest row {line_number} lacks a cut id")
                cut_ids.append(cut_id)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot inspect cut IDs in {path}") from error
    if not cut_ids or len(cut_ids) != len(set(cut_ids)):
        raise ValueError(f"Manifest must contain unique non-empty cut IDs: {path}")
    return {"cuts": len(cut_ids), "cut_ids_sha256": canonical_hash(sorted(cut_ids))}


def _build_tasks(
    *,
    gate: dict[str, Any],
    gate_sha256: str,
    run_root: Path,
    shared: dict[str, dict[str, Any]],
    ours: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = sorted(str(metric) for metric in gate["metrics"])
    manifests = {}
    for split in ("test-clean", "test-other"):
        manifest = run_root / "manifests/libritts24k" / f"{split}_eval.jsonl.gz"
        if (
            not manifest.is_file()
            or sha256_file(manifest) != gate["test_manifests"][split]
        ):
            raise ValueError(f"Test manifest does not match frozen gate: {split}")
        manifests[split] = {
            "path": manifest.resolve(),
            **_manifest_identity(manifest),
        }

    tasks = []
    for system in gate["systems"]:
        for split in ("test-clean", "test-other"):
            manifest = manifests[split]
            run_id = _run_id(system, split, gate_sha256)
            assets = _task_assets(system, shared, ours, metrics)
            environment = {
                ENV_BY_ASSET[name]: record["path"]
                for name, record in assets.items()
                if name in ENV_BY_ASSET
            }
            if system["adapter"] == "ours":
                environment["OURS_CKPT"] = assets["ours_checkpoint"]["path"]
            if system["adapter"] == "dac":
                environment["DAC_QUANTIZERS"] = str(system["num_parallel_streams"])
            tasks.append(
                {
                    "task_id": run_id,
                    "adapter": system["adapter"],
                    "system": system["system"],
                    "variant": system["variant"],
                    "seed": int(system["seed"]),
                    "split": split,
                    "roles": sorted(system["roles"]),
                    "config_hash": system["config_hash"],
                    "checkpoint_hash": system["checkpoint_hash"],
                    "decode_seed_or_policy": system["decode_seed_or_policy"],
                    "manifest": {
                        "path": str(manifest["path"]),
                        "sha256": gate["test_manifests"][split],
                        "cuts": manifest["cuts"],
                        "cut_ids_sha256": manifest["cut_ids_sha256"],
                    },
                    "assets": assets,
                    "environment": environment,
                    "run_directory": str((run_root / "evaluations" / run_id).resolve()),
                }
            )
    tasks.sort(
        key=lambda task: (
            task["adapter"],
            task["system"],
            task["seed"],
            task["split"],
        )
    )
    return tasks


def prepare_test_queue(
    *,
    test_gate: str | Path,
    asset_manifest: str | Path,
    run_root: str | Path,
    cache_root: str | Path,
    output: str | Path,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    repository_root = Path(repository_root).expanduser().resolve()
    test_gate = Path(test_gate).expanduser().resolve()
    asset_manifest = Path(asset_manifest).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Final-test queue output must end in .json")
    _validate_external_root(run_root, repository_root, "run root")
    _validate_external_root(cache_root, repository_root, "cache root")
    _validate_external_root(output, repository_root, "queue output")
    _validate_external_root(asset_manifest, repository_root, "asset manifest")
    _validate_external_root(test_gate, repository_root, "test gate")
    if not run_root.is_dir() or not cache_root.is_dir():
        raise FileNotFoundError("Run and cache roots must already exist")
    gate, gate_sha256 = _validate_gate(test_gate, repository_root)
    shared, ours = _resolve_assets(asset_manifest, gate)
    metrics = sorted(str(metric) for metric in gate["metrics"])
    tasks = _build_tasks(
        gate=gate,
        gate_sha256=gate_sha256,
        run_root=run_root,
        shared=shared,
        ours=ours,
    )
    queue = {
        "schema_version": 1,
        "status": "FINAL_TEST_QUEUE_FROZEN",
        "git_commit": gate["git_commit"],
        "test_gate": {"path": str(test_gate), "sha256": gate_sha256},
        "asset_manifest": {
            "path": str(asset_manifest),
            "sha256": sha256_file(asset_manifest),
        },
        "run_root": str(run_root),
        "cache_root": str(cache_root),
        "metrics": metrics,
        "unique_system_identities": len(gate["systems"]),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    _write_immutable(output, queue)
    return queue


def _validate_queue(
    path: Path, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = _load_json(path)
    if (
        queue.get("schema_version") != 1
        or queue.get("status") != "FINAL_TEST_QUEUE_FROZEN"
    ):
        raise ValueError("Invalid final-test queue")
    gate_record = queue.get("test_gate")
    asset_record = queue.get("asset_manifest")
    if not isinstance(gate_record, dict) or not isinstance(asset_record, dict):
        raise ValueError("Final-test queue lacks gate or asset manifest")
    gate_path = Path(str(gate_record.get("path", ""))).expanduser().resolve()
    asset_path = Path(str(asset_record.get("path", ""))).expanduser().resolve()
    if sha256_file(gate_path) != gate_record.get("sha256"):
        raise ValueError("Final-test gate changed after queue preparation")
    if sha256_file(asset_path) != asset_record.get("sha256"):
        raise ValueError("Final-test asset manifest changed after queue preparation")
    gate, gate_sha256 = _validate_gate(gate_path, repository_root)
    if gate_sha256 != gate_record["sha256"] or gate["git_commit"] != queue.get(
        "git_commit"
    ):
        raise ValueError("Final-test queue binds another gate or commit")
    run_root = Path(str(queue.get("run_root", ""))).expanduser().resolve()
    cache_root = Path(str(queue.get("cache_root", ""))).expanduser().resolve()
    _validate_external_root(run_root, repository_root, "run root")
    _validate_external_root(cache_root, repository_root, "cache root")
    _validate_external_root(asset_path, repository_root, "asset manifest")
    if str(run_root) != queue.get("run_root") or str(cache_root) != queue.get(
        "cache_root"
    ):
        raise ValueError("Final-test queue contains non-canonical roots")
    shared, ours = _resolve_assets(asset_path, gate)
    expected_tasks = _build_tasks(
        gate=gate,
        gate_sha256=gate_sha256,
        run_root=run_root,
        shared=shared,
        ours=ours,
    )
    tasks = queue.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != queue.get("task_count")
        or len(tasks) != 2 * int(queue.get("unique_system_identities", -1))
        or len({task.get("task_id") for task in tasks}) != len(tasks)
    ):
        raise ValueError("Final-test queue has inconsistent task coverage")
    if tasks != expected_tasks or queue.get("metrics") != sorted(gate["metrics"]):
        raise ValueError("Final-test queue does not rebuild from its frozen inputs")
    return queue, gate


def _load_statistics(path: Path) -> dict[str, Any]:
    statistics = _load_json(path)
    missing = [name for name in PRIMARY_BOOTSTRAP_METRICS if name not in statistics]
    if missing:
        raise ValueError(f"Statistics artifact lacks metrics: {missing}")
    low = [
        name
        for name in PRIMARY_BOOTSTRAP_METRICS
        if int(statistics[name].get("resamples", 0)) < 10_000
        or int(statistics[name].get("finite_resamples", 0)) < 10_000
    ]
    if low:
        raise ValueError(f"Statistics artifact has fewer than 10,000 resamples: {low}")
    return statistics


def verify_task(
    task: dict[str, Any], queue: dict[str, Any], gate: dict[str, Any]
) -> TaskState:
    run_directory = Path(task["run_directory"])
    present = [name for name in REQUIRED_TASK_FILES if (run_directory / name).is_file()]
    if not present:
        return TaskState(False, False)
    if len(present) != len(REQUIRED_TASK_FILES):
        if (run_directory / "summary.json").is_file():
            raise ValueError(f"Completed-looking task is incomplete: {task['task_id']}")
        return TaskState(False, False)

    records_path = run_directory / "per_utterance.jsonl"
    records = load_jsonl(records_path)
    if len({record["cut_id"] for record in records}) != len(records):
        raise ValueError(f"Task contains duplicate cut IDs: {task['task_id']}")
    summary = _load_json(run_directory / "summary.json")
    rebuilt = summarize_records(records)
    rebuilt["per_utterance_sha256"] = sha256_file(records_path)
    if summary != rebuilt:
        raise ValueError(f"Task summary does not rebuild: {task['task_id']}")
    config = _load_yaml(run_directory / "resolved_config.yaml")
    data_manifest = _load_json(run_directory / "data_manifest.json")
    model_metadata = _load_json(run_directory / "model_metadata.json")
    git_state = _load_json(run_directory / "git_state.json")
    expected = {
        "run_id": task["task_id"],
        "system": task["adapter"],
        "split": task["split"],
        "seed": task["seed"],
        "manifest_sha256": task["manifest"]["sha256"],
        "system_config_hash": task["config_hash"],
        "decode_seed_or_policy": task["decode_seed_or_policy"],
        "test_gate_sha256": queue["test_gate"]["sha256"],
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Task {task['task_id']} has unexpected {field}")
    if (
        config.get("max_cuts") is not None
        or sorted(config.get("metrics", [])) != queue["metrics"]
    ):
        raise ValueError(f"Task {task['task_id']} changed test coverage or metrics")
    if (
        data_manifest.get("sha256") != task["manifest"]["sha256"]
        or int(data_manifest.get("cuts", -1)) != task["manifest"]["cuts"]
        or len(records) != task["manifest"]["cuts"]
        or canonical_hash(sorted(record["cut_id"] for record in records))
        != task["manifest"]["cut_ids_sha256"]
        or summary.get("run_id") != task["task_id"]
        or summary.get("system") != task["system"]
        or summary.get("variant") != task["variant"]
        or int(summary.get("seed", -1)) != task["seed"]
        or summary.get("split") != task["split"]
        or summary.get("config_hash") != task["config_hash"]
        or summary.get("checkpoint_hash") != task["checkpoint_hash"]
    ):
        raise ValueError(f"Task {task['task_id']} output identity mismatch")
    if canonical_hash(model_metadata.get("metrics")) != gate.get(
        "metric_metadata_hash"
    ):
        raise ValueError(f"Task {task['task_id']} metric assets changed")
    if git_state.get("dirty") or git_state.get("commit") != gate.get("git_commit"):
        raise ValueError(f"Task {task['task_id']} was not run from the frozen commit")
    statistics_path = run_directory / "statistics.json"
    statistics_complete = statistics_path.is_file()
    if statistics_complete:
        _load_statistics(statistics_path)
    return TaskState(True, statistics_complete)


def _task_environment(
    task: dict[str, Any], queue: dict[str, Any], gpu: str
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {str(key): str(value) for key, value in task["environment"].items()}
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "DMEL_RUN_ROOT": queue["run_root"],
            "DMEL_CACHE_ROOT": queue["cache_root"],
            "DMEL_METRICS": ",".join(queue["metrics"]),
            "TEST_GATE": queue["test_gate"]["path"],
        }
    )
    return environment


def _evaluation_command(task: dict[str, Any], repository_root: Path) -> list[str]:
    return [
        str(repository_root / "scripts/evaluate.sh"),
        task["adapter"],
        task["split"],
        task["task_id"],
        "--seed",
        str(task["seed"]),
        "--max-duration",
        "30",
        "--audio-samples",
        "8",
        "--dependency-sample-limit",
        "50000",
    ]


def _statistics_command(task: dict[str, Any], repository_root: Path) -> list[str]:
    return [str(repository_root / "scripts/statistics.sh"), task["run_directory"]]


def run_test_queue(
    *,
    queue_path: str | Path,
    gpus: Iterable[str],
    repository_root: str | Path = ".",
    dry_run: bool = False,
    poll_seconds: float = 1.0,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict[str, Any]:
    repository_root = Path(repository_root).expanduser().resolve()
    queue_path = Path(queue_path).expanduser().resolve()
    queue, gate = _validate_queue(queue_path, repository_root)
    gpu_list = [str(value).strip() for value in gpus if str(value).strip()]
    if (
        not gpu_list
        or len(gpu_list) != len(set(gpu_list))
        or any("," in value for value in gpu_list)
    ):
        raise ValueError("Provide a non-empty list of unique single-GPU identifiers")

    work = []
    completed = []
    for task in queue["tasks"]:
        state = verify_task(task, queue, gate)
        if state.statistics_complete:
            completed.append(task["task_id"])
        else:
            phase = "statistics" if state.evaluation_complete else "evaluation"
            work.append({"task": task, "phase": phase})
    if dry_run:
        return {
            "status": "FINAL_TEST_QUEUE_DRY_RUN",
            "queue_sha256": sha256_file(queue_path),
            "gpus": gpu_list,
            "completed": sorted(completed),
            "pending": [item["task"]["task_id"] for item in work],
            "task_count": len(queue["tasks"]),
        }

    available = list(gpu_list)
    running: dict[Any, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    try:
        while work or running:
            while available and work and not failures:
                item = work.pop(0)
                gpu = available.pop(0)
                task = item["task"]
                command = (
                    _evaluation_command(task, repository_root)
                    if item["phase"] == "evaluation"
                    else _statistics_command(task, repository_root)
                )
                try:
                    process = popen(
                        command,
                        cwd=repository_root,
                        env=_task_environment(task, queue, gpu),
                    )
                except OSError as error:
                    available.append(gpu)
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "phase": item["phase"],
                            "exit_code": 1,
                            "error": str(error),
                        }
                    )
                    break
                running[process] = {**item, "gpu": gpu, "command": command}
            finished = [process for process in running if process.poll() is not None]
            if not finished:
                if running:
                    time.sleep(poll_seconds)
                    continue
                break
            for process in finished:
                item = running.pop(process)
                available.append(item["gpu"])
                task = item["task"]
                if process.returncode != 0:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "phase": item["phase"],
                            "exit_code": process.returncode,
                        }
                    )
                    continue
                try:
                    state = verify_task(task, queue, gate)
                except Exception as error:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "phase": item["phase"],
                            "exit_code": 1,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                if item["phase"] == "evaluation" and not state.statistics_complete:
                    work.insert(0, {"task": task, "phase": "statistics"})
                elif state.statistics_complete:
                    completed.append(task["task_id"])
                else:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "phase": item["phase"],
                            "exit_code": 1,
                        }
                    )
            available.sort(key=gpu_list.index)
    except KeyboardInterrupt:
        for process in running:
            process.terminate()
        for process in running:
            process.wait()
        raise

    status = (
        "FINAL_TEST_QUEUE_COMPLETE"
        if not failures and not work
        else "FINAL_TEST_QUEUE_FAILED"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "test_gate_sha256": queue["test_gate"]["sha256"],
        "gpus": gpu_list,
        "task_count": len(queue["tasks"]),
        "completed": sorted(set(completed)),
        "failures": failures,
        "not_started": [item["task"]["task_id"] for item in work],
    }
    state_path = queue_path.with_name(f"{queue_path.stem}.state.json")
    temporary = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_json(result))
    os.replace(temporary, state_path)
    return result


def _parse_gpus(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run the frozen final-test queue"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--test-gate", type=Path, required=True)
    prepare.add_argument("--assets", type=Path, required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--cache-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repository-root", type=Path, default=Path("."))
    run = subparsers.add_parser("run")
    run.add_argument("--queue", type=Path, required=True)
    run.add_argument("--gpus", required=True)
    run.add_argument("--repository-root", type=Path, default=Path("."))
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_test_queue(
            test_gate=args.test_gate,
            asset_manifest=args.assets,
            run_root=args.run_root,
            cache_root=args.cache_root,
            output=args.output,
            repository_root=args.repository_root,
        )
    else:
        result = run_test_queue(
            queue_path=args.queue,
            gpus=_parse_gpus(args.gpus),
            repository_root=args.repository_root,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] not in {"FINAL_TEST_QUEUE_FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
