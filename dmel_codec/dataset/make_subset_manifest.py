from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

from lhotse import CutSet

from dmel_codec.dataset.prepare_libritts import sha256_file


SCHEMA_VERSION = 1


def _source_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _temporary_manifest_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp-{os.getpid()}.jsonl.gz")


def _cut_id_hash(cut_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for cut_id in sorted(cut_ids):
        digest.update(cut_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rank(seed: int, cut_id: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{seed}:{cut_id}".encode("utf-8")).digest(), cut_id


def _metadata_path(output_path: Path) -> Path:
    name = output_path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return output_path.with_name(f"{name}.metadata.json")


def _input_metadata(paths: Sequence[Path]) -> list[dict]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _can_reuse(
    output_path: Path,
    metadata_path: Path,
    *,
    inputs: list[dict],
    max_cuts: int | None,
    seed: int,
) -> dict | None:
    if not output_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "generator_sha256": _source_sha256(),
        "inputs": inputs,
        "selection": {
            "method": "all" if max_cuts is None else "sha256_bottom_k",
            "max_cuts": max_cuts,
            "seed": seed,
        },
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    output = metadata.get("output", {})
    if (
        output.get("path") != str(output_path)
        or output.get("bytes") != output_path.stat().st_size
        or output.get("sha256") != sha256_file(output_path)
    ):
        return None
    return metadata


def make_subset_manifest(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    max_cuts: int | None = None,
    seed: int = 2026,
    force: bool = False,
) -> dict:
    if not input_paths:
        raise ValueError("At least one input manifest is required")
    if max_cuts is not None and max_cuts < 1:
        raise ValueError("max_cuts must be positive")

    inputs = [Path(path).expanduser().resolve() for path in input_paths]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(inputs)) != len(inputs):
        raise ValueError("Input manifests must be unique")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(output)
    input_metadata = _input_metadata(inputs)
    if not force:
        reusable = _can_reuse(
            output,
            metadata_path,
            inputs=input_metadata,
            max_cuts=max_cuts,
            seed=seed,
        )
        if reusable is not None:
            return reusable

    cuts = []
    seen_ids: set[str] = set()
    for path in inputs:
        for cut in CutSet.from_file(path):
            if cut.id in seen_ids:
                raise ValueError(f"Duplicate cut ID across inputs: {cut.id}")
            seen_ids.add(cut.id)
            cuts.append(cut)
    if not cuts:
        raise ValueError("Input manifests contain no cuts")

    if max_cuts is not None and len(cuts) > max_cuts:
        cuts = sorted(cuts, key=lambda cut: _rank(seed, cut.id))[:max_cuts]
    selected_ids = [cut.id for cut in cuts]
    split_counts = Counter(
        str((cut.custom or {}).get("split", "unknown")) for cut in cuts
    )
    total_duration = sum(float(cut.duration) for cut in cuts)
    capped_duration = sum(min(float(cut.duration), 3.0) for cut in cuts)

    temporary = _temporary_manifest_path(output)
    temporary.unlink(missing_ok=True)
    try:
        CutSet.from_cuts(cuts).to_file(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generator_sha256": _source_sha256(),
        "inputs": input_metadata,
        "selection": {
            "method": "all" if max_cuts is None else "sha256_bottom_k",
            "max_cuts": max_cuts,
            "seed": seed,
        },
        "cut_count": len(cuts),
        "cut_id_sha256": _cut_id_hash(selected_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "total_duration_seconds": total_duration,
        "three_second_capped_duration_seconds": capped_duration,
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    _atomic_write_json(metadata_path, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a traceable merge or deterministic subset of Lhotse cuts"
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cuts", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    metadata = make_subset_manifest(
        args.input,
        args.output,
        max_cuts=args.max_cuts,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
