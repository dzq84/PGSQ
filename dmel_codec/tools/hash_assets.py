from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path, *, source: str, revision: str, license_name: str
) -> dict:
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "source": source,
        "revision": revision,
        "license": license_name,
        "root": str(root.resolve()),
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash every file in an external asset")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    args = parser.parse_args()
    document = build_manifest(
        args.root,
        source=args.source,
        revision=args.revision,
        license_name=args.license_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
