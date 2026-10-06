from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruction_cache_key(
    *,
    system_config_hash: str,
    checkpoint_hash: str | None,
    cut_id: str,
    decode_seed_or_policy: str,
) -> str:
    return canonical_hash(
        {
            "schema": 1,
            "system_config_hash": system_config_hash,
            "checkpoint_hash": checkpoint_hash,
            "cut_id": cut_id,
            "decode_seed_or_policy": decode_seed_or_policy,
        }
    )


class ReconstructionCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Cache key must be a lowercase SHA-256 hex digest")
        return self.root / key[:2] / f"{key}.pt"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1 or payload.get("cache_key") != key:
            raise ValueError(f"Invalid reconstruction cache entry: {path}")
        return payload

    def save(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, "cache_key": key, **payload}
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        torch.save(document, temporary)
        os.replace(temporary, path)
        return path


class JsonCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Cache key must be a lowercase SHA-256 hex digest")
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("cache_key") != key:
            raise ValueError(f"Invalid JSON cache entry: {path}")
        return payload

    def save(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, "cache_key": key, **payload}
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(document, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path
