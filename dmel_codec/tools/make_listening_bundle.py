from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import wave
from pathlib import Path
from typing import Any

from dmel_codec.evaluation.cache import sha256_file


ROUTES = ("source", "codec", "vocoder", "joint")
SPLITS = ("dev-clean", "dev-other")
SAFE_CUT_ID = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_ROUTE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_or_verify(path: Path, value: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"Existing listening artifact differs: {path}")
        return
    _atomic_text(path, value)


def _wav_metadata(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            frames = stream.getnframes()
            sample_width = stream.getsampwidth()
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Unreadable PCM WAV: {path}") from error
    if channels != 1 or sample_rate != 24_000 or frames <= 0:
        raise ValueError(
            f"Listening WAV must be finite-length mono 24 kHz: {path} "
            f"(channels={channels}, sample_rate={sample_rate}, frames={frames})"
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "channels": channels,
        "sample_rate": sample_rate,
        "frames": frames,
        "sample_width_bytes": sample_width,
    }


def _sample_files(run_dir: Path) -> dict[str, dict[str, Path]]:
    sample_dir = run_dir / "audio_samples"
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)
    references: dict[str, Path] = {}
    reconstructions: dict[str, Path] = {}
    for path in sample_dir.glob("*.wav"):
        if path.name.endswith("_reference.wav"):
            cut_id = path.name[: -len("_reference.wav")]
            references[cut_id] = path
        elif path.name.endswith("_reconstruction.wav"):
            cut_id = path.name[: -len("_reconstruction.wav")]
            reconstructions[cut_id] = path
    if set(references) != set(reconstructions) or not references:
        raise ValueError(f"Incomplete audio sample pairs: {sample_dir}")
    if not all(SAFE_CUT_ID.fullmatch(cut_id) for cut_id in references):
        raise ValueError(f"Unsafe cut ID in listening samples: {sample_dir}")
    return {
        cut_id: {
            "reference": references[cut_id],
            "reconstruction": reconstructions[cut_id],
        }
        for cut_id in sorted(references)
    }


def _validated_run(run_dir: Path, split: str) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _load_json(summary_path)
    if summary.get("split") != split:
        raise ValueError(f"Listening run split mismatch: {run_dir}")
    if int(summary.get("failed_utterances", -1)) != 0:
        raise ValueError(f"Listening run has reconstruction failures: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id"),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_hash": summary.get("checkpoint_hash"),
        "config_hash": summary.get("config_hash"),
        "samples": _sample_files(run_dir),
    }


def _condition_order(
    seed: int, split: str, cut_id: str, routes: tuple[str, ...]
) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{split}:{cut_id}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    ordered = list(routes)
    generator.shuffle(ordered)
    return ordered


def _ensure_symlink(path: Path, target: Path) -> None:
    relative = os.path.relpath(target, path.parent)
    if path.is_symlink():
        if os.readlink(path) != relative:
            raise FileExistsError(f"Existing listening link differs: {path}")
        return
    if path.exists():
        raise FileExistsError(f"Refusing to replace listening artifact: {path}")
    path.symlink_to(relative)


def _html_document(
    items: list[dict[str, Any]],
    manifest_sha256: str,
    *,
    title: str,
    ratings_status: str,
    ratings_filename: str,
) -> str:
    groups = []
    for item in items:
        controls = [
            '<div class="condition"><strong>Reference</strong>'
            f'<audio controls preload="none" src="{html.escape(item["reference_file"])}">'
            "</audio></div>"
        ]
        for condition in item["conditions"]:
            controls.append(
                '<div class="condition" '
                f'data-split="{html.escape(item["split"])}" '
                f'data-cut-id="{html.escape(item["cut_id"])}" '
                f'data-label="{condition["blind_label"]}">'
                f"<strong>{condition['blind_label']}</strong>"
                f'<audio controls preload="none" src="{html.escape(condition["file"])}">'
                "</audio>"
                f'<select aria-label="Score {condition["blind_label"]}">'
                '<option value="" selected disabled>Score</option>'
                '<option value="1">1</option><option value="2">2</option>'
                '<option value="3">3</option><option value="4">4</option>'
                '<option value="5">5</option></select></div>'
            )
        groups.append(
            '<section class="cut">'
            f"<h2>{html.escape(item['split'])} / {html.escape(item['cut_id'])}</h2>"
            + "".join(controls)
            + "</section>"
        )
    return (
        (
            """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { margin: 0 auto; max-width: 1080px; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 20px; }
    h2 { font-size: 15px; margin: 0 0 12px; }
    .cut { border-top: 1px solid #8886; padding: 18px 0; }
    .condition { display: grid; grid-template-columns: 72px 1fr 88px; gap: 12px;
      align-items: center; min-height: 42px; }
    audio { width: 100%; }
    select, button { min-height: 36px; font: inherit; }
    .actions { position: sticky; bottom: 0; padding: 14px 0;
      background: Canvas; border-top: 1px solid #8886; }
    @media (max-width: 560px) {
      body { padding: 16px; }
      .condition { grid-template-columns: 1fr; gap: 4px; margin-bottom: 12px; }
    }
  </style>
</head>
<body>
  <h1>__TITLE__</h1>
"""
            + "\n".join(groups)
            + f"""
  <div class="actions"><button id="download-ratings" type="button">Download ratings</button></div>
  <script>
    const manifestSha256 = {json.dumps(manifest_sha256)};
    document.getElementById("download-ratings").addEventListener("click", () => {{
      const grouped = new Map();
      let incomplete = false;
      document.querySelectorAll(".condition[data-label]").forEach((row) => {{
        const select = row.querySelector("select");
        if (!select.value) {{ incomplete = true; return; }}
        const key = `${{row.dataset.split}}:${{row.dataset.cutId}}`;
        if (!grouped.has(key)) {{
          grouped.set(key, {{
            split: row.dataset.split,
            cut_id: row.dataset.cutId,
            scores: {{}},
          }});
        }}
        grouped.get(key).scores[row.dataset.label] = Number(select.value);
      }});
      if (incomplete) {{
        window.alert("Rate every candidate before downloading.");
        return;
      }}
      const payload = {{
        schema_version: 1,
        status: __RATINGS_STATUS__,
        listening_manifest_sha256: manifestSha256,
        created_at: new Date().toISOString(),
        items: Array.from(grouped.values()),
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"],
        {{type: "application/json"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = __RATINGS_FILENAME__;
      link.click();
      URL.revokeObjectURL(link.href);
    }});
  </script>
</body>
</html>
"""
        )
        .replace("__TITLE__", html.escape(title))
        .replace("__RATINGS_STATUS__", json.dumps(ratings_status))
        .replace("__RATINGS_FILENAME__", json.dumps(ratings_filename))
    )


def make_listening_bundle(
    run_directories: dict[str, dict[str, Path]],
    *,
    output_dir: Path,
    randomization_seed: int = 2027,
    expected_cuts_per_split: int = 8,
    routes: tuple[str, ...] = ROUTES,
    title: str = "M+ matched listening",
    status_prefix: str = "MPLUS_MATCHED",
    ratings_filename: str = "mplus_matched_listening_ratings.json",
) -> dict[str, Any]:
    if not 2 <= len(routes) <= 26 or len(set(routes)) != len(routes):
        raise ValueError("Listening routes must contain 2-26 unique values")
    if not all(SAFE_ROUTE.fullmatch(route) for route in routes):
        raise ValueError("Listening route names must be safe lowercase identifiers")
    if set(run_directories) != set(routes):
        raise ValueError(f"Listening inputs must cover exactly {routes}")
    if expected_cuts_per_split <= 0:
        raise ValueError("expected_cuts_per_split must be positive")
    if not title.strip() or not status_prefix.strip() or not ratings_filename.strip():
        raise ValueError(
            "Listening title, status prefix and ratings filename are required"
        )
    if Path(ratings_filename).name != ratings_filename:
        raise ValueError("Ratings filename must not contain a directory")
    for route in routes:
        if set(run_directories[route]) != set(SPLITS):
            raise ValueError(f"Listening route {route} must cover exactly {SPLITS}")

    runs = {
        route: {
            split: _validated_run(run_directories[route][split], split)
            for split in SPLITS
        }
        for route in routes
    }
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    for split in SPLITS:
        cut_sets = {route: set(runs[route][split]["samples"]) for route in routes}
        first_set = cut_sets[routes[0]]
        if any(cuts != first_set for cuts in cut_sets.values()):
            raise ValueError(f"Listening sample cut IDs differ for {split}")
        if len(first_set) != expected_cuts_per_split:
            raise ValueError(
                f"Expected {expected_cuts_per_split} listening cuts for {split}, "
                f"found {len(first_set)}"
            )
        for cut_id in sorted(first_set):
            reference_records = {
                route: _wav_metadata(runs[route][split]["samples"][cut_id]["reference"])
                for route in routes
            }
            reference_hashes = {
                record["sha256"] for record in reference_records.values()
            }
            if len(reference_hashes) != 1:
                raise ValueError(
                    f"Reference WAV differs across routes: {split}/{cut_id}"
                )
            reference = reference_records[routes[0]]
            prefix = f"{split}_{cut_id}"
            reference_name = f"{prefix}_reference.wav"
            _ensure_symlink(
                output_dir / reference_name,
                runs[routes[0]][split]["samples"][cut_id]["reference"],
            )
            conditions = []
            key_conditions = []
            for index, route in enumerate(
                _condition_order(randomization_seed, split, cut_id, routes)
            ):
                blind_label = chr(ord("A") + index)
                filename = f"{prefix}_{blind_label}.wav"
                reconstruction_path = runs[route][split]["samples"][cut_id][
                    "reconstruction"
                ]
                reconstruction = _wav_metadata(reconstruction_path)
                if (
                    reconstruction["frames"] != reference["frames"]
                    or reconstruction["sample_width_bytes"]
                    != reference["sample_width_bytes"]
                ):
                    raise ValueError(
                        f"Paired WAV format/length mismatch: {split}/{cut_id}/{route}"
                    )
                _ensure_symlink(output_dir / filename, reconstruction_path)
                conditions.append(
                    {
                        "blind_label": blind_label,
                        "file": filename,
                        "wav": reconstruction,
                    }
                )
                key_conditions.append(
                    {"blind_label": blind_label, "route": route, "file": filename}
                )
            items.append(
                {
                    "split": split,
                    "cut_id": cut_id,
                    "reference_file": reference_name,
                    "reference_wav": reference,
                    "conditions": conditions,
                }
            )
            key_items.append(
                {"split": split, "cut_id": cut_id, "conditions": key_conditions}
            )

    manifest = {
        "schema_version": 1,
        "status": f"{status_prefix}_LISTENING_BUNDLE",
        "randomization_seed": randomization_seed,
        "routes": list(routes),
        "splits": list(SPLITS),
        "cuts_per_split": expected_cuts_per_split,
        "runs": {
            route: {
                split: {
                    key: value
                    for key, value in runs[route][split].items()
                    if key != "samples"
                }
                for split in SPLITS
            }
            for route in routes
        },
        "items": items,
    }
    key = {
        "schema_version": 1,
        "status": f"{status_prefix}_LISTENING_KEY",
        "randomization_seed": randomization_seed,
        "items": key_items,
    }
    readme = (
        f"# {title}\n\n"
        "Open `index.html` to play the matched development samples. Candidate "
        "labels are independently randomized for each cut. `listening_key.json` is "
        "the condition key and should remain closed until ratings are recorded.\n"
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_sha256 = hashlib.sha256(manifest_text.encode()).hexdigest()
    _write_or_verify(output_dir / "manifest.json", manifest_text)
    _write_or_verify(
        output_dir / "listening_key.json",
        json.dumps(key, indent=2, sort_keys=True) + "\n",
    )
    _write_or_verify(output_dir / "README.md", readme)
    _write_or_verify(
        output_dir / "index.html",
        _html_document(
            items,
            manifest_sha256,
            title=title,
            ratings_status=f"{status_prefix}_LISTENING_RATINGS",
            ratings_filename=ratings_filename,
        ),
    )
    return manifest


def _parse_runs(
    values: list[str], routes: tuple[str, ...] = ROUTES
) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for value in values:
        identity, separator, path_text = value.partition("=")
        route, split_separator, split = identity.partition(":")
        if (
            not separator
            or not split_separator
            or route not in routes
            or split not in SPLITS
        ):
            raise ValueError("--run values must use ROUTE:SPLIT=PATH")
        if split in result.setdefault(route, {}):
            raise ValueError(f"Duplicate listening input: {route}:{split}")
        result[route][split] = Path(path_text)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic matched source/A/B/C listening bundle"
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--randomization-seed", type=int, default=2027)
    parser.add_argument("--expected-cuts-per-split", type=int, default=8)
    parser.add_argument(
        "--routes",
        default=",".join(ROUTES),
        help="Comma-separated candidate route names",
    )
    parser.add_argument("--title", default="M+ matched listening")
    parser.add_argument("--status-prefix", default="MPLUS_MATCHED")
    parser.add_argument(
        "--ratings-filename", default="mplus_matched_listening_ratings.json"
    )
    args = parser.parse_args()
    routes = tuple(part.strip() for part in args.routes.split(",") if part.strip())
    result = make_listening_bundle(
        _parse_runs(args.run, routes),
        output_dir=args.output_dir,
        randomization_seed=args.randomization_seed,
        expected_cuts_per_split=args.expected_cuts_per_split,
        routes=routes,
        title=args.title,
        status_prefix=args.status_prefix,
        ratings_filename=args.ratings_filename,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
