from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import soundfile as sf
from lhotse import AudioSource, CutSet, MonoCut, Recording
from lhotse.supervision import SupervisionSegment


LOG = logging.getLogger("prepare_libritts")
SCHEMA_VERSION = 1
SAMPLE_RATE = 24_000
SPLITS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)
TRAIN_SPLITS = SPLITS[:3]


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    utterance_id: str
    speaker_id: str
    chapter_id: str
    num_samples: int
    sample_rate: int
    channels: int
    format: str
    subtype: str

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _read_speakers(data_root: Path) -> dict[str, str]:
    path = data_root / "SPEAKERS.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LibriTTS speaker metadata: {path}")
    speakers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|", maxsplit=4)]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        speakers[fields[0]] = fields[1]
    return speakers


def _read_transcripts(split_dir: Path) -> dict[str, tuple[str, str]]:
    transcripts: dict[str, tuple[str, str]] = {}
    for path in sorted(split_dir.rglob("*.trans.tsv")):
        with path.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                fields = raw_line.rstrip("\n").split("\t", maxsplit=2)
                if len(fields) != 3:
                    raise ValueError(f"Malformed transcript at {path}:{line_number}")
                utterance_id, original, normalized = fields
                if not utterance_id or not original.strip() or not normalized.strip():
                    raise ValueError(f"Empty transcript at {path}:{line_number}")
                if utterance_id in transcripts:
                    raise ValueError(f"Duplicate transcript ID {utterance_id} in {split_dir}")
                transcripts[utterance_id] = (original, normalized)
    return transcripts


def _inspect_audio(path: Path) -> AudioInfo:
    info = sf.info(path)
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected LibriTTS utterance ID: {path.stem}")
    return AudioInfo(
        path=path.absolute(),
        utterance_id=path.stem,
        speaker_id=parts[0],
        chapter_id=parts[1],
        num_samples=int(info.frames),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        format=str(info.format),
        subtype=str(info.subtype),
    )


def _audit_split(
    data_root: Path,
    split: str,
    *,
    num_jobs: int,
    sample_size: int,
    seed: int,
) -> tuple[list[AudioInfo], dict[str, tuple[str, str]], dict]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing required LibriTTS split: {split_dir}")
    paths = sorted(split_dir.rglob("*.wav"))
    if not paths:
        raise ValueError(f"No WAV files found in {split_dir}")
    LOG.info("Auditing %s: %d WAV files", split, len(paths))
    with ThreadPoolExecutor(max_workers=max(1, num_jobs)) as pool:
        audio = list(pool.map(_inspect_audio, paths, chunksize=64))

    utterance_ids = [item.utterance_id for item in audio]
    if len(set(utterance_ids)) != len(utterance_ids):
        raise ValueError(f"Duplicate recording/utterance IDs in {split}")
    invalid_audio = [
        item.utterance_id
        for item in audio
        if item.sample_rate != SAMPLE_RATE or item.channels != 1 or item.num_samples <= 0
    ]
    if invalid_audio:
        raise ValueError(
            f"{split} contains non-mono/non-24k/empty audio; first IDs: {invalid_audio[:10]}"
        )

    transcripts = _read_transcripts(split_dir)
    audio_id_set = set(utterance_ids)
    transcript_id_set = set(transcripts)
    if audio_id_set != transcript_id_set:
        raise ValueError(
            f"{split} transcript/audio mismatch: "
            f"missing_text={sorted(audio_id_set - transcript_id_set)[:10]} "
            f"missing_audio={sorted(transcript_id_set - audio_id_set)[:10]}"
        )

    rng = random.Random(f"{seed}:{split}")
    sampled = rng.sample(audio, k=min(sample_size, len(audio)))
    waveform_checks = []
    for item in sampled:
        waveform, loaded_rate = sf.read(
            item.path, dtype="float32", always_2d=True
        )
        if loaded_rate != item.sample_rate or waveform.shape != (
            item.num_samples,
            item.channels,
        ):
            raise ValueError(
                f"Waveform/header mismatch for {item.utterance_id}: "
                f"loaded={waveform.shape}@{loaded_rate}, "
                f"header={item.num_samples}x{item.channels}@{item.sample_rate}"
            )
        waveform_checks.append(
            {
                "utterance_id": item.utterance_id,
                "num_samples": item.num_samples,
                "duration_seconds": item.duration,
            }
        )

    total_samples = sum(item.num_samples for item in audio)
    durations = [item.duration for item in audio]
    summary = {
        "utterances": len(audio),
        "recording_ids_unique": True,
        "transcripts": len(transcripts),
        "transcripts_nonempty": True,
        "speakers": len({item.speaker_id for item in audio}),
        "chapters": len({item.chapter_id for item in audio}),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "total_samples": total_samples,
        "total_duration_seconds": total_samples / SAMPLE_RATE,
        "total_duration_hours": total_samples / SAMPLE_RATE / 3600,
        "min_duration_seconds": min(durations),
        "max_duration_seconds": max(durations),
        "utterances_shorter_than_3s": sum(duration < 3.0 for duration in durations),
        "formats": sorted({item.format for item in audio}),
        "subtypes": sorted({item.subtype for item in audio}),
        "waveform_metadata_checks": waveform_checks,
    }
    return audio, transcripts, summary


def _make_cut(
    item: AudioInfo,
    transcripts: dict[str, tuple[str, str]],
    speakers: dict[str, str],
    split: str,
    *,
    include_supervision: bool,
) -> MonoCut:
    recording = Recording(
        id=item.utterance_id,
        sources=[AudioSource(type="file", channels=[0], source=str(item.path))],
        sampling_rate=item.sample_rate,
        num_samples=item.num_samples,
        duration=item.duration,
    )
    supervisions = []
    if include_supervision:
        original, normalized = transcripts[item.utterance_id]
        supervisions = [
            SupervisionSegment(
                id=item.utterance_id,
                recording_id=item.utterance_id,
                start=0.0,
                duration=item.duration,
                channel=0,
                text=normalized,
                speaker=item.speaker_id,
                gender=speakers.get(item.speaker_id),
                custom={"original_text": original, "split": split},
            )
        ]
    return MonoCut(
        id=item.utterance_id,
        start=0.0,
        duration=item.duration,
        channel=0,
        supervisions=supervisions,
        recording=recording,
        custom={
            "split": split,
            "speaker_id": item.speaker_id,
            "chapter_id": item.chapter_id,
            "num_samples": item.num_samples,
        },
    )


def _temporary_manifest_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp.jsonl.gz")


def _write_manifest(cuts: Iterable[MonoCut], path: Path) -> None:
    temporary = _temporary_manifest_path(path)
    temporary.unlink(missing_ok=True)
    try:
        CutSet.from_cuts(cuts).to_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _can_reuse(data_root: Path, output_dir: Path) -> bool:
    summary_path = output_dir / "manifest_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("data_root") != str(data_root)
        or summary.get("generator_sha256") != _source_sha256()
    ):
        return False
    for name, metadata in summary.get("manifests", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            return False
    return True


def prepare_libritts(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    num_jobs: int = 16,
    sample_size: int = 8,
    seed: int = 2026,
    force: bool = False,
) -> dict:
    data_root = Path(data_root).expanduser().absolute()
    output_dir = Path(output_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"LibriTTS root does not exist: {data_root}")
    if os.access(data_root, os.W_OK):
        raise PermissionError(
            f"LibriTTS root must be read-only for this protocol: {data_root}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Manifest output is not writable: {output_dir}")
    if not force and _can_reuse(data_root, output_dir):
        LOG.info("Existing manifests and hashes are valid; nothing to do")
        return json.loads(
            (output_dir / "manifest_summary.json").read_text(encoding="utf-8")
        )

    speakers = _read_speakers(data_root)
    split_audio: dict[str, list[AudioInfo]] = {}
    split_transcripts: dict[str, dict[str, tuple[str, str]]] = {}
    split_summaries = {}
    for split in SPLITS:
        audio, transcripts, summary = _audit_split(
            data_root,
            split,
            num_jobs=num_jobs,
            sample_size=sample_size,
            seed=seed,
        )
        split_audio[split] = audio
        split_transcripts[split] = transcripts
        split_summaries[split] = summary

    id_sets = {
        split: {item.utterance_id for item in split_audio[split]} for split in SPLITS
    }
    speaker_sets = {
        split: {item.speaker_id for item in split_audio[split]} for split in SPLITS
    }
    intersections = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            shared_ids = sorted(id_sets[left] & id_sets[right])
            shared_speakers = sorted(speaker_sets[left] & speaker_sets[right])
            intersections[f"{left}::{right}"] = {
                "utterance_ids": shared_ids,
                "speaker_ids": shared_speakers,
            }
            if shared_ids:
                raise ValueError(f"Utterance IDs overlap between {left} and {right}")

    manifest_paths = []
    for split in SPLITS:
        eval_path = output_dir / f"{split}_eval.jsonl.gz"
        _write_manifest(
            (
                _make_cut(
                    item,
                    split_transcripts[split],
                    speakers,
                    split,
                    include_supervision=True,
                )
                for item in split_audio[split]
            ),
            eval_path,
        )
        manifest_paths.append(eval_path)
        if split in TRAIN_SPLITS:
            codec_path = output_dir / f"{split}_codec.jsonl.gz"
            _write_manifest(
                (
                    _make_cut(
                        item,
                        split_transcripts[split],
                        speakers,
                        split,
                        include_supervision=False,
                    )
                    for item in split_audio[split]
                ),
                codec_path,
            )
            manifest_paths.append(codec_path)

    combined_path = output_dir / "train_codec.jsonl.gz"
    _write_manifest(
        chain.from_iterable(
            CutSet.from_file(output_dir / f"{split}_codec.jsonl.gz")
            for split in TRAIN_SPLITS
        ),
        combined_path,
    )
    manifest_paths.append(combined_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generator_sha256": _source_sha256(),
        "data_root": str(data_root),
        "resolved_data_root": str(data_root.resolve()),
        "output_dir": str(output_dir),
        "sample_rate": SAMPLE_RATE,
        "seed": seed,
        "sample_size_per_split": sample_size,
        "splits": split_summaries,
        "cross_split_intersections": intersections,
        "manifests": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_paths
        },
    }
    _write_json_atomic(summary, output_dir / "manifest_summary.json")
    LOG.info("Wrote %d manifests to %s", len(manifest_paths), output_dir)
    return summary


def _normalize_hydra_style_args(argv: Sequence[str]) -> list[str]:
    aliases = {
        "data.root": "--data-root",
        "output_dir": "--output-dir",
        "num_jobs": "--num-jobs",
        "sample_size": "--sample-size",
        "seed": "--seed",
    }
    normalized = []
    for argument in argv:
        if "=" in argument and not argument.startswith("--"):
            key, value = argument.split("=", maxsplit=1)
            if key in aliases:
                normalized.extend([aliases[key], value])
                continue
        normalized.append(argument)
    return normalized


def _parser() -> argparse.ArgumentParser:
    run_root = os.environ.get("DMEL_RUN_ROOT")
    default_output = (
        str(Path(run_root) / "manifests" / "libritts24k") if run_root else None
    )
    parser = argparse.ArgumentParser(description="Audit and index LibriTTS at 24 kHz")
    parser.add_argument(
        "--data-root", default=os.environ.get("LIBRITTS_ROOT", "/datasets/LibriTTS")
    )
    parser.add_argument("--output-dir", default=default_output, required=not default_output)
    parser.add_argument("--num-jobs", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(
        _normalize_hydra_style_args(sys.argv[1:] if argv is None else argv)
    )
    summary = prepare_libritts(
        args.data_root,
        args.output_dir,
        num_jobs=args.num_jobs,
        sample_size=args.sample_size,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps({"output_dir": summary["output_dir"], "status": "ok"}))


if __name__ == "__main__":
    main()
