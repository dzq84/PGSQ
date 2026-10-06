from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_GENERATED_ASSETS = (
    "tables/main_results.tex",
    "tables/ablations.tex",
    "tables/results_summary.tex",
    "figures/method.pdf",
    "figures/rate_quality.pdf",
)
REQUIRED_SOURCE_FILES = ("main.tex", "references.bib", *REQUIRED_GENERATED_ASSETS)
OPTIONAL_SOURCE_SUFFIXES = {
    ".bbx",
    ".bbl",
    ".bib",
    ".bst",
    ".cbx",
    ".cls",
    ".def",
    ".eps",
    ".jpeg",
    ".jpg",
    ".png",
    ".sty",
    ".tex",
}
PLACEHOLDER_MARKERS = (
    "% WORKING DRAFT",
    "Do not submit this layout",
    "Anonymous ICASSP 2027 Submission",
)
SENSITIVE_PATTERNS = {
    "private home path": re.compile(r"/(?:home|Users)/[^\s{}]+"),
    "dataset path": re.compile(r"/(?:datasets|data0|sdb)/[^\s{}]+"),
    "private run variable": re.compile(
        r"\b(?:DMEL_RUN_ROOT|DMEL_CACHE_ROOT|BIGVGAN_CKPT|BIGVGAN_CONFIG)\b"
    ),
    "credential assignment": re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]",
        re.IGNORECASE,
    ),
}
TECHNICAL_HEADINGS = (
    "Abstract",
    "Index Terms",
    "Introduction",
    "Method",
    "Experimental Setup",
    "Results and Analysis",
    "Conclusion and Limitations",
)


@dataclass(frozen=True)
class PdfInspection:
    pages: int
    page_size: str
    title: str
    author: str
    subject: str
    page_texts: tuple[str, ...]
    creator: str = ""
    producer: str = ""
    creation_date: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _scan_sensitive_text(text: str, *, label: str) -> None:
    for description, pattern in SENSITIVE_PATTERNS.items():
        match = pattern.search(text)
        if match is not None:
            raise ValueError(f"{label} contains {description}: {match.group(0)!r}")


def _validate_asset_manifest(
    paper_directory: Path,
    *,
    diagnostic: bool,
    test_gate: Path | None,
    protocol: Path | None,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = paper_directory / "artifacts/result_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError("Paper result manifest must use schema version 2")
    expected_status = "PAPER_ASSETS_DIAGNOSTIC" if diagnostic else "PAPER_ASSETS_FROZEN"
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"Expected {expected_status}, found {manifest.get('status')!r}"
        )
    if diagnostic:
        if manifest.get("test_gate") is not None:
            raise ValueError("Diagnostic paper assets must not bind a test gate")
    else:
        gate_binding = manifest.get("test_gate")
        if manifest.get("allow_incomplete") is not False:
            raise ValueError("Formal paper assets cannot be incomplete")
        if (
            not isinstance(gate_binding, dict)
            or gate_binding.get("schema_version") != 3
            or gate_binding.get("status") != "TEST_GATE_FROZEN"
            or not re.fullmatch(r"[0-9a-f]{64}", str(gate_binding.get("sha256", "")))
        ):
            raise ValueError("Formal paper assets lack a schema-v3 frozen test gate")
        if test_gate is None or not test_gate.is_file():
            raise ValueError("Formal paper package requires the frozen test-gate file")
        gate = _load_json(test_gate)
        if gate.get("schema_version") != 3 or gate.get("status") != "TEST_GATE_FROZEN":
            raise ValueError("Provided test gate is not schema-v3 TEST_GATE_FROZEN")
        if sha256_file(test_gate) != gate_binding["sha256"]:
            raise ValueError("Provided test-gate hash differs from the paper assets")
        protocol_binding = manifest.get("protocol")
        if protocol_binding is not None:
            if (
                protocol is None
                or not protocol.is_file()
                or not isinstance(protocol_binding, dict)
            ):
                raise ValueError("Formal paper package lacks the frozen protocol")
            if sha256_file(protocol) != protocol_binding.get("sha256"):
                raise ValueError("Current protocol hash differs from the paper assets")

    generated = manifest.get("generated")
    if not isinstance(generated, dict):
        raise ValueError("Paper result manifest lacks generated-file hashes")
    for relative in REQUIRED_GENERATED_ASSETS:
        expected_hash = generated.get(relative)
        if not isinstance(expected_hash, str):
            raise ValueError(f"Paper result manifest lacks {relative}")
        path = paper_directory / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Paper source asset is missing or a symlink: {relative}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"Paper source asset hash mismatch for {relative}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
    if not diagnostic:
        result_prose = (paper_directory / "tables/results_summary.tex").read_text(
            encoding="utf-8"
        )
        if "requires a schema-v3 test gate" in result_prose:
            raise ValueError("Formal result prose is still the diagnostic placeholder")
    return manifest_path, manifest


def _source_files(paper_directory: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = paper_directory / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Paper source is missing or a symlink: {relative}")
        selected[relative] = path
    for path in sorted(paper_directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(paper_directory).as_posix()
        if relative.startswith("artifacts/"):
            continue
        if path.suffix.lower() in OPTIONAL_SOURCE_SUFFIXES:
            selected.setdefault(relative, path)
    return [selected[key] for key in sorted(selected)]


def _validate_sources(
    paper_directory: Path, source_files: list[Path], *, diagnostic: bool
) -> None:
    main_text = (paper_directory / "main.tex").read_text(encoding="utf-8")
    if not diagnostic:
        present = [marker for marker in PLACEHOLDER_MARKERS if marker in main_text]
        if re.search(
            r"\b(?:anonymous|working draft|do not submit)\b",
            main_text,
            flags=re.IGNORECASE,
        ):
            present.append("draft/anonymous marker")
        if present:
            raise ValueError(
                f"Formal paper still contains placeholders: {sorted(set(present))}"
            )
    for path in source_files:
        if path.suffix.lower() not in {
            ".bbx",
            ".bbl",
            ".bib",
            ".bst",
            ".cbx",
            ".cls",
            ".def",
            ".sty",
            ".tex",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Paper text source is not UTF-8: {path}") from error
        _scan_sensitive_text(text, label=path.relative_to(paper_directory).as_posix())


def _parse_pdfinfo(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            result[key.strip()] = value.strip()
    return result


def inspect_pdf(pdf: Path) -> PdfInspection:
    try:
        metadata_output = subprocess.check_output(
            ["pdfinfo", str(pdf)], text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "pdfinfo is required to inspect the submission PDF"
        ) from error
    metadata = _parse_pdfinfo(metadata_output)
    try:
        pages = int(metadata["Pages"])
    except (KeyError, ValueError) as error:
        raise ValueError("pdfinfo did not report a valid page count") from error
    page_texts = []
    for page in range(1, pages + 1):
        try:
            page_texts.append(
                subprocess.check_output(
                    [
                        "pdftotext",
                        "-f",
                        str(page),
                        "-l",
                        str(page),
                        "-layout",
                        str(pdf),
                        "-",
                    ],
                    text=True,
                    stderr=subprocess.STDOUT,
                )
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "pdftotext is required to inspect the submission PDF"
            ) from error
    return PdfInspection(
        pages=pages,
        page_size=metadata.get("Page size", ""),
        title=metadata.get("Title", ""),
        author=metadata.get("Author", ""),
        subject=metadata.get("Subject", ""),
        page_texts=tuple(page_texts),
        creator=metadata.get("Creator", ""),
        producer=metadata.get("Producer", ""),
        creation_date=metadata.get("CreationDate", ""),
    )


def _validate_pdf(inspection: PdfInspection, *, diagnostic: bool) -> None:
    if inspection.pages != 5 or len(inspection.page_texts) != 5:
        raise ValueError(
            f"Paper must be four technical pages plus one references page; "
            f"found {inspection.pages} pages"
        )
    technical = "\n".join(inspection.page_texts[:4])
    references = inspection.page_texts[4].lstrip()
    if "Conclusion and Limitations" not in technical:
        raise ValueError("Conclusion is not present in the first four pages")
    if not references.startswith("References"):
        raise ValueError("Page 5 must begin with References")
    leaked = [
        heading
        for heading in TECHNICAL_HEADINGS
        if re.search(rf"(?m)^\s*{re.escape(heading)}\s*$", references)
    ]
    if leaked:
        raise ValueError(f"Technical content leaked onto references page: {leaked}")

    all_text = "\n".join(inspection.page_texts)
    _scan_sensitive_text(all_text, label="compiled PDF text")
    metadata_text = "\n".join(
        (
            inspection.title,
            inspection.author,
            inspection.subject,
            inspection.creator,
            inspection.producer,
            inspection.creation_date,
        )
    )
    _scan_sensitive_text(metadata_text, label="compiled PDF metadata")
    if not diagnostic and re.search(
        r"\b(?:anonymous|working draft|do not submit)\b",
        all_text + "\n" + metadata_text,
        flags=re.IGNORECASE,
    ):
        raise ValueError("Formal PDF still contains a draft or anonymous placeholder")


def _git_state(repository_root: Path, *, diagnostic: bool) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=repository_root, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot inspect git repository at {repository_root}"
        ) from error
    if not diagnostic and status:
        raise ValueError("Formal paper package requires a clean git worktree")
    return {"commit": commit, "dirty": bool(status)}


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    import io

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"Refusing to replace different artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def build_submission_package(
    *,
    paper_directory: str | Path,
    pdf: str | Path,
    archive: str | Path,
    output_manifest: str | Path,
    repository_root: str | Path,
    test_gate: str | Path | None = None,
    diagnostic: bool = False,
) -> dict[str, Any]:
    paper_directory = Path(paper_directory).expanduser().resolve()
    pdf = Path(pdf).expanduser().resolve()
    archive = Path(archive).expanduser().resolve()
    output_manifest = Path(output_manifest).expanduser().resolve()
    repository_root = Path(repository_root).expanduser().resolve()
    test_gate = None if test_gate is None else Path(test_gate).expanduser().resolve()
    if not paper_directory.is_dir():
        raise NotADirectoryError(paper_directory)
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    if archive.suffix.lower() != ".zip":
        raise ValueError("Paper source archive must have a .zip suffix")
    if output_manifest.suffix.lower() != ".json":
        raise ValueError("Paper package manifest must have a .json suffix")
    for output in (archive, output_manifest):
        try:
            output.relative_to(Path("/datasets/LibriTTS"))
        except ValueError:
            pass
        else:
            raise ValueError(
                "Paper artifacts cannot be written under /datasets/LibriTTS"
            )

    protocol_path = repository_root / "docs/EXPERIMENT_PROTOCOL.md"
    result_manifest_path, result_manifest = _validate_asset_manifest(
        paper_directory,
        diagnostic=diagnostic,
        test_gate=test_gate,
        protocol=protocol_path if protocol_path.is_file() else None,
    )
    source_files = _source_files(paper_directory)
    _validate_sources(paper_directory, source_files, diagnostic=diagnostic)
    inspection = inspect_pdf(pdf)
    _validate_pdf(inspection, diagnostic=diagnostic)
    git_state = _git_state(repository_root, diagnostic=diagnostic)

    source_hashes = {
        path.relative_to(paper_directory).as_posix(): sha256_file(path)
        for path in source_files
    }
    provenance = {
        "schema_version": 1,
        "status": (
            "PAPER_SUBMISSION_DIAGNOSTIC" if diagnostic else "PAPER_SUBMISSION_FROZEN"
        ),
        "git": git_state,
        "compiled_pdf_sha256": sha256_file(pdf),
        "paper_result_manifest": {
            "sha256": sha256_file(result_manifest_path),
            "status": result_manifest["status"],
            "test_gate": result_manifest.get("test_gate"),
        },
        "sources": source_hashes,
    }
    archive_members = {
        relative: (paper_directory / relative).read_bytes()
        for relative in source_hashes
    }
    archive_members["submission_provenance.json"] = _canonical_json(provenance)
    archive_content = _zip_bytes(archive_members)
    archive_hash = hashlib.sha256(archive_content).hexdigest()
    manifest = {
        **provenance,
        "archive": {
            "name": archive.name,
            "sha256": archive_hash,
            "size_bytes": len(archive_content),
            "members": sorted(archive_members),
        },
        "compiled_pdf": {
            "name": pdf.name,
            "sha256": provenance["compiled_pdf_sha256"],
            "size_bytes": pdf.stat().st_size,
            "inspection": {
                key: value
                for key, value in asdict(inspection).items()
                if key != "page_texts"
            },
        },
    }
    _write_immutable(archive, archive_content)
    _write_immutable(output_manifest, _canonical_json(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and audit a deterministic paper submission source archive"
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--test-gate", type=Path)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Accept diagnostic assets and draft placeholders; output remains non-formal",
    )
    args = parser.parse_args()
    result = build_submission_package(
        paper_directory=args.paper_dir,
        pdf=args.pdf,
        archive=args.archive,
        output_manifest=args.manifest,
        repository_root=args.repository_root,
        test_gate=args.test_gate,
        diagnostic=args.diagnostic,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
