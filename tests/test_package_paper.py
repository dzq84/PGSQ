from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from dmel_codec.tools.package_paper import (
    PdfInspection,
    _validate_pdf,
    build_submission_package,
)


GENERATED = (
    "tables/main_results.tex",
    "tables/ablations.tex",
    "tables/results_summary.tex",
    "figures/method.pdf",
    "figures/rate_quality.pdf",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_tree(root: Path, *, formal: bool) -> Path:
    paper = root / "paper"
    (paper / "tables").mkdir(parents=True)
    (paper / "figures").mkdir()
    (paper / "artifacts").mkdir()
    main = (
        r"\documentclass{article}"
        "\n"
        r"\author{Real Author}"
        "\n"
        r"\begin{document}Paper\end{document}"
        "\n"
    )
    if not formal:
        main = "% WORKING DRAFT\n" + main
    (paper / "main.tex").write_text(main, encoding="utf-8")
    (paper / "references.bib").write_text("@article{x, title={X}}\n", encoding="utf-8")
    (paper / "conference.sty").write_text("% local template\n", encoding="utf-8")
    (paper / "macros.tex").write_text("% local macros\n", encoding="utf-8")
    for index, relative in enumerate(GENERATED):
        path = paper / relative
        if path.suffix == ".tex":
            content = f"% generated {index}\n"
            if relative.endswith("results_summary.tex") and formal:
                content = "% Generated from frozen test runs.\n"
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(f"pdf-{index}".encode())
    generated = {relative: _sha256(paper / relative) for relative in GENERATED}
    protocol = root / "docs/EXPERIMENT_PROTOCOL.md"
    protocol.parent.mkdir()
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    gate_path = root / "TEST_GATE_FROZEN"
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "TEST_GATE_FROZEN",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "status": "PAPER_ASSETS_FROZEN" if formal else "PAPER_ASSETS_DIAGNOSTIC",
        "allow_incomplete": not formal,
        "test_gate": (
            {
                "schema_version": 3,
                "status": "TEST_GATE_FROZEN",
                "sha256": _sha256(gate_path),
            }
            if formal
            else None
        ),
        "protocol": {
            "name": protocol.name,
            "sha256": _sha256(protocol),
        },
        "generated": generated,
    }
    (paper / "artifacts/result_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return paper


def _inspection() -> PdfInspection:
    return PdfInspection(
        pages=5,
        page_size="A4",
        title="Paper",
        author="Real Author",
        subject="",
        page_texts=(
            "Abstract\n",
            "Method\n",
            "Results and Analysis\n",
            "Conclusion and Limitations\n",
            "References\n[1] Citation\n",
        ),
    )


def _clean_git(command, **_) -> str:
    if command[1:] == ["rev-parse", "HEAD"]:
        return "deadbeef\n"
    if command[1:] == ["status", "--short"]:
        return ""
    raise AssertionError(command)


def test_diagnostic_package_is_deterministic_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _paper_tree(tmp_path, formal=False)
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    monkeypatch.setattr(
        "dmel_codec.tools.package_paper.inspect_pdf", lambda _: _inspection()
    )
    monkeypatch.setattr(
        "dmel_codec.tools.package_paper.subprocess.check_output", _clean_git
    )

    first = build_submission_package(
        paper_directory=paper,
        pdf=pdf,
        archive=tmp_path / "first.zip",
        output_manifest=tmp_path / "first.json",
        repository_root=tmp_path,
        diagnostic=True,
    )
    second = build_submission_package(
        paper_directory=paper,
        pdf=pdf,
        archive=tmp_path / "second.zip",
        output_manifest=tmp_path / "second.json",
        repository_root=tmp_path,
        diagnostic=True,
    )
    assert first["status"] == "PAPER_SUBMISSION_DIAGNOSTIC"
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert _sha256(tmp_path / "first.zip") == _sha256(tmp_path / "second.zip")
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        assert archive.namelist() == sorted(
            [
                "conference.sty",
                "macros.tex",
                "main.tex",
                "references.bib",
                *GENERATED,
                "submission_provenance.json",
            ]
        )
        provenance = json.loads(archive.read("submission_provenance.json"))
    assert provenance["paper_result_manifest"]["status"] == ("PAPER_ASSETS_DIAGNOSTIC")
    assert provenance["compiled_pdf_sha256"] == _sha256(pdf)


def test_formal_package_rejects_diagnostic_assets(tmp_path: Path) -> None:
    paper = _paper_tree(tmp_path, formal=False)
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    with pytest.raises(ValueError, match="PAPER_ASSETS_FROZEN"):
        build_submission_package(
            paper_directory=paper,
            pdf=pdf,
            archive=tmp_path / "paper.zip",
            output_manifest=tmp_path / "paper.json",
            repository_root=tmp_path,
        )


def test_formal_package_rejects_draft_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _paper_tree(tmp_path, formal=True)
    with (paper / "main.tex").open("a", encoding="utf-8") as stream:
        stream.write(r"\author{Anonymous}" + "\n")
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    monkeypatch.setattr(
        "dmel_codec.tools.package_paper.inspect_pdf", lambda _: _inspection()
    )
    with pytest.raises(ValueError, match="placeholders"):
        build_submission_package(
            paper_directory=paper,
            pdf=pdf,
            archive=tmp_path / "paper.zip",
            output_manifest=tmp_path / "paper.json",
            repository_root=tmp_path,
            test_gate=tmp_path / "TEST_GATE_FROZEN",
        )


@pytest.mark.parametrize(
    ("drift_path", "message"),
    (
        ("TEST_GATE_FROZEN", "test-gate hash differs"),
        ("docs/EXPERIMENT_PROTOCOL.md", "protocol hash differs"),
    ),
)
def test_formal_package_rejects_gate_or_protocol_drift(
    tmp_path: Path,
    drift_path: str,
    message: str,
) -> None:
    paper = _paper_tree(tmp_path, formal=True)
    drift = tmp_path / drift_path
    if drift_path == "TEST_GATE_FROZEN":
        with drift.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        drift.write_text("edited after paper assets\n", encoding="utf-8")
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    with pytest.raises(ValueError, match=message):
        build_submission_package(
            paper_directory=paper,
            pdf=pdf,
            archive=tmp_path / "paper.zip",
            output_manifest=tmp_path / "paper.json",
            repository_root=tmp_path,
            test_gate=tmp_path / "TEST_GATE_FROZEN",
        )


def test_package_rejects_edited_generated_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _paper_tree(tmp_path, formal=False)
    (paper / "tables/main_results.tex").write_text("edited\n", encoding="utf-8")
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    monkeypatch.setattr(
        "dmel_codec.tools.package_paper.inspect_pdf", lambda _: _inspection()
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        build_submission_package(
            paper_directory=paper,
            pdf=pdf,
            archive=tmp_path / "paper.zip",
            output_manifest=tmp_path / "paper.json",
            repository_root=tmp_path,
            diagnostic=True,
        )


def test_package_rejects_private_paths_in_source(tmp_path: Path) -> None:
    paper = _paper_tree(tmp_path, formal=False)
    with (paper / "main.tex").open("a", encoding="utf-8") as stream:
        stream.write("% /home/private-user/secret.ckpt\n")
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    with pytest.raises(ValueError, match="private home path"):
        build_submission_package(
            paper_directory=paper,
            pdf=pdf,
            archive=tmp_path / "paper.zip",
            output_manifest=tmp_path / "paper.json",
            repository_root=tmp_path,
            diagnostic=True,
        )


def test_pdf_preflight_rejects_technical_content_on_reference_page() -> None:
    inspection = PdfInspection(
        **{
            **_inspection().__dict__,
            "page_texts": (
                "Abstract\n",
                "Method\n",
                "Results and Analysis\n",
                "Conclusion and Limitations\n",
                "References\nConclusion and Limitations\n",
            ),
        }
    )
    with pytest.raises(ValueError, match="Technical content leaked"):
        _validate_pdf(inspection, diagnostic=False)


def test_formal_wrapper_has_no_diagnostic_bypass() -> None:
    repository = Path(__file__).resolve().parents[1]
    wrapper = (repository / "scripts/package_paper.sh").read_text(encoding="utf-8")
    assert "dmel_codec.tools.package_paper" in wrapper
    assert "--diagnostic" not in wrapper
    assert "COMPILED_PDF OUTPUT_ZIP" in wrapper
    assert "TEST_GATE" in wrapper


def test_formal_package_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = _paper_tree(tmp_path, formal=True)
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"compiled-pdf")
    monkeypatch.setattr(
        "dmel_codec.tools.package_paper.inspect_pdf", lambda _: _inspection()
    )
    monkeypatch.setattr(
        "dmel_codec.tools.package_paper.subprocess.check_output", _clean_git
    )
    arguments = {
        "paper_directory": paper,
        "pdf": pdf,
        "archive": tmp_path / "paper.zip",
        "output_manifest": tmp_path / "paper.json",
        "repository_root": tmp_path,
        "test_gate": tmp_path / "TEST_GATE_FROZEN",
    }
    first = build_submission_package(**arguments)
    second = build_submission_package(**arguments)
    assert first == second
    (paper / "main.tex").write_text("changed\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        build_submission_package(**arguments)
