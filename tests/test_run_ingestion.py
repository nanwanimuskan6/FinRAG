"""Tests for the manual PDF ingestion runner."""

from pathlib import Path

import pymupdf
import pytest

from src.ingestion.run_ingestion import main


def test_runner_prints_a_first_page_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner prints metadata and a 500-character text preview."""
    pdf_path = tmp_path / "sample_report.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Revenue grew by 12 percent.")
    document.new_page()
    document.save(pdf_path)
    document.close()

    exit_code = main([str(pdf_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Total pages: 2" in captured.out
    assert "Source filename: sample_report.pdf" in captured.out
    assert "Page number: 1" in captured.out
    assert "Revenue grew by 12 percent." in captured.out
    assert captured.err == ""


def test_runner_reports_a_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner reports a clear error when the PDF does not exist."""
    exit_code = main([str(tmp_path / "missing.pdf")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: PDF file not found:" in captured.err
    assert captured.out == ""


def test_runner_reports_an_invalid_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner reports a clear error for a non-PDF file."""
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("not a PDF", encoding="utf-8")

    exit_code = main([str(invalid_pdf)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: Invalid PDF file:" in captured.err
    assert captured.out == ""
