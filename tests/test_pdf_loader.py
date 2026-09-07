"""Tests for page-level PDF text extraction."""

from pathlib import Path

import pymupdf
import pytest

from src.ingestion.pdf_loader import load_pdf


def test_load_pdf_returns_page_level_records(tmp_path: Path) -> None:
    """A temporary multi-page PDF retains text, order, and filename metadata."""
    pdf_path = tmp_path / "annual_report.pdf"
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Revenue increased during the year.")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "Net income also increased.")
    document.save(pdf_path)
    document.close()

    records = load_pdf(pdf_path)

    assert len(records) == 2
    assert records[0]["page_number"] == 1
    assert records[1]["page_number"] == 2
    assert "Revenue increased" in records[0]["text"]
    assert "Net income also increased" in records[1]["text"]
    assert records[0]["source_filename"] == "annual_report.pdf"
    assert records[1]["source_filename"] == "annual_report.pdf"


def test_load_pdf_retains_an_empty_page(tmp_path: Path) -> None:
    """An empty PDF page is returned with an empty text value."""
    pdf_path = tmp_path / "report_with_empty_page.pdf"
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Operating income increased.")
    document.new_page()
    document.save(pdf_path)
    document.close()

    records = load_pdf(pdf_path)

    assert len(records) == 2
    assert records[1]["page_number"] == 2
    assert records[1]["text"] == ""


def test_load_pdf_raises_for_a_missing_file(tmp_path: Path) -> None:
    """A path that does not exist raises the documented error."""
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "does-not-exist.pdf")


def test_load_pdf_raises_for_an_invalid_pdf(tmp_path: Path) -> None:
    """A non-PDF file raises the documented validation error."""
    invalid_pdf = tmp_path / "not-a-pdf.pdf"
    invalid_pdf.write_text("This is not a PDF.", encoding="utf-8")

    with pytest.raises(ValueError):
        load_pdf(invalid_pdf)
