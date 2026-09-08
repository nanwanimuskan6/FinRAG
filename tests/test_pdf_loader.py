"""Tests for page-level PDF text extraction."""

from pathlib import Path

import pymupdf
import pytest

from src.ingestion.pdf_loader import has_financial_table_signal, load_pdf


def test_financial_table_signal_targets_statement_and_highlight_pages() -> None:
    assert has_financial_table_signal("Consolidated Financial Statements")
    assert has_financial_table_signal("10-Year Financial Highlights")
    assert not has_financial_table_signal("Corporate social responsibility overview")


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


def test_detected_tables_are_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default loading avoids expensive/noisy generic table detection."""
    pdf_path = tmp_path / "report.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Financial Highlights")
    document.save(pdf_path)
    document.close()

    calls = 0

    def fail_if_called(_: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("table detection should not run by default")

    monkeypatch.setattr(pymupdf.Page, "find_tables", fail_if_called)

    records = load_pdf(pdf_path)

    assert "Financial Highlights" in records[0]["text"]
    assert calls == 0


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
