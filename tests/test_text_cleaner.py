"""Tests for conservative PDF page-text cleaning."""

from src.ingestion.pdf_loader import PDFPageRecord
from src.ingestion.text_cleaner import clean_page_text, clean_pages


def test_clean_page_text_normalizes_repeated_whitespace() -> None:
    """Repeated spaces and tabs are reduced without changing words."""
    text = "  Revenue   increased\t\tby  12%  "

    assert clean_page_text(text) == "Revenue increased by 12%"


def test_clean_page_text_normalizes_line_endings_and_blank_lines() -> None:
    """Different line endings and excessive blank lines are standardized."""
    text = "First line\r\n\r\n\r\nSecond line\rThird line\n\n\n\nFourth line"

    assert clean_page_text(text) == "First line\n\nSecond line\nThird line\n\nFourth line"


def test_clean_page_text_preserves_financial_numbers() -> None:
    """Whitespace cleanup does not alter financial values."""
    text = "Revenue  1,234.50\nEPS  12.75\nLoss  (45.00)"

    cleaned_text = clean_page_text(text)

    assert "1,234.50" in cleaned_text
    assert "12.75" in cleaned_text
    assert "(45.00)" in cleaned_text


def test_clean_pages_preserves_page_metadata() -> None:
    """Cleaning text leaves page number and source filename unchanged."""
    pages: list[PDFPageRecord] = [
        {
            "page_number": 7,
            "text": "  Net   profit  ",
            "source_filename": "annual_report.pdf",
        }
    ]

    cleaned_pages = clean_pages(pages)

    assert cleaned_pages == [
        {
            "page_number": 7,
            "text": "Net profit",
            "source_filename": "annual_report.pdf",
        }
    ]
    assert pages[0]["text"] == "  Net   profit  "


def test_clean_page_text_preserves_empty_text() -> None:
    """An empty page remains empty after cleaning."""
    assert clean_page_text("") == ""
