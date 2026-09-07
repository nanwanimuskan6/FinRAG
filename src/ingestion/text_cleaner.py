"""Conservative normalization utilities for page-level PDF text."""

import re
from collections.abc import Sequence

from src.ingestion.pdf_loader import PDFPageRecord


_REPEATED_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]{2,}")
_EXCESSIVE_BLANK_LINES = re.compile(r"\n(?:[ \t]*\n){2,}")


def clean_page_text(text: str) -> str:
    """Conservatively normalize whitespace in text extracted from one page.

    The function normalizes line endings, collapses repeated spaces or tabs
    within a line, limits consecutive blank lines to one, and trims whitespace
    at the page boundaries. It does not change line order or text content other
    than whitespace normalization.

    Args:
        text: Raw text extracted from a single PDF page.

    Returns:
        The whitespace-normalized page text.
    """
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = _REPEATED_HORIZONTAL_WHITESPACE.sub(" ", normalized_text)
    normalized_text = _EXCESSIVE_BLANK_LINES.sub("\n\n", normalized_text)
    return normalized_text.strip()


def clean_pages(pages: Sequence[PDFPageRecord]) -> list[PDFPageRecord]:
    """Return page records with conservatively cleaned text.

    Args:
        pages: Page records produced by :func:`src.ingestion.pdf_loader.load_pdf`.

    Returns:
        New page records preserving page number and source filename, with only
        their text values whitespace-normalized.
    """
    return [
        PDFPageRecord(
            page_number=page["page_number"],
            text=clean_page_text(page["text"]),
            source_filename=page["source_filename"],
        )
        for page in pages
    ]
