"""PDF text and table extraction for FinRAG."""

from pathlib import Path
from typing import TypedDict

import fitz


_FINANCIAL_TABLE_SIGNALS = (
    "financial highlights",
    "financial performance",
    "consolidated financial statements",
    "standalone financial statements",
    "statement of profit",
    "balance sheet",
    "cash flows",
    "changes in equity",
    "segment information",
    "key indicators",
)


class PDFPageRecord(TypedDict):
    page_number: int
    text: str
    source_filename: str


def has_financial_table_signal(text: str) -> bool:
    """Return whether a page is likely to contain a useful financial table."""
    normalized_text = text.lower()
    return any(signal in normalized_text for signal in _FINANCIAL_TABLE_SIGNALS)


def load_pdf(
    pdf_path: str | Path,
    *,
    include_detected_tables: bool = False,
) -> list[PDFPageRecord]:
    """Extract page text, optionally appending detected table data.

    The default is clean layout-sorted text. PyMuPDF's generic table detector
    can emit many decorative-layout fragments in annual reports; those inflated
    the retrieval index without improving benchmark recall. Table extraction is
    therefore opt-in until a report-specific table parser is available.
    """

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    records: list[PDFPageRecord] = []

    try:
        with fitz.open(path) as document:

            for page_number, page in enumerate(document, start=1):

                # Normal text extraction
                text = page.get_text("text", sort=True)

                # Table extraction
                table_texts: list[str] = []

                try:
                    # Table detection is intentionally restricted to financial
                    # pages. Running it on every annual-report page creates a
                    # large number of decorative-layout fragments that drown
                    # out actual statement rows in keyword retrieval.
                    tables = (
                        page.find_tables()
                        if include_detected_tables and has_financial_table_signal(text)
                        else None
                    )

                    for table in tables.tables if tables is not None else []:
                        extracted = table.extract()

                        if extracted:
                            rows = []

                            for row in extracted:
                                cleaned_row = [
                                    str(cell).strip() if cell is not None else ""
                                    for cell in row
                            ]
                                rows.append(" | ".join(cleaned_row))

                            table_texts.append(
                                "\n".join(rows)
                            )

                except Exception:
                    # If table detection fails on a particular page,
                    # retain normal text extraction.
                    pass

                # Combine normal text + structured tables
                combined_parts = []

                if text.strip():
                    combined_parts.append(text.strip())

                if table_texts:
                    combined_parts.append(
                        "\n\n[TABLE DATA]\n"
                        + "\n\n".join(table_texts)
                    )

                combined_text = "\n\n".join(combined_parts)

                records.append(
                    PDFPageRecord(
                        page_number=page_number,
                        text=combined_text,
                        source_filename=path.name,
                    )
                )

        return records

    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError(f"Invalid PDF file: {path}") from error
