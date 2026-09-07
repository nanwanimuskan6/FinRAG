"""PDF text and table extraction for FinRAG."""

from pathlib import Path
from typing import TypedDict

import fitz


class PDFPageRecord(TypedDict):
    page_number: int
    text: str
    source_filename: str


def load_pdf(pdf_path: str | Path) -> list[PDFPageRecord]:
    """Extract normal text AND detected tables from every PDF page."""

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
                    tables = page.find_tables()

                    for table in tables.tables:
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
    