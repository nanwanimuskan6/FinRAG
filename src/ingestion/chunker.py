"""Page-aware text and table chunking utilities."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from src.ingestion.pdf_loader import PDFPageRecord


class PDFChunkRecord(TypedDict):
    chunk_id: str
    text: str
    page_number: int
    source_filename: str


def _source_stem(source_filename: str) -> str:
    stem = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        Path(source_filename).stem,
    ).strip("_")

    return stem or "document"


def chunk_pages(
    pages: Sequence[PDFPageRecord],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[PDFChunkRecord]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be at least zero and smaller than chunk_size"
        )

    chunks: list[PDFChunkRecord] = []

    for page in pages:

        text = page["text"]

        if not text.strip():
            continue

        source_stem = _source_stem(page["source_filename"])

        # Split text into chunks
        start = 0
        chunk_number = 1

        while start < len(text):

            end = min(start + chunk_size, len(text))

            if end < len(text):

                newline = text.rfind("\n", start + chunk_size // 2, end)

                if newline != -1:
                    end = newline + 1
                else:

                    space = text.rfind(" ", start + chunk_size // 2, end)

                    if space != -1:
                        end = space + 1

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    PDFChunkRecord(
                        chunk_id=(
                            f"{source_stem}_"
                            f"p{page['page_number']}_"
                            f"c{chunk_number:02d}"
                        ),
                        text=chunk_text,
                        page_number=page["page_number"],
                        source_filename=page["source_filename"],
                    )
                )

                chunk_number += 1

            if end >= len(text):
                break

            start = max(
                end - chunk_overlap,
                start + 1,
            )

    return chunks
