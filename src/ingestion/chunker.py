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

        chunk_number = 1

        # Table detector output is intentionally separated from narrative text.
        # Financial table rows often contain the exact metric/value pair, so
        # smaller row-preserving chunks are much easier to retrieve than a
        # mixed multi-column page extraction.
        narrative, marker, table_data = text.partition("\n\n[TABLE DATA]\n")
        sections = [(narrative, chunk_size, chunk_overlap)]
        if marker and table_data.strip():
            sections.append(
                (
                    table_data,
                    min(chunk_size, 500),
                    min(chunk_overlap, 100),
                )
            )

        for section_text, section_size, section_overlap in sections:
            start = 0
            while start < len(section_text):
                end = min(start + section_size, len(section_text))

                if end < len(section_text):
                    newline = section_text.rfind("\n", start + section_size // 2, end)
                    if newline != -1:
                        end = newline + 1
                    else:
                        space = section_text.rfind(" ", start + section_size // 2, end)
                        if space != -1:
                            end = space + 1

                # Preserve the source characters exactly. Stripping individual
                # chunks would remove a boundary space and make the promised
                # character overlap smaller than ``chunk_overlap``.
                chunk_text = section_text[start:end]

                if chunk_text.strip():
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

                if end >= len(section_text):
                    break

                start = max(end - section_overlap, start + 1)

    return chunks
