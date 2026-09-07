"""Command-line utility for inspecting cleaned PDF chunks."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from src.ingestion.chunker import PDFChunkRecord, chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_cleaner import clean_pages


def _print_chunk(chunk: PDFChunkRecord, heading: str) -> None:
    """Print a chunk and its provenance in a readable format."""
    print(heading)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page number: {chunk['page_number']}")
    print(f"Source filename: {chunk['source_filename']}")
    print("Text:")
    print(chunk["text"])
    print()


def main(arguments: Sequence[str] | None = None) -> int:
    """Load, clean, and chunk a PDF before printing chunking diagnostics.

    Args:
        arguments: Optional command-line arguments, excluding the program name.

    Returns:
        Zero when inspection succeeds, otherwise one for a file or PDF error.
    """
    parser = argparse.ArgumentParser(
        description="Inspect page-level cleaning and chunking results for a PDF."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to inspect.")
    args = parser.parse_args(arguments)

    try:
        pages = load_pdf(args.pdf_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    cleaned_pages = clean_pages(pages)
    chunks = chunk_pages(cleaned_pages)
    source_filename = pages[0]["source_filename"] if pages else args.pdf_path.name
    non_empty_pages = sum(bool(page["text"]) for page in cleaned_pages)
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]

    print(f"Source filename: {source_filename}")
    print(f"Total PDF pages: {len(pages)}")
    print(f"Total non-empty pages: {non_empty_pages}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Minimum chunk length: {min(chunk_lengths, default=0)}")
    print(f"Maximum chunk length: {max(chunk_lengths, default=0)}")
    average_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"Average chunk length: {average_length:.2f}")
    print()

    for chunk_number, chunk in enumerate(chunks[:3], start=1):
        _print_chunk(chunk, f"Chunk {chunk_number}:")

    page_50_chunk = next((chunk for chunk in chunks if chunk["page_number"] == 50), None)
    if page_50_chunk is None:
        print("No non-empty chunk found for page 50.")
    else:
        _print_chunk(page_50_chunk, "First chunk from page 50:")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
