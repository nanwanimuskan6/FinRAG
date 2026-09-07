"""Command-line runner for inspecting text extracted from a PDF."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from src.ingestion.pdf_loader import load_pdf


def main(arguments: Sequence[str] | None = None) -> int:
    """Load a PDF and print a concise summary of its first page.

    Args:
        arguments: Optional command-line arguments, excluding the program name.

    Returns:
        Zero when ingestion succeeds, otherwise one for a file or PDF error.
    """
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and display its first-page summary."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to ingest.")
    args = parser.parse_args(arguments)

    try:
        pages = load_pdf(args.pdf_path)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Total pages: {len(pages)}")

    if not pages:
        print("The PDF contains no pages.")
        return 0

    first_page = pages[0]
    print(f"Source filename: {first_page['source_filename']}")
    print(f"Page number: {first_page['page_number']}")
    print("First page text (first 500 characters):")
    print(first_page["text"][:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
