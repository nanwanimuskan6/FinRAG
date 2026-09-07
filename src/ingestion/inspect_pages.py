"""Command-line utility for inspecting selected PDF extraction results."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from src.ingestion.pdf_loader import load_pdf


PAGES_TO_INSPECT = (2, 10, 50, 100)


def main(arguments: Sequence[str] | None = None) -> int:
    """Print text previews for selected pages in a PDF.

    Args:
        arguments: Optional command-line arguments, excluding the program name.

    Returns:
        Zero when the PDF is inspected successfully, otherwise one.
    """
    parser = argparse.ArgumentParser(
        description="Inspect extracted text from selected pages of a PDF."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to inspect.")
    args = parser.parse_args(arguments)

    try:
        pages = load_pdf(args.pdf_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    total_pages = len(pages)
    for page_number in PAGES_TO_INSPECT:
        if page_number > total_pages:
            print(f"Skipping page {page_number}: PDF has only {total_pages} pages.")
            continue

        page = pages[page_number - 1]
        print(f"Page number: {page['page_number']}")
        print(f"Source filename: {page['source_filename']}")
        print("Extracted text (first 1000 characters):")
        print(page["text"][:1000])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
