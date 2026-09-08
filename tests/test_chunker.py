"""Tests for page-aware, overlapping PDF text chunking."""

from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import PDFPageRecord


def test_short_page_produces_one_chunk_with_preserved_metadata() -> None:
    """A short page becomes one chunk without changing its provenance."""
    pages: list[PDFPageRecord] = [
        {
            "page_number": 3,
            "text": "Revenue increased by 12%.",
            "source_filename": "annual_report.pdf",
        }
    ]

    chunks = chunk_pages(pages)

    assert chunks == [
        {
            "chunk_id": "annual_report_p3_c01",
            "text": "Revenue increased by 12%.",
            "page_number": 3,
            "source_filename": "annual_report.pdf",
        }
    ]


def test_long_text_produces_bounded_overlapping_chunks() -> None:
    """Long text splits into multiple chunks within the requested size."""
    text = "Financial performance remained resilient. " * 40
    chunks = chunk_pages(
        [{"page_number": 1, "text": text, "source_filename": "report.pdf"}],
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 300 for chunk in chunks)
    assert chunks[0]["text"][-50:] == chunks[1]["text"][:50]


def test_empty_pages_produce_no_chunks() -> None:
    """Cleaned empty pages do not create empty chunk records."""
    pages: list[PDFPageRecord] = [
        {"page_number": 2, "text": "", "source_filename": "report.pdf"}
    ]

    assert chunk_pages(pages) == []


def test_chunk_ids_are_deterministic() -> None:
    """Equivalent input always produces the same ordered chunk identifiers."""
    pages: list[PDFPageRecord] = [
        {
            "page_number": 42,
            "text": "Annual report content. " * 30,
            "source_filename": "RIL Annual Report 2024-25.pdf",
        }
    ]

    first_result = chunk_pages(pages, chunk_size=150, chunk_overlap=25)
    second_result = chunk_pages(pages, chunk_size=150, chunk_overlap=25)

    assert first_result == second_result
    assert first_result[0]["chunk_id"] == "RIL_Annual_Report_2024_25_p42_c01"


def test_multiple_pages_are_not_mixed() -> None:
    """Chunks retain text and provenance from only their original page."""
    pages: list[PDFPageRecord] = [
        {
            "page_number": 1,
            "text": "FIRST_PAGE_TOKEN " * 20,
            "source_filename": "report.pdf",
        },
        {
            "page_number": 2,
            "text": "SECOND_PAGE_TOKEN " * 20,
            "source_filename": "report.pdf",
        },
    ]

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)

    first_page_chunks = [chunk for chunk in chunks if chunk["page_number"] == 1]
    second_page_chunks = [chunk for chunk in chunks if chunk["page_number"] == 2]
    assert first_page_chunks
    assert second_page_chunks
    assert all("SECOND_PAGE_TOKEN" not in chunk["text"] for chunk in first_page_chunks)
    assert all("FIRST_PAGE_TOKEN" not in chunk["text"] for chunk in second_page_chunks)


def test_detected_table_data_uses_compact_row_preserving_chunks() -> None:
    table_rows = "\n".join(
        f"Metric {index} | {index * 100} crore"
        for index in range(40)
    )
    chunks = chunk_pages(
        [
            {
                "page_number": 1,
                "text": "Narrative overview.\n\n[TABLE DATA]\n" + table_rows,
                "source_filename": "report.pdf",
            }
        ],
        chunk_size=1000,
        chunk_overlap=150,
    )

    table_chunks = [chunk for chunk in chunks if "Metric 39" in chunk["text"]]
    assert table_chunks
    assert all(len(chunk["text"]) <= 500 for chunk in table_chunks)
    assert all("[TABLE DATA]" not in chunk["text"] for chunk in chunks)
