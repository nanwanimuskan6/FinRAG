"""Tests for persistent ChromaDB storage with synthetic embeddings."""

from pathlib import Path

import numpy as np
import pytest

from src.ingestion.chunker import PDFChunkRecord
from src.retrieval.vector_store import ChromaVectorStore


def _sample_chunks() -> list[PDFChunkRecord]:
    """Return small chunks with distinct provenance for testing."""
    return [
        {
            "chunk_id": "report_p1_c01",
            "text": "Revenue increased during the year.",
            "page_number": 1,
            "source_filename": "report.pdf",
        },
        {
            "chunk_id": "report_p2_c01",
            "text": "Debt declined during the year.",
            "page_number": 2,
            "source_filename": "report.pdf",
        },
    ]


def _sample_embeddings() -> np.ndarray:
    """Return normalized synthetic vectors for testing."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def test_collection_creation_add_count_query_and_metadata(tmp_path: Path) -> None:
    """A persistent collection stores and retrieves aligned chunk metadata."""
    store = ChromaVectorStore(tmp_path / "chroma_db", "test_report_chunks")
    chunks = _sample_chunks()
    embeddings = _sample_embeddings()

    assert store.count() == 0
    store.add_chunks(chunks, embeddings)
    assert store.count() == 2

    results = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)

    assert results == [
        {
            "chunk_id": "report_p1_c01",
            "text": "Revenue increased during the year.",
            "page_number": 1,
            "source_filename": "report.pdf",
            "distance": pytest.approx(0.0),
        }
    ]


def test_duplicate_ids_are_rejected_clearly(tmp_path: Path) -> None:
    """Duplicate chunk IDs in a batch or an existing collection are rejected."""
    store = ChromaVectorStore(tmp_path / "chroma_db", "duplicate_id_chunks")
    chunks = _sample_chunks()
    embeddings = _sample_embeddings()
    store.add_chunks(chunks, embeddings)

    with pytest.raises(ValueError, match="already exist"):
        store.add_chunks(chunks, embeddings)


def test_invalid_embedding_shapes_are_rejected(tmp_path: Path) -> None:
    """One-dimensional embeddings cannot be added to the collection."""
    store = ChromaVectorStore(tmp_path / "chroma_db", "invalid_shape_chunks")

    with pytest.raises(ValueError, match="two-dimensional"):
        store.add_chunks(_sample_chunks(), np.array([1.0, 0.0, 0.0]))
