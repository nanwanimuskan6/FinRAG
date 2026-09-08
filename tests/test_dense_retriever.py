"""Tests for dense retrieval over a synthetic embedding artifact."""

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.dense_retriever import ChromaDenseRetriever, DenseRetriever


class FakeQueryEmbedder:
    """Query encoder with fixed vectors and no external model dependency."""

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def encode_query(self, query: str) -> np.ndarray:
        """Return the configured vector while recording the incoming query."""
        self.queries.append(query)
        return self.vector


def _write_artifact(path: Path) -> None:
    """Create three ordered, normalized synthetic document embeddings."""
    np.savez_compressed(
        path,
        embeddings=np.array(
            [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=np.float32
        ),
        chunk_ids=np.array(["chunk_a", "chunk_b", "chunk_c"]),
        page_numbers=np.array([10, 20, 30]),
        source_filenames=np.array(["report.pdf", "report.pdf", "report.pdf"]),
        texts=np.array(["Revenue grew.", "Debt declined.", "Risk factors."]),
    )


def _retriever(tmp_path: Path) -> DenseRetriever:
    """Build a retriever over a temporary artifact and fixed query vector."""
    artifact_path = tmp_path / "embeddings.npz"
    _write_artifact(artifact_path)
    return DenseRetriever(
        artifact_path,
        embedder=FakeQueryEmbedder(np.array([1.0, 0.0], dtype=np.float32)),
    )


def test_retrieve_ranks_chunks_and_preserves_metadata(tmp_path: Path) -> None:
    """The closest vector ranks first with its aligned metadata and text."""
    results = _retriever(tmp_path).retrieve("How did revenue change?", top_k=2)

    assert [result["chunk_id"] for result in results] == ["chunk_a", "chunk_b"]
    assert results[0]["text"] == "Revenue grew."
    assert results[0]["page_number"] == 10
    assert results[0]["source_filename"] == "report.pdf"
    assert results[0]["similarity_score"] == pytest.approx(1.0)


def test_retrieve_honors_top_k_and_returns_all_available_when_larger(
    tmp_path: Path,
) -> None:
    """A larger requested result count returns every available chunk."""
    retriever = _retriever(tmp_path)

    assert len(retriever.retrieve("revenue", top_k=1)) == 1
    assert len(retriever.retrieve("revenue", top_k=10)) == 3


def test_retrieve_rejects_empty_queries_and_invalid_top_k(tmp_path: Path) -> None:
    """Invalid user input is rejected before retrieval runs."""
    retriever = _retriever(tmp_path)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ")
    with pytest.raises(ValueError, match="top_k must be at least one"):
        retriever.retrieve("revenue", top_k=0)


class FakeVectorStore:
    def count(self) -> int:
        return 1

    def query(self, embedding: np.ndarray, top_k: int) -> list[dict[str, object]]:
        assert embedding.tolist() == [1.0, 0.0]
        assert top_k == 1
        return [
            {
                "chunk_id": "chunk_a",
                "text": "Revenue grew.",
                "page_number": 10,
                "source_filename": "report.pdf",
                "distance": 0.2,
            }
        ]


def test_chroma_dense_retriever_preserves_provenance_and_converts_distance() -> None:
    retriever = ChromaDenseRetriever(
        embedder=FakeQueryEmbedder(np.array([1.0, 0.0], dtype=np.float32)),
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
    )

    result = retriever.retrieve("revenue", top_k=1)

    assert result == [
        {
            "chunk_id": "chunk_a",
            "text": "Revenue grew.",
            "page_number": 10,
            "source_filename": "report.pdf",
            "similarity_score": pytest.approx(0.8),
        }
    ]
