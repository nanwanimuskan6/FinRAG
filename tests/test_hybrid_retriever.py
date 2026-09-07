"""Tests for deterministic Reciprocal Rank Fusion of retriever results."""

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:
    """Fixed dense ranker that does not load an embedding model."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Return a deterministic dense result order."""
        return [
            {
                "chunk_id": "chunk_a",
                "text": "Revenue overview.",
                "page_number": 1,
                "source_filename": "report.pdf",
                "similarity_score": 0.90,
            },
            {
                "chunk_id": "chunk_b",
                "text": "Revenue detail.",
                "page_number": 2,
                "source_filename": "report.pdf",
                "similarity_score": 0.80,
            },
        ][:top_k]


class FakeBM25Retriever:
    """Fixed BM25 ranker that does not require an artifact."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Return a deterministic BM25 result order."""
        return [
            {
                "chunk_id": "chunk_b",
                "text": "Revenue detail.",
                "page_number": 2,
                "source_filename": "report.pdf",
                "bm25_score": 7.0,
            },
            {
                "chunk_id": "chunk_c",
                "text": "Debt detail.",
                "page_number": 3,
                "source_filename": "report.pdf",
                "bm25_score": 5.0,
            },
        ][:top_k]


def _retriever() -> HybridRetriever:
    """Create a hybrid retriever with injected deterministic rankers."""
    return HybridRetriever(
        dense_retriever=FakeDenseRetriever(),
        bm25_retriever=FakeBM25Retriever(),
        candidate_count=2,
        rrf_k=60,
    )


def test_hybrid_combines_scores_and_preserves_metadata() -> None:
    """A chunk found by both rankers receives both scores and the top RRF rank."""
    results = _retriever().retrieve("revenue", top_k=3)

    assert [result["chunk_id"] for result in results] == ["chunk_b", "chunk_a", "chunk_c"]
    assert results[0]["dense_score"] == 0.80
    assert results[0]["bm25_score"] == 7.0
    assert results[0]["text"] == "Revenue detail."
    assert results[0]["page_number"] == 2
    assert results[0]["source_filename"] == "report.pdf"
    assert results[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_honors_top_k_and_is_deterministic() -> None:
    """Repeated requests have deterministic RRF ordering and respect top_k."""
    retriever = _retriever()

    first_result = retriever.retrieve("revenue", top_k=1)
    second_result = retriever.retrieve("revenue", top_k=1)

    assert first_result == second_result
    assert len(first_result) == 1


def test_hybrid_validates_inputs_and_candidate_configuration() -> None:
    """Invalid query, top_k, and candidate settings raise clear errors."""
    with pytest.raises(ValueError, match="candidate_count must be at least one"):
        HybridRetriever(
            dense_retriever=FakeDenseRetriever(),
            bm25_retriever=FakeBM25Retriever(),
            candidate_count=0,
        )

    retriever = _retriever()
    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ")
    with pytest.raises(ValueError, match="top_k must be at least one"):
        retriever.retrieve("revenue", top_k=0)
