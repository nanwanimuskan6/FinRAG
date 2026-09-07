"""Tests for BM25 keyword retrieval over synthetic artifacts."""

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.bm25_retriever import BM25Retriever, tokenize


def _write_artifact(path: Path) -> None:
    """Create an artifact with keyword-distinct chunk text."""
    np.savez_compressed(
        path,
        embeddings=np.eye(3, dtype=np.float32),
        chunk_ids=np.array(["chunk_a", "chunk_b", "chunk_c"]),
        page_numbers=np.array([10, 20, 30]),
        source_filenames=np.array(["report.pdf", "report.pdf", "report.pdf"]),
        texts=np.array(
            ["Revenue growth improved.", "Debt levels declined.", "Revenue revenue outlook."]
        ),
    )


def test_tokenize_is_case_insensitive_and_keeps_numeric_tokens() -> None:
    """Indexing and querying use the same normalized tokenization."""
    assert tokenize("Revenue 2024-25") == ["revenue", "2024", "25"]


def test_bm25_exact_keyword_matching_preserves_metadata(tmp_path: Path) -> None:
    """The strongest keyword match ranks first with its original metadata."""
    artifact_path = tmp_path / "embeddings.npz"
    _write_artifact(artifact_path)
    retriever = BM25Retriever(artifact_path)

    results = retriever.retrieve("revenue", top_k=2)

    assert [result["chunk_id"] for result in results] == ["chunk_c", "chunk_a"]
    assert results[0]["text"] == "Revenue revenue outlook."
    assert results[0]["page_number"] == 30
    assert results[0]["source_filename"] == "report.pdf"
    assert results[0]["bm25_score"] > results[1]["bm25_score"]


def test_bm25_validates_query_and_top_k(tmp_path: Path) -> None:
    """Empty queries and invalid result counts fail clearly."""
    artifact_path = tmp_path / "embeddings.npz"
    _write_artifact(artifact_path)
    retriever = BM25Retriever(artifact_path)

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("!!!")
    with pytest.raises(ValueError, match="top_k must be at least one"):
        retriever.retrieve("revenue", top_k=0)
