"""Tests for financial metric-aware cross-encoder reranking."""

from src.retrieval.reranker import Reranker
import pytest


class FlatScoreModel:
    """A model whose equal scores make the lexical tiebreaker observable."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> list[float]:
        return [0.0] * len(sentences)


def test_reranker_prefers_the_exact_financial_metric_not_generic_highlights() -> None:
    reranker = Reranker(model=FlatScoreModel())
    results = [
        {
            "chunk_id": "generic",
            "text": "Consolidated financial performance and revenue highlights.",
            "page_number": 5,
            "source_filename": "report.pdf",
        },
        {
            "chunk_id": "exact",
            "text": "A. Equity Share Capital: 6,766 crore.",
            "page_number": 101,
            "source_filename": "report.pdf",
        },
    ]

    reranked = reranker.rerank(
        "What was the equity share capital in FY 2024-25?", results
    )

    assert [result["chunk_id"] for result in reranked] == ["exact", "generic"]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_phrase_bonus_cannot_override_clear_semantic_relevance():
    class Model:
        def predict(self, sentences, **kwargs):
            return [10.0, -10.0]

    results = [
        {"chunk_id": "correct", "text": "Capital issued: 6,766 crore in 2025.",
         "page_number": 1, "source_filename": "report.pdf"},
        {"chunk_id": "wrong", "text": "Equity share capital accounting policy, no values.",
         "page_number": 2, "source_filename": "report.pdf"},
    ]
    assert Reranker(model=Model()).rerank("equity share capital", results)[0]["chunk_id"] == "correct"


def test_invalid_model_scores_are_rejected():
    class Model:
        def predict(self, sentences, **kwargs):
            return [float("nan")]

    with pytest.raises(ValueError, match="finite score"):
        Reranker(model=Model()).rerank("revenue", [{"text": "Revenue"}])


def test_relevant_numeric_row_is_not_demoted_by_candidate_rank():
    class Model:
        def predict(self, sentences, **kwargs):
            return [-10.0] * (len(sentences) - 1) + [10.0]

    results = [
        {"chunk_id": f"noise_{i:03d}", "text": "Annual report company overview.",
         "page_number": i + 1, "source_filename": "report.pdf"}
        for i in range(50)
    ]
    correct = {"chunk_id": "evidence", "text": "Operating profit: 123 crore.",
               "page_number": 60, "source_filename": "report.pdf"}
    results.append(correct)
    ranked = Reranker(model=Model()).rerank("operating profit", results, top_k=1)
    assert ranked[0]["chunk_id"] == "evidence"
    assert ranked[0]["text"] == correct["text"]
    assert ranked[0]["page_number"] == 60


def test_reranking_does_not_mutate_candidate_order():
    results = [
        {"chunk_id": "b", "text": "Revenue 123 crore.",
         "page_number": 1, "source_filename": "report.pdf"},
        {"chunk_id": "a", "text": "Revenue 123 crore.",
         "page_number": 2, "source_filename": "report.pdf"},
    ]
    reranker = Reranker(model=FlatScoreModel())
    first = reranker.rerank("revenue", results)
    assert first == reranker.rerank("revenue", results)
    assert [r["chunk_id"] for r in results] == ["b", "a"]


def test_low_relevance_numeric_row_is_not_promoted():
    class Model:
        def predict(self, sentences, **kwargs):
            return [0.9, 0.1]
    results = [
        {"chunk_id": "correct", "text": "Capital issued was 123 crore in 2025.",
         "page_number": 1, "source_filename": "report.pdf"},
        {"chunk_id": "wrong", "text": "Equity share capital: 456 crore for another entity.",
         "page_number": 2, "source_filename": "report.pdf"},
    ]
    assert Reranker(model=Model()).rerank("equity share capital", results)[0]["chunk_id"] == "correct"
