"""Financial-aware cross-encoder reranking for FinRAG."""

from collections.abc import Sequence
from typing import TypedDict

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankedResult(TypedDict):
    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    rerank_score: float


class Reranker:

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.model = CrossEncoder(model_name)

    def _financial_boost(
        self,
        query: str,
        text: str,
    ) -> float:
        """Boost evidence containing important financial query concepts."""

        query_lower = query.lower()
        text_lower = text.lower()

        boost = 0.0

        # General revenue terms
        if "revenue" in query_lower and "revenue" in text_lower:
            boost += 1.5

        # Consolidated-company evidence
        if "consolidated" in query_lower:
            if "consolidated" in text_lower:
                boost += 2.5

        # Financial-year matching
        if "2024-25" in query_lower and "2024-25" in text_lower:
            boost += 1.5

        # Stronger phrases for company-level revenue
        strong_phrases = [
            "consolidated revenue",
            "revenue from operations",
            "value of sales and services",
            "consolidated financial statements",
            "financial performance",
        ]

        for phrase in strong_phrases:
            if phrase in text_lower:
                boost += 1.0

        # Penalize obvious unrelated financial metrics
        negative_phrases = [
            "net debt",
            "gross debt",
            "total debt",
            "employee benefit",
            "goodwill",
        ]

        for phrase in negative_phrases:
            if phrase in text_lower:
                boost -= 2.0

        return boost

    def rerank(
        self,
        query: str,
        results: Sequence[dict],
        top_k: int = 5,
    ) -> list[RerankedResult]:

        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least one")

        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        model_scores = self.model.predict(pairs)

        reranked = []

        for result, model_score in zip(results, model_scores):

            financial_boost = self._financial_boost(
                query,
                result["text"],
            )

            final_score = float(model_score) + financial_boost

            reranked.append(
                RerankedResult(
                    chunk_id=result["chunk_id"],
                    text=result["text"],
                    page_number=result["page_number"],
                    source_filename=result["source_filename"],
                    rerank_score=final_score,
                )
            )

        reranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]