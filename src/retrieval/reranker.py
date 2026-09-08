"""Financial-aware cross-encoder reranking for FinRAG."""

from collections.abc import Sequence
import math
import re
from typing import Protocol
from typing import TypedDict

from sentence_transformers import CrossEncoder

from src.retrieval.ranking_text import (
    metric_row_patterns, metric_row_strength, scoring_query, scoring_text,
)


# Modern multilingual BGE cross-encoder. It is materially stronger than the
# original MS MARCO MiniLM baseline on nuanced, long-form financial passages.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankedResult(TypedDict):
    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    rerank_score: float


class CrossEncoderBackend(Protocol):
    """Minimal cross-encoder interface, allowing deterministic tests."""

    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: object) -> Sequence[float]:
        """Return one relevance score for each query/document pair."""


class Reranker:

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model: CrossEncoderBackend | None = None,
        rrf_k: int = 20,
        local_files_only: bool = True,
        batch_size: int = 4,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        self.batch_size = batch_size
        self.model = (
            model
            if model is not None
            else CrossEncoder(model_name, local_files_only=local_files_only)
        )
        self.rrf_k = rrf_k

    def _financial_boost(
        self,
        query: str,
        text: str,
    ) -> float:
        """Prefer evidence containing the exact financial metric in the query.

        Broad terms such as ``financial performance`` occur throughout an
        annual report and previously pushed generic highlights above the exact
        statement row.  We only reward meaningful two-to-four-word phrases
        from the user's question, leaving semantic relevance to the
        cross-encoder.
        """
        ignored_tokens = {
            "a", "an", "and", "for", "from", "how", "in", "is", "of",
            "the", "to", "was", "what", "were", "with", "year",
        }
        query_tokens = [
            token for token in re.findall(r"[a-z]+", query.lower())
            if token not in ignored_tokens and token != "fy"
        ]
        text_lower = text.lower()
        best_phrase_length = 0

        for phrase_length in range(4, 1, -1):
            for start in range(len(query_tokens) - phrase_length + 1):
                phrase = " ".join(query_tokens[start : start + phrase_length])
                if phrase in text_lower:
                    best_phrase_length = max(best_phrase_length, phrase_length)

        return 1.5 * best_phrase_length

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

        focused_query = scoring_query(query, results)
        row_patterns = metric_row_patterns(focused_query)
        pairs = [
            (focused_query, scoring_text(result["text"]))
            for result in results
        ]

        model_scores = self.model.predict(
            pairs, show_progress_bar=False, batch_size=self.batch_size,
        )

        semantic_scores = [float(score) for score in model_scores]
        if len(semantic_scores) != len(results) or not all(map(math.isfinite, semantic_scores)):
            raise ValueError("reranker must return one finite score per candidate")
        semantic_order = sorted(
            range(len(results)),
            key=lambda index: (
                -semantic_scores[index],
                -self._financial_boost(query, results[index]["text"]),
                results[index]["chunk_id"],
            ),
        )
        semantic_ranks = {
            result_index: rank
            for rank, result_index in enumerate(semantic_order, start=1)
        }

        # Preserve candidate recall: the hybrid rank already combines dense and
        # BM25 evidence. Fusing it with the cross-encoder rank prevents a noisy
        # reranker from dropping an otherwise correct table row out of top-k.
        reranked = []
        for original_rank, result in enumerate(results, start=1):
            semantic_rank = semantic_ranks[original_rank - 1]
            # BGE returns relevance probabilities. Promote explicit numeric
            # rows only when the model also considers the passage relevant.
            row_strength = (
                metric_row_strength(result["text"], row_patterns)
                if semantic_scores[original_rank - 1] >= 0.5 else 0
            )
            final_score = (
                2.0 * row_strength
                + 1 / (self.rrf_k + original_rank)
                + 1 / (self.rrf_k + semantic_rank)
                # Break rank-fusion ties in favour of an exact metric phrase.
                + self._financial_boost(query, result["text"]) * 1e-8
            )
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
            key=lambda result: (-result["rerank_score"], result["chunk_id"]),
        )

        return reranked[:top_k]
