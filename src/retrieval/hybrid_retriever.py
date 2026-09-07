"""Hybrid dense and BM25 retrieval combined with Reciprocal Rank Fusion."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict

from src.retrieval.bm25_retriever import BM25RetrievalResult, BM25Retriever
from src.retrieval.build_embeddings import DEFAULT_OUTPUT_PATH
from src.retrieval.dense_retriever import DenseRetrievalResult, DenseRetriever


class DenseRanker(Protocol):
    """Dense retrieval interface required by :class:`HybridRetriever`."""

    def retrieve(self, query: str, top_k: int = 5) -> list[DenseRetrievalResult]:
        """Return a ranked dense result list."""


class BM25Ranker(Protocol):
    """BM25 retrieval interface required by :class:`HybridRetriever`."""

    def retrieve(self, query: str, top_k: int = 5) -> list[BM25RetrievalResult]:
        """Return a ranked BM25 result list."""


class HybridRetrievalResult(TypedDict):
    """One RRF-fused result with original metadata and component scores."""

    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float


class HybridRetriever:
    """Fuse dense and BM25 rankings for one saved financial-report artifact."""

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_OUTPUT_PATH,
        *,
        dense_retriever: DenseRanker | None = None,
        bm25_retriever: BM25Ranker | None = None,
        candidate_count: int = 100,
        rrf_k: int = 60,
    ) -> None:
        """Initialize dense/BM25 retrievers and RRF configuration.

        Args:
            artifact_path: Shared artifact used by default retrievers.
            dense_retriever: Optional injected dense retriever for testing.
            bm25_retriever: Optional injected BM25 retriever for testing.
            candidate_count: Number of candidates requested from each retriever.
            rrf_k: RRF smoothing constant.
        """
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least one")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")

        self.candidate_count = candidate_count
        self.rrf_k = rrf_k
        self._dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DenseRetriever(artifact_path)
        )
        self._bm25_retriever = (
            bm25_retriever
            if bm25_retriever is not None
            else BM25Retriever(artifact_path)
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[HybridRetrievalResult]:
        """Retrieve and fuse dense and BM25 candidates with Reciprocal Rank Fusion.

        Equal RRF scores are ordered by chunk ID to ensure deterministic output.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")

        dense_results = self._dense_retriever.retrieve(query, self.candidate_count)
        bm25_results = self._bm25_retriever.retrieve(query, self.candidate_count)
        fused_results: dict[str, HybridRetrievalResult] = {}

        self._add_dense_scores(fused_results, dense_results)
        self._add_bm25_scores(fused_results, bm25_results)

        ranked_results = sorted(
            fused_results.values(), key=lambda result: (-result["rrf_score"], result["chunk_id"])
        )
        return ranked_results[:top_k]

    def _add_dense_scores(
        self,
        fused_results: dict[str, HybridRetrievalResult],
        results: Sequence[DenseRetrievalResult],
    ) -> None:
        """Add dense ranks and scores to the RRF accumulator."""
        for rank, result in enumerate(results, start=1):
            entry = fused_results.setdefault(
                result["chunk_id"],
                self._new_result(result),
            )
            self._validate_metadata(entry, result)
            entry["dense_score"] = result["similarity_score"]
            entry["rrf_score"] += 1 / (self.rrf_k + rank)

    def _add_bm25_scores(
        self,
        fused_results: dict[str, HybridRetrievalResult],
        results: Sequence[BM25RetrievalResult],
    ) -> None:
        """Add BM25 ranks and scores to the RRF accumulator."""
        for rank, result in enumerate(results, start=1):
            entry = fused_results.setdefault(
                result["chunk_id"],
                self._new_result(result),
            )
            self._validate_metadata(entry, result)
            entry["bm25_score"] = result["bm25_score"]
            entry["rrf_score"] += 1 / (self.rrf_k + rank)

    @staticmethod
    def _new_result(
        result: DenseRetrievalResult | BM25RetrievalResult,
    ) -> HybridRetrievalResult:
        """Create a fused result shell from the first result list that contains it."""
        return HybridRetrievalResult(
            chunk_id=result["chunk_id"],
            text=result["text"],
            page_number=result["page_number"],
            source_filename=result["source_filename"],
            dense_score=None,
            bm25_score=None,
            rrf_score=0.0,
        )

    @staticmethod
    def _validate_metadata(
        entry: HybridRetrievalResult,
        result: DenseRetrievalResult | BM25RetrievalResult,
    ) -> None:
        """Reject mismatched metadata for the same chunk ID across rankers."""
        if (
            entry["text"] != result["text"]
            or entry["page_number"] != result["page_number"]
            or entry["source_filename"] != result["source_filename"]
        ):
            raise ValueError("dense and BM25 metadata must align for each chunk ID")
