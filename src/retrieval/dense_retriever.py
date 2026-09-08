"""Dense cosine-similarity retrieval over saved FinRAG embedding artifacts."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np

from src.retrieval.build_embeddings import DEFAULT_OUTPUT_PATH
from src.retrieval.embedder import EmbeddingModel
from src.retrieval.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)


class QueryEmbedder(Protocol):
    """Minimal query-encoding interface required by :class:`DenseRetriever`."""

    def encode_query(self, query: str) -> np.ndarray:
        """Return one normalized query embedding."""


class DenseRetrievalResult(TypedDict):
    """One ranked dense-retrieval result with chunk provenance."""

    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    similarity_score: float


class DenseRetriever:
    """Retrieve relevant report chunks using normalized BGE embedding vectors."""

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_OUTPUT_PATH,
        *,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        """Load one embedding artifact and initialize or accept a query encoder.

        Args:
            artifact_path: Path to the NPZ artifact created by the embedding stage.
            embedder: Optional query encoder, primarily for tests.

        Raises:
            ValueError: If artifact arrays are missing, malformed, or misaligned.
        """
        self.artifact_path = Path(artifact_path)
        (
            self._embeddings,
            self._chunk_ids,
            self._page_numbers,
            self._source_filenames,
            self._texts,
        ) = self._load_artifact(self.artifact_path)
        self._embedder = (
            embedder if embedder is not None else EmbeddingModel(local_files_only=True)
        )

    @property
    def count(self) -> int:
        """Return the number of document chunks available for retrieval."""
        return int(self._embeddings.shape[0])

    def retrieve(self, query: str, top_k: int = 5) -> list[DenseRetrievalResult]:
        """Return the highest-scoring chunks for a natural-language query.

        The stored document vectors and BGE query vector are L2-normalized, so
        their dot product equals cosine similarity. Equal scores are resolved by
        the original artifact order for deterministic results.

        Args:
            query: Natural-language question to retrieve against.
            top_k: Maximum number of ranked chunks to return.

        Raises:
            ValueError: If the query is empty, ``top_k`` is invalid, or the
                query embedding does not match the artifact dimension.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if self.count == 0:
            return []

        query_embedding = np.asarray(self._embedder.encode_query(query), dtype=np.float32)
        if query_embedding.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        if query_embedding.shape[0] != self._embeddings.shape[1]:
            raise ValueError("query embedding dimension must match document embeddings")
        if not np.isfinite(query_embedding).all():
            raise ValueError("query embedding must contain only finite values")

        similarity_scores = self._embeddings @ query_embedding
        artifact_indices = np.arange(self.count)
        ranked_indices = np.lexsort((artifact_indices, -similarity_scores))
        selected_indices = ranked_indices[: min(top_k, self.count)]

        return [
            DenseRetrievalResult(
                chunk_id=str(self._chunk_ids[index]),
                text=str(self._texts[index]),
                page_number=int(self._page_numbers[index]),
                source_filename=str(self._source_filenames[index]),
                similarity_score=float(similarity_scores[index]),
            )
            for index in selected_indices
        ]

    @staticmethod
    def _load_artifact(
        artifact_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and validate aligned arrays from a saved embedding artifact."""
        required_fields = {
            "embeddings",
            "chunk_ids",
            "page_numbers",
            "source_filenames",
            "texts",
        }
        with np.load(artifact_path, allow_pickle=False) as artifact:
            missing_fields = required_fields.difference(artifact.files)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"embedding artifact is missing required fields: {missing}")
            embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
            chunk_ids = np.asarray(artifact["chunk_ids"])
            page_numbers = np.asarray(artifact["page_numbers"])
            source_filenames = np.asarray(artifact["source_filenames"])
            texts = np.asarray(artifact["texts"])

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if embeddings.shape[1] <= 0:
            raise ValueError("embedding dimension must be greater than zero")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")

        record_count = embeddings.shape[0]
        named_arrays = {
            "chunk IDs": chunk_ids,
            "page numbers": page_numbers,
            "source filenames": source_filenames,
            "texts": texts,
        }
        for name, values in named_arrays.items():
            if values.ndim != 1 or values.shape[0] != record_count:
                raise ValueError(f"number of embeddings must equal number of {name}")

        return embeddings, chunk_ids, page_numbers, source_filenames, texts


class ChromaDenseRetriever:
    """Dense retrieval backed by the persistent Chroma vector database.

    It uses the same BGE query encoder as the embedding pipeline, so Chroma's
    cosine-distance search is compatible with the vectors already persisted by
    :mod:`src.retrieval.build_vector_store`.
    """

    def __init__(
        self,
        database_path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        *,
        embedder: QueryEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self._embedder = (
            embedder if embedder is not None else EmbeddingModel(local_files_only=True)
        )
        self._store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore(database_path, collection_name)
        )

    @property
    def count(self) -> int:
        """Return the number of chunks indexed in Chroma."""
        return self._store.count()

    def retrieve(self, query: str, top_k: int = 5) -> list[DenseRetrievalResult]:
        """Retrieve the nearest stored chunks and convert distance to similarity."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if self.count == 0:
            return []

        query_embedding = np.asarray(self._embedder.encode_query(query), dtype=np.float32)
        if query_embedding.ndim != 1 or query_embedding.size == 0:
            raise ValueError("query embedding must be a non-empty one-dimensional vector")
        if not np.isfinite(query_embedding).all():
            raise ValueError("query embedding must contain only finite values")

        return [
            DenseRetrievalResult(
                chunk_id=result["chunk_id"],
                text=result["text"],
                page_number=result["page_number"],
                source_filename=result["source_filename"],
                similarity_score=1.0 - result["distance"],
            )
            for result in self._store.query(query_embedding, top_k)
        ]
