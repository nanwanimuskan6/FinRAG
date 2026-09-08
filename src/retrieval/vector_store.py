"""Persistent ChromaDB storage for precomputed FinRAG chunk embeddings."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.ingestion.chunker import PDFChunkRecord


DEFAULT_CHROMA_PATH = Path("data/processed/chroma_db")
DEFAULT_COLLECTION_NAME = "ril_annual_report_chunks"


class RetrievedChunk(TypedDict):
    """A nearest-neighbor result with stored chunk provenance."""

    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    distance: float


class ChromaVectorStore:
    """Persistent Chroma collection that stores externally generated embeddings."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Open a persistent Chroma database and create or get its collection.

        Args:
            database_path: Directory where Chroma persists its database files.
            collection_name: Name of the collection containing report chunks.
        """
        import chromadb

        self.database_path = Path(database_path)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.database_path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def count(self) -> int:
        """Return the number of stored chunk documents."""
        return int(self._collection.count())

    def reset(self) -> None:
        """Delete and recreate this collection for an explicit index rebuild.

        Call this only after a source PDF, chunking strategy, or embedding model
        has changed. It affects only ``self.collection_name`` in this database.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_chunks(
        self, chunks: Sequence[PDFChunkRecord], embeddings: np.ndarray
    ) -> None:
        """Add ordered chunks and externally generated embeddings to Chroma.

        Args:
            chunks: Chunk records in the same order as embedding rows.
            embeddings: Two-dimensional, finite embedding array.

        Raises:
            ValueError: If input shapes are invalid, IDs are duplicated, or an
                ID already exists in the collection.
        """
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        self._validate_add_inputs(chunks, embedding_array)
        if not chunks:
            return

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        existing_ids = self._collection.get(ids=chunk_ids)["ids"]
        if existing_ids:
            raise ValueError("chunk IDs already exist in the collection")

        self._collection.add(
            ids=chunk_ids,
            embeddings=embedding_array.tolist(),
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk["chunk_id"],
                    "page_number": int(chunk["page_number"]),
                    "source_filename": chunk["source_filename"],
                }
                for chunk in chunks
            ],
        )

    def query(
        self, query_embedding: np.ndarray | Sequence[float], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Return the nearest stored chunks for one embedding vector.

        Args:
            query_embedding: One-dimensional BGE query embedding.
            top_k: Maximum number of nearest chunks to return.

        Returns:
            Nearest chunks with their text, provenance, and Chroma distance.

        Raises:
            ValueError: If the query embedding is invalid or ``top_k`` is not
                positive.
        """
        query_array = np.asarray(query_embedding, dtype=np.float32)
        if query_array.ndim != 1:
            raise ValueError("query_embedding must be one-dimensional")
        if query_array.size == 0:
            raise ValueError("query_embedding must not be empty")
        if not np.isfinite(query_array).all():
            raise ValueError("query_embedding must contain only finite values")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_array.tolist()],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                page_number=int(metadata["page_number"]),
                source_filename=str(metadata["source_filename"]),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    @staticmethod
    def _validate_add_inputs(
        chunks: Sequence[PDFChunkRecord], embeddings: np.ndarray
    ) -> None:
        """Validate chunk/embedding alignment before writing to Chroma."""
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("number of embeddings must match number of chunks")
        if embeddings.shape[1] <= 0:
            raise ValueError("embedding dimension must be greater than zero")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk IDs must be unique")
