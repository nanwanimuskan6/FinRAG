"""Dense embedding utilities for FinRAG document chunks and user queries."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingBackend(Protocol):
    """Minimal Sentence Transformers interface used by :class:`EmbeddingModel`."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray:
        """Encode text into sentence embeddings."""

    def get_sentence_embedding_dimension(self) -> int:
        """Return the dimensionality of each embedding."""


class EmbeddingModel:
    """Reusable BGE encoder for document chunks and retrieval queries.

    Documents are encoded without an instruction. Queries use BGE's published
    retrieval instruction, which aligns query embeddings with passage embeddings.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        model: EmbeddingBackend | None = None,
    ) -> None:
        """Load a Sentence Transformers model once, or accept an injected model.

        Args:
            model_name: Hugging Face model identifier to load when no model is injected.
            model: Optional preconstructed backend, primarily for testing.
        """
        self.model_name = model_name
        self._model = model if model is not None else self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str) -> EmbeddingBackend:
        """Load the Sentence Transformers backend only when it is needed."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode document texts in input order as normalized NumPy embeddings.

        Args:
            texts: Document chunk texts to encode.

        Returns:
            A two-dimensional array with one embedding per input text. An empty
            input returns an array of shape ``(0, embedding_dimension)``.
        """
        text_list = list(texts)
        if not text_list:
            dimension = self._model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype=np.float32)

        embeddings = self._model.encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one query using BGE's retrieval instruction.

        Args:
            query: User query to encode.

        Returns:
            A one-dimensional NumPy array containing the query embedding.
        """
        embeddings = self._model.encode(
            [f"{BGE_QUERY_INSTRUCTION}{query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        return embedding_array if embedding_array.ndim == 1 else embedding_array[0]
