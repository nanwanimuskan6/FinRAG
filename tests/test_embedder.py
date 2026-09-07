"""Tests for the dependency-injectable BGE embedding wrapper."""

from collections.abc import Sequence

import numpy as np

from src.retrieval.embedder import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_MODEL_NAME,
    EmbeddingModel,
)


class FakeEmbeddingBackend:
    """In-memory embedding backend that never downloads a model."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        """Return the fixed embedding size used by this test double."""
        return 3

    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray:
        """Return deterministic vectors that reflect the input order."""
        text_list = list(sentences)
        self.calls.append((text_list, kwargs))
        return np.array(
            [[index, len(text), 1.0] for index, text in enumerate(text_list)],
            dtype=np.float32,
        )


def test_empty_input_returns_an_empty_array_without_encoding() -> None:
    """Empty documents return a correctly shaped NumPy array safely."""
    backend = FakeEmbeddingBackend()
    embedder = EmbeddingModel(model=backend)

    embeddings = embedder.encode_texts([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 3)
    assert backend.calls == []


def test_encode_texts_preserves_order_and_output_shape() -> None:
    """Document embeddings are two-dimensional and retain input order."""
    backend = FakeEmbeddingBackend()
    embedder = EmbeddingModel(model=backend)

    embeddings = embedder.encode_texts(["first", "second"])

    assert embeddings.shape == (2, 3)
    assert embeddings[:, 1].tolist() == [5.0, 6.0]
    assert backend.calls[0][0] == ["first", "second"]
    assert backend.calls[0][1]["normalize_embeddings"] is True


def test_encode_query_returns_one_embedding_with_bge_instruction() -> None:
    """Queries use BGE's retrieval instruction and return one vector."""
    backend = FakeEmbeddingBackend()
    embedder = EmbeddingModel(model=backend)

    embedding = embedder.encode_query("What was revenue growth?")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (3,)
    assert backend.calls[0][0] == [
        f"{BGE_QUERY_INSTRUCTION}What was revenue growth?"
    ]
    assert backend.calls[0][1]["normalize_embeddings"] is True


def test_default_model_configuration_and_injected_backend() -> None:
    """The default BGE model name is retained without loading a real model."""
    backend = FakeEmbeddingBackend()

    embedder = EmbeddingModel(model=backend)

    assert embedder.model_name == DEFAULT_MODEL_NAME
    assert embedder._model is backend
