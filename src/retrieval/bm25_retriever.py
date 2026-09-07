"""BM25 keyword retrieval over the saved FinRAG embedding artifact metadata."""

from collections import Counter
from pathlib import Path
import math
import re
from typing import TypedDict

import numpy as np

from src.retrieval.build_embeddings import DEFAULT_OUTPUT_PATH


_TOKEN_PATTERN = re.compile(r"\b\w+\b")


class BM25RetrievalResult(TypedDict):
    """One ranked BM25 result with original chunk provenance."""

    chunk_id: str
    text: str
    page_number: int
    source_filename: str
    bm25_score: float


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text consistently for BM25 indexing and queries."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 retriever over the chunk texts in one embedding artifact."""

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_OUTPUT_PATH,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Load chunk metadata and build a BM25 index.

        Args:
            artifact_path: NPZ artifact containing chunk metadata and embeddings.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 document-length normalization parameter.
        """
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.artifact_path = Path(artifact_path)
        (
            self._chunk_ids,
            self._page_numbers,
            self._source_filenames,
            self._texts,
        ) = self._load_metadata(self.artifact_path)
        self.k1 = k1
        self.b = b
        self._term_frequencies = [Counter(tokenize(str(text))) for text in self._texts]
        self._document_lengths = np.asarray(
            [sum(frequencies.values()) for frequencies in self._term_frequencies],
            dtype=np.float64,
        )
        self._average_document_length = (
            float(self._document_lengths.mean()) if self.count else 0.0
        )
        self._document_frequencies = Counter(
            token
            for frequencies in self._term_frequencies
            for token in frequencies
        )

    @property
    def count(self) -> int:
        """Return the number of indexed chunks."""
        return int(self._texts.shape[0])

    def retrieve(self, query: str, top_k: int = 5) -> list[BM25RetrievalResult]:
        """Return the highest-scoring keyword matches for a natural-language query.

        Equal BM25 scores retain the original artifact order, making results
        deterministic across repeated retrieval calls.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if self.count == 0:
            return []

        scores = np.zeros(self.count, dtype=np.float64)
        for token in set(query_tokens):
            document_frequency = self._document_frequencies.get(token, 0)
            if document_frequency == 0:
                continue

            inverse_document_frequency = math.log(
                1 + (self.count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for index, frequencies in enumerate(self._term_frequencies):
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                length_ratio = (
                    self._document_lengths[index] / self._average_document_length
                    if self._average_document_length
                    else 0.0
                )
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1) / denominator
                )

        artifact_indices = np.arange(self.count)
        ranked_indices = np.lexsort((artifact_indices, -scores))
        selected_indices = ranked_indices[: min(top_k, self.count)]
        return [
            BM25RetrievalResult(
                chunk_id=str(self._chunk_ids[index]),
                text=str(self._texts[index]),
                page_number=int(self._page_numbers[index]),
                source_filename=str(self._source_filenames[index]),
                bm25_score=float(scores[index]),
            )
            for index in selected_indices
        ]

    @staticmethod
    def _load_metadata(
        artifact_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and validate chunk metadata aligned with artifact embeddings."""
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
            embeddings = np.asarray(artifact["embeddings"])
            chunk_ids = np.asarray(artifact["chunk_ids"])
            page_numbers = np.asarray(artifact["page_numbers"])
            source_filenames = np.asarray(artifact["source_filenames"])
            texts = np.asarray(artifact["texts"])

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        record_count = embeddings.shape[0]
        metadata_arrays = {
            "chunk IDs": chunk_ids,
            "page numbers": page_numbers,
            "source filenames": source_filenames,
            "texts": texts,
        }
        for name, values in metadata_arrays.items():
            if values.ndim != 1 or values.shape[0] != record_count:
                raise ValueError(f"number of embeddings must equal number of {name}")
        return chunk_ids, page_numbers, source_filenames, texts
