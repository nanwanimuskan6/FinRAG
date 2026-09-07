"""Build a persistent ChromaDB collection from a saved embedding artifact."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from src.ingestion.chunker import PDFChunkRecord
from src.retrieval.build_embeddings import DEFAULT_OUTPUT_PATH
from src.retrieval.vector_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)


@dataclass(frozen=True)
class VectorStoreBuildSummary:
    """Summary details from one vector-store build."""

    source_filename: str
    chunks_loaded: int
    embedding_dimension: int
    documents_stored: int
    collection_name: str
    database_path: Path


def load_embedding_artifact(
    artifact_path: str | Path,
) -> tuple[np.ndarray, list[PDFChunkRecord]]:
    """Load and validate aligned embeddings and chunk metadata from an NPZ file.

    Raises:
        ValueError: If required fields are absent or the artifact is malformed.
    """
    path = Path(artifact_path)
    required_fields = {
        "embeddings",
        "chunk_ids",
        "page_numbers",
        "source_filenames",
        "texts",
    }

    with np.load(path, allow_pickle=False) as artifact:
        missing_fields = required_fields.difference(artifact.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"embedding artifact is missing required fields: {missing}")
        embeddings = np.asarray(artifact["embeddings"], dtype=np.float32)
        chunk_ids = artifact["chunk_ids"].tolist()
        page_numbers = artifact["page_numbers"].tolist()
        source_filenames = artifact["source_filenames"].tolist()
        texts = artifact["texts"].tolist()

    _validate_artifact_arrays(
        embeddings, chunk_ids, page_numbers, source_filenames, texts
    )
    chunks = [
        PDFChunkRecord(
            chunk_id=str(chunk_id),
            text=str(text),
            page_number=int(page_number),
            source_filename=str(source_filename),
        )
        for chunk_id, text, page_number, source_filename in zip(
            chunk_ids, texts, page_numbers, source_filenames, strict=True
        )
    ]
    return embeddings, chunks


def _validate_artifact_arrays(
    embeddings: np.ndarray,
    chunk_ids: Sequence[object],
    page_numbers: Sequence[object],
    source_filenames: Sequence[object],
    texts: Sequence[object],
) -> None:
    """Validate all embedding artifact arrays before insertion."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if embeddings.shape[1] <= 0:
        raise ValueError("embedding dimension must be greater than zero")

    embedding_count = embeddings.shape[0]
    named_arrays = {
        "texts": texts,
        "chunk IDs": chunk_ids,
        "page numbers": page_numbers,
        "source filenames": source_filenames,
    }
    for name, values in named_arrays.items():
        if embedding_count != len(values):
            raise ValueError(f"number of embeddings must equal number of {name}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk IDs must be unique")


def build_vector_store(
    artifact_path: str | Path = DEFAULT_OUTPUT_PATH,
    database_path: str | Path = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStoreBuildSummary:
    """Load a saved artifact and insert its aligned records into ChromaDB."""
    embeddings, chunks = load_embedding_artifact(artifact_path)
    store = ChromaVectorStore(database_path, collection_name)
    store.add_chunks(chunks, embeddings)

    source_filename = chunks[0]["source_filename"] if chunks else ""
    return VectorStoreBuildSummary(
        source_filename=source_filename,
        chunks_loaded=len(chunks),
        embedding_dimension=int(embeddings.shape[1]),
        documents_stored=store.count(),
        collection_name=store.collection_name,
        database_path=store.database_path,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Build the persistent ChromaDB collection from an NPZ artifact."""
    parser = argparse.ArgumentParser(
        description="Build a persistent ChromaDB collection from BGE embeddings."
    )
    parser.add_argument("artifact_path", nargs="?", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--database-path", type=Path, default=DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    args = parser.parse_args(arguments)

    try:
        summary = build_vector_store(
            args.artifact_path, args.database_path, args.collection
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Source filename: {summary.source_filename}")
    print(f"Number of chunks loaded: {summary.chunks_loaded}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Number of documents stored: {summary.documents_stored}")
    print(f"Collection name: {summary.collection_name}")
    print(f"Persistent database path: {summary.database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
