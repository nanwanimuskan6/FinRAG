"""Build and persist normalized BGE embeddings for a financial report PDF."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from src.ingestion.chunker import PDFChunkRecord, chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_cleaner import clean_pages
from src.retrieval.embedder import EmbeddingModel


DEFAULT_PDF_PATH = Path("data/raw/RIL_Annual_Report_2024_25.pdf")
DEFAULT_OUTPUT_PATH = Path("data/processed/ril_embeddings.npz")
_NORMALIZATION_TOLERANCE = 1e-3


@dataclass(frozen=True)
class EmbeddingBuildSummary:
    """Summary details from one successful embedding build."""

    source_filename: str
    total_pages: int
    total_chunks: int
    embedding_shape: tuple[int, int]
    minimum_norm: float
    maximum_norm: float
    output_path: Path


def validate_embeddings(
    embeddings: np.ndarray, chunks: Sequence[PDFChunkRecord]
) -> np.ndarray:
    """Validate embedding shape, alignment, numeric values, and normalization.

    Args:
        embeddings: Document embeddings, one row per chunk.
        chunks: Chunk records aligned with the embedding rows.

    Returns:
        The L2 norm of each embedding row.

    Raises:
        ValueError: If embeddings are misaligned, malformed, non-finite, or not
            normalized within the configured floating-point tolerance.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if embeddings.shape[0] != len(chunks):
        raise ValueError("number of embeddings must match number of chunks")
    if embeddings.shape[1] <= 0:
        raise ValueError("embedding dimension must be greater than zero")
    if np.isnan(embeddings).any():
        raise ValueError("embeddings must not contain NaN values")
    if np.isinf(embeddings).any():
        raise ValueError("embeddings must not contain infinite values")

    norms = np.linalg.norm(embeddings, axis=1)
    if norms.size and not np.allclose(
        norms, 1.0, rtol=_NORMALIZATION_TOLERANCE, atol=_NORMALIZATION_TOLERANCE
    ):
        raise ValueError("embeddings must be L2-normalized")
    return norms


def save_embedding_artifact(
    output_path: str | Path,
    chunks: Sequence[PDFChunkRecord],
    embeddings: np.ndarray,
) -> None:
    """Save embeddings and aligned chunk metadata to a compressed NPZ artifact.

    Args:
        output_path: Destination ``.npz`` path.
        chunks: Ordered chunk records.
        embeddings: Ordered, normalized embedding rows.
    """
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    validate_embeddings(embedding_array, chunks)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embedding_array,
        chunk_ids=np.asarray([chunk["chunk_id"] for chunk in chunks], dtype=str),
        page_numbers=np.asarray([chunk["page_number"] for chunk in chunks], dtype=np.int64),
        source_filenames=np.asarray(
            [chunk["source_filename"] for chunk in chunks], dtype=str
        ),
        texts=np.asarray([chunk["text"] for chunk in chunks], dtype=str),
    )


def build_embeddings(
    pdf_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    embedder: EmbeddingModel | None = None,
) -> EmbeddingBuildSummary:
    """Run PDF loading, cleaning, chunking, embedding, validation, and saving.

    Args:
        pdf_path: Source financial-report PDF.
        output_path: Destination for the compressed embedding artifact.
        embedder: Optional initialized embedder, primarily for testing.

    Returns:
        A summary of the generated, validated embedding artifact.
    """
    pages = load_pdf(pdf_path)
    cleaned_pages = clean_pages(pages)
    chunks = chunk_pages(cleaned_pages)
    model = embedder if embedder is not None else EmbeddingModel()
    embeddings = model.encode_texts([chunk["text"] for chunk in chunks])
    norms = validate_embeddings(embeddings, chunks)
    save_embedding_artifact(output_path, chunks, embeddings)

    embedding_shape = (int(embeddings.shape[0]), int(embeddings.shape[1]))
    source_filename = pages[0]["source_filename"] if pages else Path(pdf_path).name
    return EmbeddingBuildSummary(
        source_filename=source_filename,
        total_pages=len(pages),
        total_chunks=len(chunks),
        embedding_shape=embedding_shape,
        minimum_norm=float(norms.min()) if norms.size else 0.0,
        maximum_norm=float(norms.max()) if norms.size else 0.0,
        output_path=Path(output_path),
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Build a saved BGE embedding artifact from a report PDF."""
    parser = argparse.ArgumentParser(
        description="Build normalized BGE embeddings for a financial report PDF."
    )
    parser.add_argument("pdf_path", nargs="?", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(arguments)

    try:
        summary = build_embeddings(args.pdf_path, args.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Source filename: {summary.source_filename}")
    print(f"Total pages: {summary.total_pages}")
    print(f"Total chunks: {summary.total_chunks}")
    print(f"Embedding shape: {summary.embedding_shape}")
    print(f"Embedding dimension: {summary.embedding_shape[1]}")
    print(f"Minimum norm: {summary.minimum_norm:.6f}")
    print(f"Maximum norm: {summary.maximum_norm:.6f}")
    print(f"Saved file: {summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
