"""Tests for the saved embedding artifact format."""

from pathlib import Path

import numpy as np

from src.ingestion.chunker import PDFChunkRecord
from src.retrieval.build_embeddings import save_embedding_artifact


def test_save_embedding_artifact_preserves_embedding_metadata_alignment(
    tmp_path: Path,
) -> None:
    """Fake normalized embeddings round-trip with ordered chunk metadata."""
    chunks: list[PDFChunkRecord] = [
        {
            "chunk_id": "report_p1_c01",
            "text": "First chunk text.",
            "page_number": 1,
            "source_filename": "report.pdf",
        },
        {
            "chunk_id": "report_p2_c01",
            "text": "Second chunk text.",
            "page_number": 2,
            "source_filename": "report.pdf",
        },
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    output_path = tmp_path / "embeddings.npz"

    save_embedding_artifact(output_path, chunks, embeddings)

    with np.load(output_path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "embeddings",
            "chunk_ids",
            "page_numbers",
            "source_filenames",
            "texts",
        }
        assert artifact["embeddings"].shape == (2, 2)
        assert np.array_equal(artifact["embeddings"], embeddings)
        assert artifact["chunk_ids"].tolist() == ["report_p1_c01", "report_p2_c01"]
        assert artifact["page_numbers"].tolist() == [1, 2]
        assert artifact["source_filenames"].tolist() == ["report.pdf", "report.pdf"]
        assert artifact["texts"].tolist() == ["First chunk text.", "Second chunk text."]
