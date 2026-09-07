# FinRAG

## PDF ingestion

`src/ingestion/pdf_loader.py` provides `load_pdf(pdf_path)`, a reusable
PyMuPDF-based loader. It returns one structured record per PDF page with the
one-based page number, extracted text, and source filename. Empty pages are
retained with an empty text value so that page metadata is never lost.

The component deliberately only performs PDF text extraction; it does not yet
chunk documents, create embeddings, retrieve content, or call an LLM.

### Manual ingestion

To inspect a PDF manually, place it in `data/raw/` and run:

```powershell
python -m src.ingestion.run_ingestion data/raw/<filename>.pdf
```

The command prints the total page count plus the source filename, page number,
and first 500 characters extracted from the first page.

### Inspect selected pages

To preview extraction quality on pages 2, 10, 50, and 100, run:

```powershell
python -m src.ingestion.inspect_pages data/raw/<filename>.pdf
```

Pages that are outside the PDF's page range are skipped with a message.

## Conservative text cleaning

`src/ingestion/text_cleaner.py` normalizes line endings, repeated horizontal
whitespace, excessive blank lines, and page-boundary whitespace. It preserves
the original line order, page metadata, words, and financial values; it does
not infer, summarize, or restructure extracted content.

## Smart chunking

`src/ingestion/chunker.py` converts cleaned page records into deterministic,
page-isolated chunks. It targets 1,000 characters with 150 characters of
overlap, preferring paragraph boundaries, then line or whitespace boundaries.
Oversized paragraphs are safely split at the target length; every chunk keeps
its original page number and source filename.

### Inspect chunks

To inspect chunk statistics and representative chunks from a PDF, run:

```powershell
python -m src.ingestion.inspect_chunks data/raw/<filename>.pdf
```

## Embeddings

`src/retrieval/embedder.py` uses Sentence Transformers with
`BAAI/bge-small-en-v1.5` to encode document chunks and user queries as dense
vectors. This compact retrieval model uses raw chunk text for documents and
the BGE retrieval instruction for queries, helping align queries with relevant
financial-report passages while keeping the model reusable in memory.

### Build real report embeddings

`src/retrieval/build_embeddings.py` runs the existing PDF loading, cleaning,
and page-isolated chunking pipeline, then saves normalized BGE embeddings and
aligned chunk metadata in `data/processed/ril_embeddings.npz`.

```powershell
python -m src.retrieval.build_embeddings data/raw/RIL_Annual_Report_2024_25.pdf
```

## Persistent vector storage

FinRAG stores the pipeline output as: PDF chunks → BGE embeddings → a persistent
ChromaDB collection. `src/retrieval/build_vector_store.py` loads the saved NPZ
artifact without regenerating embeddings and writes the chunk text plus page and
source metadata to `data/processed/chroma_db/` for later semantic retrieval.

## Dense retrieval

`src/retrieval/dense_retriever.py` encodes a user question with BGE and compares
its normalized vector to every saved document embedding using a dot product.
The highest cosine-similarity scores identify the most relevant report chunks,
while preserving each chunk's page and source metadata.

## Hybrid retrieval

`src/retrieval/bm25_retriever.py` adds deterministic BM25 keyword retrieval over
the same saved chunks used by dense retrieval. `src/retrieval/hybrid_retriever.py`
combines dense and BM25 rank lists with Reciprocal Rank Fusion (RRF), preserving
chunk provenance and returning both component scores alongside the fused score.
