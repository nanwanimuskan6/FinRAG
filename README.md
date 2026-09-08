# FinRAG

A financial-document RAG project that retrieves evidence from annual-report PDFs with source-page citations. Built with pretrained models; currently a terminal app, evaluated on Reliance Industries' FY 2024-25 report.

## How it works

PDF extraction and chunking -> BGE embeddings -> ChromaDB + BM25 search -> rank fusion -> financial-aware BGE reranking -> cited evidence for Gemini.

- Preserves page numbers and source text.
- Prioritizes relevant metric/value rows and reduces report-footer noise.
- Checks generated numbers against retrieved evidence.

**Stack:** Python, PyMuPDF, Sentence Transformers, ChromaDB, PyTorch, Google GenAI.

## Results

Measured on a **20-question development benchmark**:

| Metric | Result |
|---|---:|
| Evidence hit rate @1 | **75%** |
| Evidence hit rate @5 | **90%** |
| Evidence hit rate @10 | **95%** |
| MRR @10 (answer matching) | **0.785** |
| Tests passing | **74** |

Top-1 evidence coverage improved from 50% to 75%. These are retrieval results on a small development set, not independent-test or generated-answer accuracy. Precision and F1 have not been measured.

## Run locally

Use Python 3.11+ in a virtual environment. From the project root:

```powershell
python -m pip install -r requirements.txt
```

Place the report at `data/raw/RIL_Annual_Report_2024_25.pdf`, then build the index and download the reranker once:

```powershell
python -m src.retrieval.build_embeddings
python -m src.retrieval.build_vector_store --rebuild
python -c "from src.retrieval.reranker import Reranker; Reranker(local_files_only=False)"
```

The first setup downloads pretrained models. `--rebuild` replaces the named FinRAG collection. PDFs, indexes, and model files are excluded from Git.

For interactive answers, set your own Gemini API key in the same terminal:

```powershell
$env:GEMINI_API_KEY = "your_api_key"
python -m app.rag_app
```

Example: *What was Reliance's net worth in FY 2024-25?*

Gemini answer generation is implemented but has not been evaluated end to end. A key is required for answers; retrieval evaluation runs locally without one. Numeric checks alone cannot verify the metric, year, or scope, and may reject calculated values.

## Evaluate

```powershell
python -m src.evaluation.evaluate_retrieval
python -m pytest -q
```

Metrics are saved to `data/processed/retrieval_metrics.json`. Use `--skip-rerank` for a hybrid-only evaluation.
