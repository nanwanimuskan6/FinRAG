"""
FinRAG - Financial Report Intelligence Assistant

Pipeline:
PDF
    -> Dense Retrieval
    -> BM25 Retrieval
    -> Hybrid Retrieval / RRF
    -> Cross Encoder Reranking
    -> Financial Evidence Extraction
    -> Grounded LLM fallback
    -> Answer + Source
"""

from pathlib import Path
import re

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import ChromaDenseRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.evidence import select_evidence
from src.generation.llm_generator import LLMGenerator
from src.retrieval.vector_store import DEFAULT_CHROMA_PATH


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = Path(
    "data/raw/RIL_Annual_Report_2024_25.pdf"
)

EMBEDDING_PATH = Path(
    "data/processed/ril_embeddings.npz"
)

VECTOR_DATABASE_PATH = DEFAULT_CHROMA_PATH

CANDIDATE_COUNT = 100
FINAL_TOP_K = 10
# Keep diagnostic evidence available for development without presenting it to
# users as multiple answers.
SHOW_RETRIEVED_EVIDENCE = False


# ============================================================
# EVIDENCE BUILDER
# ============================================================

def build_evidence(results):
    """Convert retrieved results into LLM-readable evidence."""

    evidence_parts = []

    for index, result in enumerate(results, start=1):

        evidence_parts.append(
            f"""
================ EVIDENCE {index} ================

PDF Page: {result["page_number"]}
Chunk ID: {result["chunk_id"]}
Reranker Score: {result.get("rerank_score", 0.0):.4f}

{result["text"]}

====================================================
"""
        )

    return "\n".join(evidence_parts)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(query, evidence):

    return f"""
You are FinRAG, a financial document question-answering assistant.

Answer the question using ONLY the supplied evidence.

QUESTION:
{query}

EVIDENCE:
{evidence}

STRICT RULES:

1. Use only the supplied evidence.
2. Never use outside knowledge.
3. Never invent a number.
4. Identify the exact financial metric requested.
5. Distinguish consolidated figures from standalone figures.
6. Distinguish company-level figures from segment figures.
7. Preserve the units used in the document.
8. If the answer is not explicitly available, say:
   "I could not find the answer in the provided document."
9. Give a concise answer.
10. Do not add a source citation yourself.

ANSWER:
"""


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(text):
    """Extract financial-style numbers."""

    return re.findall(
        r"\d+(?:,\d+)*(?:\.\d+)?",
        text
    )


# ============================================================
# REVENUE EXTRACTION
# ============================================================

def extract_consolidated_revenue(results):
    """
    Extract consolidated revenue directly from retrieved
    financial-statement evidence.

    Expected pattern in RIL report:

    Value of Sales & Services (Revenue) 10,71,174
    """

    patterns = [

        r"Value\s+of\s+Sales\s*&\s*Services\s*\(Revenue\)"
        r"\s+([\d,]+)",

        r"Value\s+of\s+Sales\s+&\s+Services\s*\(Revenue\)"
        r"\s+([\d,]+)",

        r"Value\s+of\s+Sales\s+and\s+Services"
        r"\s+\(?Revenue\)?\s+([\d,]+)",
    ]

    for result in results:

        text = result["text"]

        # We specifically want consolidated financial evidence.
        if "CONSOLIDATED FINANCIAL STATEMENTS" not in text.upper():
            continue

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                value = match.group(1)

                return {
                    "answer": (
                        f"The total consolidated revenue for "
                        f"FY 2024-25 was **₹{value} crore**."
                    ),
                    "page": result["page_number"],
                    "chunk_id": result["chunk_id"],
                }

    # Second pass: allow the 10-year consolidated highlights.
    for result in results:

        text = result["text"]

        if "10-YEAR FINANCIAL HIGHLIGHTS" not in text.upper():
            continue

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            if match:

                value = match.group(1)

                return {
                    "answer": (
                        f"The total consolidated revenue for "
                        f"FY 2024-25 was **₹{value} crore**."
                    ),
                    "page": result["page_number"],
                    "chunk_id": result["chunk_id"],
                }

    return None


# ============================================================
# GENERIC EVIDENCE VALIDATION
# ============================================================

def validate_llm_answer(answer, results):
    """
    Check whether numerical values produced by the LLM
    actually occur in retrieved evidence.
    """

    answer_numbers = set(
        extract_numbers(answer)
    )

    evidence_numbers = set()

    for result in results:

        evidence_numbers.update(
            extract_numbers(
                result["text"]
            )
        )

    # Every value must be supported, including small percentages and ratios.
    # Numeric occurrence is a guardrail, not proof of semantic correctness.
    return {number.replace(",", "") for number in answer_numbers}.issubset(
        {number.replace(",", "") for number in evidence_numbers}
    )


# ============================================================
# SOURCE PAGE FINDER
# ============================================================

def find_source_pages(answer, results):

    answer_numbers = set(
        extract_numbers(answer)
    )

    source_pages = []

    for result in results:

        text_numbers = set(
            extract_numbers(
                result["text"]
            )
        )

        if answer_numbers.intersection(
            text_numbers
        ):

            page = result["page_number"]

            if page not in source_pages:
                source_pages.append(page)

    return source_pages


# ============================================================
# FORMAT ANSWER
# ============================================================

def format_answer(
    answer,
    results,
    forced_page=None
):

    answer = answer.strip()

    if answer.startswith(("I could not find", "I cannot find")):
        return answer

    # Remove accidental source text generated by LLM.
    answer = re.sub(
        r"\n?\s*Source\s*:.*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    if forced_page is not None:

        source_pages = [
            forced_page
        ]

    else:

        source_pages = find_source_pages(
            answer,
            results
        )

    # Fallback to top evidence pages.
    if not source_pages:

        source_pages = [
            result["page_number"]
            for result in results[:2]
        ]

    source_pages = list(
        dict.fromkeys(
            source_pages
        )
    )

    page_text = ", ".join(
        str(page)
        for page in source_pages[:3]
    )

    return (
        f"{answer}\n\n"
        f"Source: RIL Annual Report 2024-25, "
        f"PDF page {page_text}."
    )


# ============================================================
# DISPLAY EVIDENCE
# ============================================================

def display_results(results):

    print()
    print("=" * 70)
    print("TOP RETRIEVED EVIDENCE")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[{index}] PDF Page: "
            f"{result['page_number']}"
        )

        print(
            f"Reranker Score: "
            f"{result.get('rerank_score', 0.0):.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print()

        print(
            result["text"][:3000]
        )

        print()
        print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinRAG - Financial Report Intelligence Assistant")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if not PDF_PATH.exists():

        print(
            f"\nERROR: PDF not found:\n{PDF_PATH}"
        )

        return

    # --------------------------------------------------------
    # CHECK EMBEDDINGS
    # --------------------------------------------------------

    if not EMBEDDING_PATH.exists():

        print(
            "\nERROR: Embedding file not found."
        )

        print(
            "\nRun:"
        )

        print(
            "python -m src.retrieval.build_embeddings"
        )

        return

    if not VECTOR_DATABASE_PATH.exists():
        print(f"\nERROR: Vector database not found: {VECTOR_DATABASE_PATH}")
        print("Run: python -m src.retrieval.build_vector_store")
        return

    # --------------------------------------------------------
    # LOAD HYBRID RETRIEVER
    # --------------------------------------------------------

    print(
        "\nLoading hybrid retriever..."
    )

    dense_retriever = ChromaDenseRetriever(
        database_path=VECTOR_DATABASE_PATH,
    )
    retriever = HybridRetriever(
        artifact_path=EMBEDDING_PATH,
        dense_retriever=dense_retriever,
        candidate_count=CANDIDATE_COUNT
    )

    print(
        "Hybrid retriever loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD RERANKER
    # --------------------------------------------------------

    print(
        "\nLoading reranker..."
    )

    reranker = Reranker()

    print(
        "Reranker loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD LLM
    # --------------------------------------------------------

    print(
        "\nLoading language model..."
    )

    llm = LLMGenerator()

    print(
        "Language model loaded successfully."
    )

    # --------------------------------------------------------
    # QUESTION LOOP
    # --------------------------------------------------------

    while True:

        print()

        query = input(
            "Ask a question (type 'exit' to quit): "
        ).strip()

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nFinRAG session ended."
            )

            break

        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not query:

            print(
                "\nPlease enter a question."
            )

            continue

        # ----------------------------------------------------
        # HYBRID RETRIEVAL
        # ----------------------------------------------------

        print(
            "\nRetrieving relevant evidence..."
        )

        try:

            candidate_results = retriever.retrieve(
                query,
                top_k=2 * CANDIDATE_COUNT
            )

        except Exception as error:

            print(
                "\nRetrieval error:"
            )

            print(error)

            continue

        if not candidate_results:

            print(
                "\nNo relevant evidence found."
            )

            continue

        print(
            f"Retrieved {len(candidate_results)} "
            f"candidate chunks."
        )

        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        print(
            "Reranking evidence..."
        )

        try:

            results = reranker.rerank(
                query,
                candidate_results,
                top_k=FINAL_TOP_K
            )
            results = select_evidence(candidate_results, results, FINAL_TOP_K)

        except Exception as error:

            print(
                "\nReranking error:"
            )

            print(error)

            continue

        if not results:

            print(
                "\nNo evidence remained after reranking."
            )

            continue

        # ----------------------------------------------------
        # DISPLAY RETRIEVED EVIDENCE
        # ----------------------------------------------------

        if SHOW_RETRIEVED_EVIDENCE:
            display_results(results)

        # ----------------------------------------------------
        # DIRECT FINANCIAL EXTRACTION
        # ----------------------------------------------------

        evidence = build_evidence(
            results
        )

        prompt = build_prompt(
            query,
            evidence
        )

        print()
        print("=" * 70)
        print("GENERATING ANSWER...")
        print("=" * 70)

        try:

            raw_answer = llm.generate(
                prompt
            )

        except Exception as error:

            print(
                "\nGeneration error:"
            )

            print(error)

            continue

        # ----------------------------------------------------
        # VALIDATE LLM ANSWER
        # ----------------------------------------------------

        if validate_llm_answer(
            raw_answer,
            results
        ):

            final_answer = format_answer(
                raw_answer,
                results
            )

        else:

            final_answer = format_answer(
                "I could not find a reliable answer in the provided document.",
                results
            )

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(
            final_answer
        )

        print()
        print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
