"""
Evaluate FinRAG retrieval quality using Recall@K and MRR.

Run from the FinRAG project root:

    python -m src.evaluation.evaluate_retrieval
"""

from __future__ import annotations

from typing import Any

from src.evaluation.eval_dataset import get_evaluation_dataset
from src.retrieval.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

TOP_K_VALUES = [1, 3, 5, 10]

# Number of candidates retrieved before final evaluation.
CANDIDATE_COUNT = 100


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text so answer matching is less sensitive
    to capitalization, commas, spaces, etc.
    """
    return (
        str(text)
        .lower()
        .replace(",", "")
        .replace("₹", "")
        .replace("`", "")
        .replace("crore", "")
        .replace("cr", "")
        .strip()
    )


# ---------------------------------------------------------
# RELEVANCE CHECK
# ---------------------------------------------------------

def result_is_relevant(
    result: dict[str, Any],
    expected_answer: str,
) -> bool:
    """
    Determine whether a retrieved chunk contains the expected answer.

    This is intentionally simple and transparent for the first
    retrieval evaluation.
    """

    chunk_text = normalize_text(result.get("text", ""))
    expected = normalize_text(expected_answer)

    if not expected:
        return False

    # Exact answer occurrence
    if expected in chunk_text:
        return True

    # Handle numeric answers with spaces/commas.
    expected_digits = (
        expected.replace(" ", "")
        .replace(".", "")
        .replace("/", "")
        .replace("-", "")
    )

    chunk_digits = (
        chunk_text.replace(" ", "")
        .replace(".", "")
        .replace("/", "")
        .replace("-", "")
    )

    if expected_digits and expected_digits in chunk_digits:
        return True

    return False


# ---------------------------------------------------------
# RECIPROCAL RANK
# ---------------------------------------------------------

def reciprocal_rank(
    results: list[dict[str, Any]],
    expected_answer: str,
) -> float:
    """
    Return reciprocal rank of the first relevant result.

    Example:
        relevant result at rank 1 -> 1.0
        rank 2 -> 0.5
        rank 5 -> 0.2
        not found -> 0.0
    """

    for rank, result in enumerate(results, start=1):
        if result_is_relevant(result, expected_answer):
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# RECALL@K
# ---------------------------------------------------------

def recall_at_k(
    results: list[dict[str, Any]],
    expected_answer: str,
    k: int,
) -> float:
    """
    Return 1 if a relevant result occurs within top-k,
    otherwise 0.
    """

    top_results = results[:k]

    for result in top_results:
        if result_is_relevant(result, expected_answer):
            return 1.0

    return 0.0


# ---------------------------------------------------------
# PRINT RESULT
# ---------------------------------------------------------

def print_result(
    question_number: int,
    item: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """
    Print evaluation details for one question.
    """

    print("=" * 80)
    print(f"QUESTION {question_number}")
    print("=" * 80)

    print(f"Question : {item['question']}")
    print(f"Expected : {item['answer']}")
    print(f"Expected pages : {item.get('pages', [])}")
    print()

    print("TOP RETRIEVED RESULTS")
    print("-" * 80)

    for rank, result in enumerate(results[:10], start=1):

        relevant = result_is_relevant(
            result,
            item["answer"],
        )

        print(f"\nRank {rank}")
        print(f"Relevant : {relevant}")
        print(f"Chunk ID : {result.get('chunk_id')}")
        print(f"PDF page : {result.get('page_number')}")

        if "rrf_score" in result:
            print(f"RRF score : {result['rrf_score']:.6f}")

        if "similarity_score" in result:
            print(
                f"Dense score : "
                f"{result['similarity_score']:.6f}"
            )

        if "bm25_score" in result:
            print(
                f"BM25 score : "
                f"{result['bm25_score']:.6f}"
            )

        print(
            "Text : "
            + result.get("text", "")[:500]
            .replace("\n", " ")
        )

    print()


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------

def main() -> None:

    print()
    print("=" * 80)
    print("FINRAG RETRIEVAL EVALUATION")
    print("=" * 80)

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    dataset = get_evaluation_dataset()

    print(f"Total evaluation questions: {len(dataset)}")
    print()

    # -----------------------------------------------------
    # LOAD RETRIEVER
    # -----------------------------------------------------

    print("Loading hybrid retriever...")
    print(
        f"Candidate count: {CANDIDATE_COUNT}"
    )

    retriever = HybridRetriever(
        candidate_count=CANDIDATE_COUNT
    )

    print("Retriever loaded successfully.")
    print()

    # -----------------------------------------------------
    # METRIC STORAGE
    # -----------------------------------------------------

    recall_scores = {
        k: []
        for k in TOP_K_VALUES
    }

    mrr_scores = []

    # -----------------------------------------------------
    # EVALUATE EACH QUESTION
    # -----------------------------------------------------

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        expected_answer = item["answer"]

        print()
        print(
            f"[{index}/{len(dataset)}] "
            f"Evaluating: {question}"
        )

        try:

            results = retriever.retrieve(
                question,
                top_k=CANDIDATE_COUNT,
            )

        except Exception as error:

            print(
                f"ERROR while retrieving question "
                f"{index}: {error}"
            )

            # Count retrieval failure as zero.
            for k in TOP_K_VALUES:
                recall_scores[k].append(0.0)

            mrr_scores.append(0.0)

            continue

        # -------------------------------------------------
        # RECALL@K
        # -------------------------------------------------

        for k in TOP_K_VALUES:

            score = recall_at_k(
                results,
                expected_answer,
                k,
            )

            recall_scores[k].append(score)

        # -------------------------------------------------
        # MRR
        # -------------------------------------------------

        rr = reciprocal_rank(
            results,
            expected_answer,
        )

        mrr_scores.append(rr)

        # -------------------------------------------------
        # PRINT QUESTION DETAILS
        # -------------------------------------------------

        print_result(
            index,
            item,
            results,
        )

        print(
            f"MRR contribution: {rr:.4f}"
        )

        print(
            "Recall: "
            + ", ".join(
                f"R@{k}={recall_scores[k][-1]:.0f}"
                for k in TOP_K_VALUES
            )
        )

    # -----------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 80)

    total_questions = len(dataset)

    if total_questions == 0:
        print("No evaluation questions found.")
        return

    # -----------------------------------------------------
    # RECALL
    # -----------------------------------------------------

    for k in TOP_K_VALUES:

        score = (
            sum(recall_scores[k])
            / total_questions
        )

        percentage = score * 100

        print(
            f"Recall@{k}: "
            f"{percentage:.2f}%"
        )

    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    mrr = (
        sum(mrr_scores)
        / total_questions
    )

    print(
        f"MRR: {mrr:.4f}"
    )

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    recall_5 = (
        sum(recall_scores[5])
        / total_questions
    )

    if recall_5 >= 0.90:
        print(
            "Excellent retrieval quality."
        )

    elif recall_5 >= 0.75:
        print(
            "Good retrieval quality, "
            "but some questions may still fail."
        )

    elif recall_5 >= 0.50:
        print(
            "Moderate retrieval quality. "
            "Retrieval needs improvement."
        )

    else:
        print(
            "Weak retrieval quality. "
            "The retriever needs significant improvement."
        )

    print()
    print("=" * 80)
    print("Evaluation complete.")
    print("=" * 80)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()