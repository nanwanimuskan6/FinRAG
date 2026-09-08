"""Compare candidate coverage, hybrid ranking and reranking on labelled questions.

AnswerHit@K measures answer occurrence; PageHit@K separately checks page labels.
Neither metric establishes that a generated answer is correct.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.evaluation.eval_dataset import get_evaluation_dataset
from src.retrieval.evidence import select_evidence

TOP_K_VALUES = [1, 3, 5, 10]
CANDIDATE_COUNT = 100


def normalize_text(text: str) -> str:
    text = str(text).lower().replace(",", "").replace("\u20b9", "").replace("`", "")
    text = re.sub(r"\b(?:crores?|cr)\b", "", text)
    return " ".join(text.split())


def result_is_relevant(result: dict[str, Any], expected_answer: str) -> bool:
    expected = normalize_text(expected_answer)
    if not expected:
        return False
    # Preserve decimal points and require complete numeric/token boundaries.
    return re.search(r"(?<![\w.])" + re.escape(expected) + r"(?![\w.]|\d)",
                     normalize_text(result.get("text", ""))) is not None


def reciprocal_rank(results, expected_answer: str) -> float:
    return next((1 / rank for rank, result in enumerate(results, 1)
                 if result_is_relevant(result, expected_answer)), 0.0)


def recall_at_k(results, expected_answer: str, k: int) -> float:
    """Legacy helper: this binary measure is answer hit rate, not passage recall."""
    if k < 1:
        raise ValueError("k must be at least one")
    return float(any(result_is_relevant(r, expected_answer) for r in results[:k]))


def metrics(results, item):
    return {
        **{f"answer_hit@{k}": recall_at_k(results, item["answer"], k)
           for k in TOP_K_VALUES},
        **{f"page_hit@{k}": float(any(r["page_number"] in item.get("pages", [])
                                     for r in results[:k])) for k in TOP_K_VALUES},
        "mrr@10": reciprocal_rank(results[:10], item["answer"]),
        **{f"evidence_hit@{k}": evidence_hit(results[:k], item) for k in TOP_K_VALUES},
    }


def evidence_hit(results, item):
    """For derived answers require every labelled operand in the evidence set."""
    return float(all(any(result_is_relevant(r, answer) for r in results)
                     for answer in item.get("evidence_answers", [item["answer"]])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_COUNT,
                        help="Candidates from each ranker; retain their full union")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/processed/retrieval_metrics.json"))
    args = parser.parse_args()
    if args.candidates < 1:
        parser.error("--candidates must be positive")

    from src.retrieval.dense_retriever import ChromaDenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker

    retriever = HybridRetriever(dense_retriever=ChromaDenseRetriever(),
                                candidate_count=args.candidates)
    reranker = None if args.skip_rerank else Reranker()
    rows = []
    for index, item in enumerate(get_evaluation_dataset(), 1):
        print(f"[{index}] {item['question']}", flush=True)
        candidates = retriever.retrieve(item["question"], top_k=2 * args.candidates)
        row = {"question": item["question"], "expected_answer": item["answer"],
               "expected_pages": item["pages"], "candidate_count": len(candidates),
               "candidate_answer_hit": recall_at_k(candidates, item["answer"], max(1, len(candidates))),
               "candidate_evidence_hit": evidence_hit(candidates, item),
               "candidates": candidates,
               "hybrid": metrics(candidates, item)}
        if reranker:
            results = reranker.rerank(item["question"], candidates, top_k=10)
            row["reranked"] = metrics(results, item)
            row["top_results"] = results
            context = select_evidence(candidates, results)
            row["context_evidence_hit"] = evidence_hit(context, item)
            row["context_count"] = len(context)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in {"top_results", "candidates"}}), flush=True)

    summary = {}
    if rows:
        summary["candidate_answer_hit"] = sum(r["candidate_answer_hit"] for r in rows) / len(rows)
        summary["candidate_evidence_hit"] = sum(r["candidate_evidence_hit"] for r in rows) / len(rows)
        if reranker:
            summary["context_evidence_hit"] = sum(r["context_evidence_hit"] for r in rows) / len(rows)
        for stage in ("hybrid", "reranked"):
            if stage in rows[0]:
                summary[stage] = {key: sum(r[stage][key] for r in rows) / len(rows)
                                  for key in rows[0][stage]}
    report = {"configuration": {"candidates_per_ranker": args.candidates,
                                "reranking": not args.skip_rerank},
              "questions": len(rows), "summary": summary, "details": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
