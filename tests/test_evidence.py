from src.retrieval.evidence import select_evidence


def test_evidence_preserves_both_rankers_coverage_without_duplicates():
    candidates = [{"chunk_id": str(i)} for i in range(30)]
    reranked = list(reversed(candidates))
    results = select_evidence(candidates, reranked)
    assert len(results) == 20
    assert {r["chunk_id"] for r in results} == {
        r["chunk_id"] for r in candidates[:10] + reranked[:10]
    }
    assert len(select_evidence(candidates, candidates)) == 10
