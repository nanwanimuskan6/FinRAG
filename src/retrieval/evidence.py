"""Keep complementary hybrid and reranker evidence for answer generation."""


def select_evidence(candidates, reranked, per_ranker: int = 10):
    """Retain both top-k lists, deduplicated by ID, with reranked results first.

    Reranking can demote a correct table row. This bounded union preserves
    either ranker's top-k coverage without passing the entire candidate pool
    to the answer model. At most 2 * per_ranker chunks are returned.
    """
    if per_ranker < 1:
        raise ValueError("per_ranker must be at least one")
    selected = {}
    for result in [*reranked[:per_ranker], *candidates[:per_ranker]]:
        selected.setdefault(result["chunk_id"], result)
    return list(selected.values())
