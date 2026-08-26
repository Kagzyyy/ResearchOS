"""Reciprocal rank fusion.

RRF merges ranked lists without needing the scores to be comparable, which is
exactly the problem when combining BM25 (unbounded) with cosine (bounded).

    rrf(d) = sum over rankers of  1 / (k + rank(d))

Rank is 1-based. k damps the influence of the very top of each list; 60 is the
value from the original Cormack et al. paper and the usual default.
"""

from __future__ import annotations

Run = dict[str, dict[str, float]]


def reciprocal_rank_fusion(runs: list[Run], k: int = 60, top_k: int = 50) -> Run:
    """Fuse several runs into one. Every run must use the same query ids."""
    if not runs:
        raise ValueError("reciprocal_rank_fusion needs at least one run")

    query_ids: list[str] = []
    for run in runs:
        for qid in run:
            if qid not in query_ids:
                query_ids.append(qid)

    fused: Run = {}
    for qid in query_ids:
        totals: dict[str, float] = {}
        for run in runs:
            ranked = sorted(run.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                totals[doc_id] = totals.get(doc_id, 0.0) + 1.0 / (k + rank)
        best = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        fused[qid] = dict(best)
    return fused
