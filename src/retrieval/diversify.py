"""Per-paper result diversification.

FR4.2 in REQUIREMENTS-ML.md, and a requirement that did not exist in the parent
project. On SciFact one document was one abstract, so a top-5 was automatically
five distinct sources. Here a paper yields dozens of chunks, and the five
best-scoring chunks for "explain JEPA" are frequently five consecutive
paragraphs of the same paper - which is a straight failure for any question that
needs more than one source.

Capping chunks per paper costs a little precision on single-source questions and
buys the multi-source coverage that "explain" and "compare" require.
"""

from __future__ import annotations

from typing import Any

Run = dict[str, dict[str, float]]


def cap_per_paper(
    scored: dict[str, float],
    corpus: dict[str, dict[str, Any]],
    top_k: int,
    max_per_paper: int = 2,
) -> list[tuple[str, float]]:
    """Take the best `top_k` chunks, allowing at most `max_per_paper` from any paper.

    Greedy over the score-sorted list: keep a chunk unless its paper already has
    `max_per_paper` entries. Order within the result is still by score.
    """
    ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    seen: dict[str, int] = {}
    kept: list[tuple[str, float]] = []
    overflow: list[tuple[str, float]] = []

    for chunk_id, score in ordered:
        paper_id = corpus.get(chunk_id, {}).get("paper_id", chunk_id)
        if seen.get(paper_id, 0) < max_per_paper:
            seen[paper_id] = seen.get(paper_id, 0) + 1
            kept.append((chunk_id, score))
            if len(kept) == top_k:
                return kept
        else:
            overflow.append((chunk_id, score))

    # Corpus too small or too concentrated to fill top_k under the cap - top up
    # rather than return a short list.
    kept.extend(overflow[: top_k - len(kept)])
    return kept[:top_k]


def distinct_papers(results: list[dict[str, Any]]) -> int:
    """How many distinct papers a result set spans. Used by the US2/US3 checks."""
    return len({r.get("paper_id") for r in results if r.get("paper_id")})
