"""Negative mining.

Two datasets are produced from exactly the same positives, in the same order,
with the same number of negatives per group. The ONLY difference is where the
negatives come from:

  hard    - passages the retriever already ranked highly for this query
            (ranks `hard_min_rank`..`top_k`), gold excluded. These look right
            and are not, so the model has to learn to discriminate.
  random  - passages drawn uniformly from the corpus, gold excluded. Almost
            always trivially unrelated.

Rows 4 and 5 of the results table differ only in which of these files is used.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.data.loader import Corpus, Qrels, passage_text

Run = dict[str, dict[str, float]]


def _ranked_ids(candidates: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)]


def build_groups(
    corpus: Corpus,
    queries: dict[str, str],
    qrels: Qrels,
    run: Run,
    num_negatives: int,
    hard_min_rank: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (hard_groups, random_groups) - parallel lists sharing positives."""
    all_doc_ids = list(corpus.keys())
    hard_groups: list[dict[str, Any]] = []
    random_groups: list[dict[str, Any]] = []
    short_pools = 0

    for qid, query_text in queries.items():
        gold_ids = [d for d, rel in qrels.get(qid, {}).items() if rel > 0 and d in corpus]
        if not gold_ids:
            continue

        candidates = _ranked_ids(run.get(qid, {}))
        # Everything the retriever surfaced at or below hard_min_rank that is not gold.
        hard_pool = [d for d in candidates[hard_min_rank - 1 :] if d not in gold_ids]

        for positive_id in gold_ids:
            if len(hard_pool) >= num_negatives:
                hard_ids = rng.sample(hard_pool, num_negatives)
            else:
                short_pools += 1
                hard_ids = list(hard_pool)
                while len(hard_ids) < num_negatives:
                    pick = rng.choice(all_doc_ids)
                    if pick not in gold_ids and pick not in hard_ids:
                        hard_ids.append(pick)

            random_ids: list[str] = []
            while len(random_ids) < num_negatives:
                pick = rng.choice(all_doc_ids)
                if pick not in gold_ids and pick not in random_ids:
                    random_ids.append(pick)

            base = {
                "query_id": qid,
                "query": query_text,
                "positive_id": positive_id,
                "positive": passage_text(corpus, positive_id),
            }
            hard_groups.append(
                {
                    **base,
                    "negative_ids": hard_ids,
                    "negatives": [passage_text(corpus, d) for d in hard_ids],
                }
            )
            random_groups.append(
                {
                    **base,
                    "negative_ids": random_ids,
                    "negatives": [passage_text(corpus, d) for d in random_ids],
                }
            )

    if short_pools:
        print(f"[mine] {short_pools} groups had a short hard pool and were topped up randomly")
    return hard_groups, random_groups


def write_jsonl(groups: list[dict[str, Any]], path: str | Path) -> None:
    """Write one JSON object per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")
    print(f"[mine] wrote {len(groups)} groups -> {target}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts."""
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
