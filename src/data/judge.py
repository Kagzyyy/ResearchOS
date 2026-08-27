"""Pooled LLM relevance judging, to recover false negatives.

The problem this solves
-----------------------
Generated-query labels mark exactly one paper relevant - the one the query was
written from. But a question about paper A is often also answered by B and C.
Those count as misses, so absolute recall reads lower than reality.

The fix is pooling, the same technique TREC uses: take the union of the top-k
results from every configuration under test, judge only that pool, and treat
anything unjudged as non-relevant. You never judge the whole corpus, only the
few hundred documents any system actually surfaced.

Honesty boundary
----------------
The judgements here come from a model, so metrics computed against them are a
SYNTHETIC signal. They are reported as a secondary number, never as the
headline. The generated-query labels stay primary because they are structural -
the query was written from that paper - whereas these are an opinion.

Pooling is also biased toward what current systems retrieve: a paper no
configuration surfaced is never judged, so it can never be counted relevant.
That inflates all systems equally and understates how much is being missed.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.api.llm import _complete

JUDGE_SYSTEM_PROMPT = """You judge whether a research paper is relevant to a search \
query, the way a careful researcher would.

RELEVANT means: someone who typed this query would consider this paper a useful \
result. It addresses the same problem, method, or application area.

NOT_RELEVANT means: the paper is about something else. Sharing generic vocabulary \
("neural network", "training", "benchmark") is not relevance.

Be strict. Most papers in a search result are not relevant.

Answer with exactly one word: RELEVANT or NOT_RELEVANT."""


def _judge_one(query: str, title: str, abstract: str, model: str, max_tokens: int) -> int:
    user = (
        f"Query: {query}\n\n"
        f"Paper title: {title}\n"
        f"Paper abstract: {abstract[:1400]}\n\n"
        "RELEVANT or NOT_RELEVANT?"
    )
    try:
        out = _complete(JUDGE_SYSTEM_PROMPT, user, model, max_tokens, 0.0)
    except Exception:  # noqa: BLE001 - an unjudged pair is simply non-relevant
        return 0
    return 1 if "NOT_RELEVANT" not in out.upper() and "RELEVANT" in out.upper() else 0


def build_pool(
    runs: dict[str, dict[str, dict[str, float]]], depth: int = 10
) -> dict[str, set[str]]:
    """Union of the top-`depth` papers each configuration returned, per query."""
    pool: dict[str, set[str]] = {}
    for run in runs.values():
        for qid, hits in run.items():
            top = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:depth]
            pool.setdefault(qid, set()).update(pid for pid, _ in top)
    return pool


def judge_pool(
    pool: dict[str, set[str]],
    queries: dict[str, str],
    papers_by_id: dict[str, dict[str, Any]],
    base_qrels: dict[str, dict[str, int]],
    model: str,
    out_path: str | Path,
    max_tokens: int = 700,
    workers: int = 8,
) -> dict[str, dict[str, int]]:
    """Judge every pooled (query, paper) pair. Resumable; cached to disk."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, int] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[f"{rec['query_id']}|{rec['paper_id']}"] = rec["relevant"]
        print(f"[judge] resuming: {len(cache)} cached judgements")

    tasks: list[tuple[str, str]] = []
    for qid, paper_ids in pool.items():
        for pid in paper_ids:
            # The structural gold is already known - never spend a call on it.
            if pid in base_qrels.get(qid, {}):
                continue
            if f"{qid}|{pid}" in cache:
                continue
            if pid in papers_by_id:
                tasks.append((qid, pid))

    print(f"[judge] {len(tasks)} pairs to judge across {len(pool)} queries")
    lock = threading.Lock()
    handle = open(target, "a", encoding="utf-8")
    done = 0

    def work(item: tuple[str, str]) -> None:
        nonlocal done
        qid, pid = item
        meta = papers_by_id[pid]
        verdict = _judge_one(
            queries[qid], meta.get("title", ""), meta.get("abstract", ""), model, max_tokens
        )
        with lock:
            cache[f"{qid}|{pid}"] = verdict
            handle.write(
                json.dumps({"query_id": qid, "paper_id": pid, "relevant": verdict}) + "\n"
            )
            handle.flush()
            done += 1
            if done % 100 == 0:
                print(f"[judge] {done}/{len(tasks)}")

    try:
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool_exec:
                list(pool_exec.map(work, tasks))
    finally:
        handle.close()

    # Structural gold stays relevant; pooled positives are added on top.
    merged: dict[str, dict[str, int]] = {
        qid: dict(rels) for qid, rels in base_qrels.items()
    }
    added = 0
    for key, relevant in cache.items():
        qid, pid = key.split("|", 1)
        if relevant and qid in merged and pid not in merged[qid]:
            merged[qid][pid] = 1
            added += 1

    per_query = added / max(1, len(merged))
    print(f"[judge] added {added} relevant papers ({per_query:.2f} extra per query)")
    return merged
