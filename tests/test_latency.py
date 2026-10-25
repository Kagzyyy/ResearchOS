"""Tests for the search-latency paths: adaptive reranking and the query cache.

Offline - BM25, dense search and the cross-encoder are replaced by fakes that
record what they were asked, so what is under test is the control flow.

Run: `python tests/test_latency.py`
"""

from __future__ import annotations

import sys
import threading
import types
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api.ml_pipeline as mp  # noqa: E402

Q = mp.PSEUDO_QID


class FakeReranker:
    """Scores chunk 'cN' as table[N]; records every batch it is given."""

    def __init__(self, score_of):
        self.score_of = score_of
        self.batches: list[list[str]] = []

    def rerank(self, queries, run, corpus, top_k=50):
        out = {}
        for qid, cands in run.items():
            ids = sorted(cands, key=cands.get, reverse=True)[:top_k]
            self.batches.append(ids)
            out[qid] = {c: self.score_of(c) for c in ids}
        return out, []


def _pipeline(score_of, n=30, early=None):
    fused = {f"c{i}": 1.0 - i / 100 for i in range(n)}      # c0 best ... cN-1 worst
    fake = types.SimpleNamespace(
        cfg={"retrieval": {"early_exit": early}},
        retrieve_k=n, rrf_k=60, corpus={},
        reranker=FakeReranker(score_of),
        bm25=types.SimpleNamespace(search=lambda m, top_k: {Q: {}}),
        dense=types.SimpleNamespace(search=lambda m, top_k: {Q: {}}),
        _gpu_lock=threading.Lock(), _cache_lock=threading.Lock(),
        _rank_cache=OrderedDict(), cache_hits=0, early_exits=0,
    )
    fake._rerank_adaptive = types.MethodType(mp.MLPipeline._rerank_adaptive, fake)
    fake._fused = fused
    return fake


EARLY = {"first": 15, "tau": 0.6}


def test_confident_head_stops_after_one_batch():
    p = _pipeline(lambda c: 0.9, early=EARLY)
    scores = p._rerank_adaptive({Q: "q"}, p._fused)
    assert len(p.reranker.batches) == 1 and len(p.reranker.batches[0]) == 15
    assert len(scores) == 15 and p.early_exits == 1


def test_unsure_head_scores_everything():
    p = _pipeline(lambda c: 0.1, early=EARLY)
    scores = p._rerank_adaptive({Q: "q"}, p._fused)
    assert [len(b) for b in p.reranker.batches] == [15, 15]
    assert set(scores) == set(p._fused), "a hard query must still see every candidate"
    assert p.early_exits == 0


def test_second_batch_is_exactly_the_remainder():
    p = _pipeline(lambda c: 0.1, early=EARLY)
    p._rerank_adaptive({Q: "q"}, p._fused)
    first, second = p.reranker.batches
    assert not set(first) & set(second)
    assert first == [f"c{i}" for i in range(15)], "head = the best fused candidates"


def test_late_strong_candidates_are_not_lost_when_head_is_weak():
    """A great candidate at fused rank 25 must surface when the head is poor."""
    p = _pipeline(lambda c: 0.99 if c == "c25" else 0.1, early=EARLY)
    scores = p._rerank_adaptive({Q: "q"}, p._fused)
    assert max(scores, key=scores.get) == "c25"


def test_disabled_or_small_candidate_set_scores_in_one_batch():
    p = _pipeline(lambda c: 0.9, early=None)
    assert len(p._rerank_adaptive({Q: "q"}, p._fused)) == 30
    assert len(p.reranker.batches) == 1
    p = _pipeline(lambda c: 0.9, n=10, early=EARLY)
    assert len(p._rerank_adaptive({Q: "q"}, p._fused)) == 10
    assert len(p.reranker.batches) == 1


def test_mean_of_top_five_not_top_one_decides():
    """One lucky 0.99 among weak scores must not trigger an early exit."""
    p = _pipeline(lambda c: 0.99 if c == "c0" else 0.2, early=EARLY)
    p._rerank_adaptive({Q: "q"}, p._fused)
    assert len(p.reranker.batches) == 2


def test_repeated_query_skips_search_and_rerank():
    p = _pipeline(lambda c: 0.9, early=EARLY)
    p.bm25 = types.SimpleNamespace(search=lambda m, top_k: {Q: {f"c{i}": 1.0 for i in range(30)}})
    p.dense = types.SimpleNamespace(search=lambda m, top_k: {Q: {f"c{i}": 1.0 for i in range(30)}})
    first = mp.MLPipeline._ranked(p, "How do Transformers   work?")
    batches = len(p.reranker.batches)
    again = mp.MLPipeline._ranked(p, "how do transformers work?")      # case/space-insensitive
    assert again == first and len(p.reranker.batches) == batches, "cache hit must not rerank"
    assert p.cache_hits == 1


def test_cached_result_is_a_copy():
    """Callers mutate scores; that must not corrupt the cache."""
    p = _pipeline(lambda c: 0.9, early=EARLY)
    p.bm25 = types.SimpleNamespace(search=lambda m, top_k: {Q: {f"c{i}": 1.0 for i in range(30)}})
    p.dense = types.SimpleNamespace(search=lambda m, top_k: {Q: {f"c{i}": 1.0 for i in range(30)}})
    a = mp.MLPipeline._ranked(p, "q one")
    a.clear()
    assert mp.MLPipeline._ranked(p, "q one"), "second read must be intact"


def test_cache_is_bounded():
    p = _pipeline(lambda c: 0.9, early=EARLY)
    p.bm25 = types.SimpleNamespace(search=lambda m, top_k: {Q: {"c0": 1.0}})
    p.dense = types.SimpleNamespace(search=lambda m, top_k: {Q: {"c0": 1.0}})
    saved = mp.RANK_CACHE_MAX
    mp.RANK_CACHE_MAX = 3
    try:
        for i in range(10):
            mp.MLPipeline._ranked(p, f"query {i}")
        assert len(p._rank_cache) == 3 and "query 9" in p._rank_cache
    finally:
        mp.RANK_CACHE_MAX = saved


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
