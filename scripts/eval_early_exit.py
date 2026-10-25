"""Can the reranker stop early when the first candidates already look confident?

    python scripts/eval_early_exit.py --config configs/ml.yaml

Reranking is the dominant search cost (linear in candidates scored; seconds on a
2-vCPU CPU). The idea: score only the first `n1` fused candidates, and if their
top-5 mean score already clears the confidence bar `tau`, stop; otherwise score
the rest. Easy queries - where the answer is obvious - then cost a fraction.

This simulates every (n1, tau) policy offline from ONE full rerank per query,
so the comparison is exact: the same scores, just a different subset available
to the policy. For each policy it reports

  * recall@10 on the specific and vague query sets - what stopping early costs
  * the share of queries that exit early, and the mean candidates scored -
    what it saves (rerank latency is proportional to the latter)
  * top-10 overlap with the full rerank - whether the user would even notice
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ml.yaml")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--out", default="results/early_exit.json")
    args = ap.parse_args()

    from src.api.ml_pipeline import PSEUDO_QID, MLPipeline
    from src.retrieval.fusion import reciprocal_rank_fusion

    cfg = load_config(args.config)
    cfg["citation"]["allow_remote"] = False
    cfg.setdefault("diagnosis", {})["enabled"] = False
    p = MLPipeline(cfg)

    specific = [json.loads(line) for line in open("data/generated_eval.jsonl", encoding="utf-8")]
    vague_q = json.loads(Path("results/v2_rescue.cache.json").read_text(encoding="utf-8"))["queries"]
    sets = {"specific": [(it["query"], it["gold_paper_id"]) for it in specific],
            "vague": [(q, pid) for pid, q in vague_q.items()]}

    # One full rerank per query: (fused order, score per chunk).
    scored: dict[str, list[tuple[list[str], dict[str, float]]]] = {}
    for name, items in sets.items():
        scored[name] = []
        for q, _ in items:
            m = {PSEUDO_QID: q}
            fused = reciprocal_rank_fusion(
                [p.bm25.search(m, top_k=args.depth), p.dense.search(m, top_k=args.depth)],
                k=p.rrf_k, top_k=args.depth)
            order = [c for c, _ in sorted(fused[PSEUDO_QID].items(), key=lambda kv: -kv[1])]
            ranked, _ = p.reranker.rerank(m, fused, p.corpus, top_k=args.depth)
            scored[name].append((order, ranked[PSEUDO_QID]))
        print(f"[early-exit] scored {len(items)} {name} queries", flush=True)

    def top_papers(scores: dict[str, float], k: int = 10) -> list[str]:
        best: dict[str, float] = {}
        for cid, s in scores.items():
            pid = p.corpus[cid]["paper_id"]
            best[pid] = max(best.get(pid, float("-inf")), s)
        return [pid for pid, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]

    def mean_top(scores: dict[str, float], n: int = 5) -> float:
        top = sorted(scores.values(), reverse=True)[:n]
        return sum(top) / len(top) if top else 0.0

    def run(n1: int | None, tau: float | None) -> dict:
        out = {}
        for name, items in sets.items():
            hits = overlap = overlap5 = early = scored_total = 0
            for (q, gold), (order, scores) in zip(items, scored[name]):
                first = {c: scores[c] for c in order[:n1] if c in scores} if n1 else scores
                stop = n1 is not None and mean_top(first) >= tau
                used = first if stop else scores
                early += stop
                scored_total += len(first) if stop else len(scores)
                hits += gold in top_papers(used)
                full10, full5 = top_papers(scores), top_papers(scores, 5)
                # Normalised by the full list's own length: a query that yields
                # fewer than 10 papers must still be able to score 1.0.
                overlap += len(set(top_papers(used)) & set(full10)) / max(len(full10), 1)
                overlap5 += len(set(top_papers(used, 5)) & set(full5)) / max(len(full5), 1)
            n = len(items)
            out[name] = {"recall_at_10": round(hits / n, 4), "early_exit": round(early / n, 3),
                         "mean_scored": round(scored_total / n, 1),
                         "top10_overlap_with_full": round(overlap / n, 3),
                         "top5_overlap_with_full": round(overlap5 / n, 3)}
        return out

    results = {"full": run(None, None)}
    for n1 in (10, 15, 20):
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
            results[f"n1={n1},tau={tau}"] = run(n1, tau)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'policy':<18}{'spec R@10':>10}{'vague R@10':>11}{'exit%':>7}{'scored':>8}"
          f"{'top10 same':>11}{'top5 same':>10}")
    for k, v in results.items():
        s, g = v["specific"], v["vague"]
        exit_all = (s["early_exit"] * 163 + g["early_exit"] * 250) / 413
        scored_all = (s["mean_scored"] * 163 + g["mean_scored"] * 250) / 413
        ov = (s["top10_overlap_with_full"] * 163 + g["top10_overlap_with_full"] * 250) / 413
        ov5 = (s["top5_overlap_with_full"] * 163 + g["top5_overlap_with_full"] * 250) / 413
        print(f"{k:<18}{s['recall_at_10']:>10.3f}{g['recall_at_10']:>11.3f}"
              f"{exit_all:>7.0%}{scored_all:>8.1f}{ov:>11.3f}{ov5:>10.3f}")


if __name__ == "__main__":
    main()
