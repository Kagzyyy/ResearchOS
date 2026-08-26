"""Is the row 4 vs row 5 difference real, or is it noise?

The headline claim of this project is a delta between two reranker arms. On a
300-query test split a small delta can easily be sampling noise, so this script
tests it properly instead of eyeballing the means:

  - per-query nDCG@10 for both arms, from pytrec_eval
  - a paired bootstrap over queries (the pairing matters: both arms see the same
    queries and the same candidate lists, so the comparison is within-query)
  - a two-sided paired permutation test

Writes results/significance.json.

    python scripts/significance.py --config configs/rerank_hard_ft.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytrec_eval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import device, load_config, seed_everything
from src.data.loader import load_split
from src.rerank.model import Reranker

METRIC = "ndcg_cut_10"


def per_query_ndcg(qrels: dict, run: dict) -> dict[str, float]:
    """nDCG@10 for every query, using the official trec_eval implementation."""
    clean_qrels = {q: {d: int(r) for d, r in rels.items()} for q, rels in qrels.items()}
    evaluator = pytrec_eval.RelevanceEvaluator(clean_qrels, {"ndcg_cut.10"})
    return {qid: scores[METRIC] for qid, scores in evaluator.evaluate(run).items()}


def paired_bootstrap(
    diffs: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Percentile CI and the fraction of resamples that flip the sign."""
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    observed = diffs.mean()
    flipped = float((means <= 0).mean() if observed > 0 else (means >= 0).mean())
    return float(lo), float(hi), flipped


def paired_permutation(diffs: np.ndarray, n_resamples: int, rng: np.random.Generator) -> float:
    """Two-sided p-value by randomly flipping the sign of each query's difference."""
    observed = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= observed).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rerank_hard_ft.yaml")
    parser.add_argument("--model-a", default="models/reranker-random/final", help="control arm")
    parser.add_argument("--model-b", default="models/reranker-hard/final", help="treatment arm")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--out", default="results/significance.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    rng = np.random.default_rng(seed)

    corpus, queries, qrels = load_split(
        cfg["dataset"]["url"], cfg["dataset"]["split"], cfg["dataset"]["download_dir"]
    )
    fused = json.loads(Path(cfg["paths"]["fused_run"]).read_text(encoding="utf-8"))
    top_k = cfg["rerank"]["top_k"]

    scores: dict[str, dict[str, float]] = {}
    for label, model_path in (("random", args.model_a), ("hard", args.model_b)):
        reranker = Reranker(
            model_name_or_path=model_path,
            max_length=cfg["reranker"]["max_length"],
            batch_size=cfg["reranker"]["batch_size"],
            device=device(),
            fp16=True,
        )
        run, _timings = reranker.rerank(queries, fused, corpus, top_k=top_k)
        scores[label] = per_query_ndcg(qrels, run)
        print(f"[sig] {label}: mean nDCG@10 = {np.mean(list(scores[label].values())):.4f}")
        del reranker

    qids = sorted(set(scores["random"]) & set(scores["hard"]))
    a = np.array([scores["random"][q] for q in qids])
    b = np.array([scores["hard"][q] for q in qids])
    diffs = b - a

    lo, hi, flip = paired_bootstrap(diffs, args.resamples, rng)
    p_value = paired_permutation(diffs, args.resamples, rng)

    wins = int((diffs > 1e-9).sum())
    losses = int((diffs < -1e-9).sum())
    ties = len(diffs) - wins - losses

    result = {
        "seed": seed,
        "n_queries": len(qids),
        "resamples": args.resamples,
        "control": {"model": args.model_a, "mean_ndcg@10": round(float(a.mean()), 5)},
        "treatment": {"model": args.model_b, "mean_ndcg@10": round(float(b.mean()), 5)},
        "mean_difference": round(float(diffs.mean()), 5),
        "bootstrap_95_ci": [round(lo, 5), round(hi, 5)],
        "permutation_p_value": round(p_value, 4),
        "per_query": {"treatment_better": wins, "control_better": losses, "tied": ties},
        "significant_at_0.05": bool(p_value < 0.05),
    }

    print(f"\n[sig] mean difference : {result['mean_difference']:+.5f}")
    print(f"[sig] 95% CI          : [{lo:+.5f}, {hi:+.5f}]")
    print(f"[sig] permutation p   : {p_value:.4f}")
    print(f"[sig] per-query       : {wins} better / {losses} worse / {ties} tied")
    print(f"[sig] significant     : {result['significant_at_0.05']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[sig] wrote {out}")


if __name__ == "__main__":
    main()
