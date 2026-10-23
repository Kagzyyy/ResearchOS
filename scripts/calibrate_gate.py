"""Measure the confidence gate, and whether retrieval still finds what it used to.

    python scripts/calibrate_gate.py --config configs/ml.yaml --tag v1

The gate (`answer.min_mean_score`) decides when ResearchOS says "nothing in
here answers that". Its threshold was set by eye from a handful of queries;
a corpus 35x larger changes what "confident" means - more papers means more
near-misses for any query, which pushes every score distribution up. So this
measures it, on three query groups:

  specific   163 questions written from corpus papers (data/generated_eval.jsonl)
  vague      250 half-remembered descriptions of corpus papers (the rescue eval's)
  off-topic  questions this tool should decline

and reports each group's mean-of-top-5 distribution, the AUC separating
answerable from off-topic, the threshold that maximises Youden's J, and
recall@10 on both gold-labelled groups - the regression check that adding
papers did not bury the ones users could already find.

Live sources and diagnosis are off: deterministic, and costs nothing.
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

OFF_TOPIC = [
    "how long should sourdough bread proof in the fridge",
    "best exercises for lower back pain",
    "what caused the fall of the roman empire",
    "how to file taxes as a freelancer in the uk",
    "symptoms of vitamin d deficiency in adults",
    "how do tides work",
    "recipe for authentic pad thai",
    "how to grow tomatoes in containers",
    "who won the 1966 football world cup",
    "how does a mortgage refinance work",
    "what is the plot of pride and prejudice",
    "how to remove red wine stains from carpet",
    "treatment for a sprained ankle",
    "history of the byzantine empire",
    "how to train a puppy to sit",
    "what is the capital of mongolia",
    "how do vaccines trigger immunity",
    "best time to visit kyoto",
    "how to change a car tyre",
    "what is impressionism in painting",
    "how does compound interest work",
    "causes of the first world war",
    "how to brew pour over coffee",
    "symptoms of seasonal allergies",
    "how do volcanoes form",
    "what is a sonnet",
    "how to negotiate a salary offer",
    "how does photosynthesis work in plants",
    "rules of cricket for beginners",
    "how to knit a scarf",
    "what is the keto diet",
    "how do solar panels work at home",
    "history of jazz music",
    "how to care for succulents",
    "what is inflation and why does it happen",
    "how to write a cover letter",
    "migration patterns of monarch butterflies",
    "how to make cold brew coffee",
    "what is stoic philosophy",
    "how to repair a leaking tap",
]


def stats(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    pick = lambda q: xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 4), "p05": round(pick(0.05), 4),
            "median": round(pick(0.5), 4), "p95": round(pick(0.95), 4), "max": round(xs[-1], 4)}


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Threshold maximising Youden's J = TPR - FPR (answerable = positive)."""
    best = (0.0, -1.0)
    for t in sorted(set(pos + neg)):
        j = sum(p >= t for p in pos) / len(pos) - sum(n >= t for n in neg) / len(neg)
        if j > best[1]:
            best = (t, j)
    return round(best[0], 4), round(best[1], 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ml.yaml")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["citation"]["allow_remote"] = False
    cfg.setdefault("diagnosis", {})["enabled"] = False
    from src.api.ml_pipeline import MLPipeline
    pipe = MLPipeline(cfg)
    threshold = cfg["answer"]["min_mean_score"]

    specific = [json.loads(line) for line in open("data/generated_eval.jsonl", encoding="utf-8")]
    rescue = json.loads(Path("results/v2_rescue.cache.json").read_text(encoding="utf-8"))["queries"]
    vague = [{"query": q, "gold_paper_id": pid} for pid, q in rescue.items()]

    def run(items: list[dict]) -> tuple[list[float], float | None]:
        scores, hits = [], 0
        for it in items:
            r = pipe.find_papers(it["query"], top_k=10)
            scores.append(r["mean_top_score"])
            gold = it.get("gold_paper_id")
            hits += bool(gold and any(p["paper_id"] == gold for p in r["papers"]))
        has_gold = any(it.get("gold_paper_id") for it in items)
        return scores, (round(hits / len(items), 4) if has_gold else None)

    groups = {}
    for name, items in (("specific", specific), ("vague", vague),
                        ("off_topic", [{"query": q} for q in OFF_TOPIC])):
        scores, recall = run(items)
        groups[name] = {"scores": stats(scores), "recall_at_10": recall,
                        "pass_gate": round(sum(s >= threshold for s in scores) / len(scores), 4),
                        "_raw": scores}
        print(f"[gate] {name:<9} {groups[name]['scores']}  recall@10={recall}  "
              f"pass@{threshold}={groups[name]['pass_gate']:.1%}", flush=True)

    answerable = groups["specific"]["_raw"] + groups["vague"]["_raw"]
    off = groups["off_topic"]["_raw"]
    t, j = best_threshold(answerable, off)
    summary = {
        "tag": args.tag, "config": args.config,
        "corpus_papers": len(pipe.papers_by_id), "corpus_passages": len(pipe.corpus),
        "current_threshold": threshold,
        "auc_answerable_vs_off_topic": round(auc(answerable, off), 4),
        "auc_specific_vs_off_topic": round(auc(groups["specific"]["_raw"], off), 4),
        "youden_threshold": t, "youden_j": j,
        # The whole trade-off, not one point: share of each group passing.
        "pass_rate_by_threshold": {
            f"{th:.2f}": {k: round(sum(x >= th for x in v["_raw"]) / len(v["_raw"]), 4)
                          for k, v in groups.items()}
            for th in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35)
        },
        "raw_scores": {k: [round(x, 4) for x in v["_raw"]] for k, v in groups.items()},
        "groups": {k: {kk: vv for kk, vv in v.items() if kk != "_raw"} for k, v in groups.items()},
    }
    out = Path(args.out_dir) / f"gate_calibration_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("groups", "raw_scores")},
                     indent=2))
    print(f"[gate] saved -> {out}")


if __name__ == "__main__":
    main()
