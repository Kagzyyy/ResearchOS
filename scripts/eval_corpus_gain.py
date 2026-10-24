"""What does a bigger corpus buy? Findability of papers only it contains.

    python scripts/eval_corpus_gain.py --small configs/ml.yaml --big configs/ml.v3.yaml

calibrate_gate.py measures the cost of growing the corpus: its vague-query
targets are drawn from the small corpus, so extra papers can only hurt them
(more near-duplicates to outrank the target). This measures the other side:
targets drawn from papers ONLY the big corpus has.

The small corpus is given its production fallback - live arXiv search when
its confidence gate fires - so the comparison is "v1 as deployed" vs "v3",
not a strawman that can never find these papers. OpenAlex is left out of the
live arm: its keyless budget (~100 searches/day) is shared with the live app.

Queries are written by the same model and prompt as eval_v2_rescue.py:
half-remembered descriptions with no title words or method names.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import src.config  # noqa: F401,E402  - loads .env

from critic_localization import mcnemar_exact, wilson  # noqa: E402
from eval_v2_rescue import GEN_SYSTEM, groq, rank_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", default="configs/ml.yaml")
    ap.add_argument("--big", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--gen-model", default="qwen/qwen3.8-27b")
    ap.add_argument("--out", default="results/corpus_gain.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from src.api.ml_pipeline import MLPipeline
    from src.config import load_config

    out = Path(args.out)
    cache_path = out.with_suffix(".cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    queries = cache.setdefault("queries", {})

    big_cfg = load_config(args.big)
    big_cfg["citation"]["allow_remote"] = False
    big_cfg["diagnosis"]["enabled"] = False
    big = MLPipeline(big_cfg)
    small_ids = {json.loads(line)["paper_id"]
                 for line in open(load_config(args.small)["paths"]["papers"], encoding="utf-8")}
    pool = [p for pid, p in big.papers_by_id.items()
            if pid not in small_ids and len((p.get("abstract") or "").split()) >= 80]
    targets = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    print(f"[gain] {len(targets)} targets only the big corpus has (pool {len(pool):,})")

    for i, p in enumerate(targets, 1):
        if p["paper_id"] not in queries:
            queries[p["paper_id"]] = groq(
                GEN_SYSTEM, f"Title: {p['title']}\n\nAbstract: {p['abstract'][:1500]}",
                args.gen_model)
            if i % 10 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    rows = [{"paper_id": p["paper_id"], "query": queries[p["paper_id"]],
             "big_rank": rank_of(p["paper_id"], big.find_papers(queries[p["paper_id"]], 10)["papers"])}
            for p in targets]
    del big   # free the big index before loading the small one

    small_cfg = load_config(args.small)
    small_cfg["citation"]["live_sources"] = ["arxiv"]
    small_cfg["diagnosis"]["enabled"] = False
    small = MLPipeline(small_cfg)
    for r in rows:
        res = small.find_papers(r["query"], 10)
        r["small_rank"] = rank_of(r["paper_id"], res["papers"])
        r["small_source"] = res["source"]

    n = len(rows)
    hits = {k: [bool(r[f"{k}_rank"]) for r in rows] for k in ("small", "big")}
    summary = {"targets": n, "small_config": args.small, "big_config": args.big}
    for k in ("small", "big"):
        found = sum(hits[k])
        lo, hi = wilson(found, n)
        summary[k] = {"recall_at_10": round(found / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    summary["small"]["went_live"] = sum(r["small_source"] != "corpus" for r in rows)
    only_small, only_big, p = mcnemar_exact(hits["small"], hits["big"])
    summary["paired"] = {"only_small": only_small, "only_big": only_big, "p_exact": round(p, 6)}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
