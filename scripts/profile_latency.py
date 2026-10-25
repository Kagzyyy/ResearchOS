"""Where does a search spend its time, and what does cutting depth cost in recall?

    python scripts/profile_latency.py --config configs/ml.cpu.yaml --threads 2 --depths 30 20 15
    python scripts/profile_latency.py --config configs/ml.yaml --recall --depths 60 30 20

Latency is measured per stage - BM25, dense query encode + FAISS, rank fusion,
cross-encoder rerank - over real queries, for each first-stage depth (how many
fused candidates the reranker scores). Rerank cost is linear in depth, so it is
the lever; recall@10 on the specific and vague query sets (see calibrate_gate.py)
says what pulling it costs. `--threads N` pins torch to N CPU threads to match
a 2-vCPU Space on a bigger machine.

Run one config per process: the pipeline holds the whole index in RAM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import load_config  # noqa: E402


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ml.cpu.yaml")
    ap.add_argument("--depths", type=int, nargs="+", default=[30])
    ap.add_argument("--queries", type=int, default=20, help="queries timed per depth")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--recall", action="store_true", help="also measure recall@10 per depth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    from src.api.ml_pipeline import PSEUDO_QID, MLPipeline
    from src.retrieval.fusion import reciprocal_rank_fusion

    cfg = load_config(args.config)
    cfg["citation"]["allow_remote"] = False
    cfg.setdefault("diagnosis", {})["enabled"] = False
    t0 = time.time()
    p = MLPipeline(cfg)
    load_s = time.time() - t0

    specific = [json.loads(line) for line in open("data/generated_eval.jsonl", encoding="utf-8")]
    vague_q = json.loads(Path("results/v2_rescue.cache.json").read_text(encoding="utf-8"))["queries"]
    vague = [{"query": q, "gold_paper_id": pid} for pid, q in vague_q.items()]
    timed = [it["query"] for it in (specific[::8] + vague[::12])][: args.queries]

    report = {"config": args.config, "threads": args.threads or None, "load_s": round(load_s, 1),
              "papers": len(p.papers_by_id), "passages": len(p.corpus), "depths": {}}
    for depth in args.depths:
        p.retrieve_k = depth
        st = {"bm25": [], "dense": [], "fuse": [], "rerank": [], "total": []}
        for q in timed:
            m = {PSEUDO_QID: q}
            a = time.perf_counter()
            b = p.bm25.search(m, top_k=depth)
            t1 = time.perf_counter()
            d = p.dense.search(m, top_k=depth)
            t2 = time.perf_counter()
            fused = reciprocal_rank_fusion([b, d], k=p.rrf_k, top_k=depth)
            t3 = time.perf_counter()
            p.reranker.rerank(m, fused, p.corpus, top_k=depth)
            t4 = time.perf_counter()
            for k, v in (("bm25", t1 - a), ("dense", t2 - t1), ("fuse", t3 - t2),
                         ("rerank", t4 - t3), ("total", t4 - a)):
                st[k].append(v * 1000)
        entry = {k: {"median": round(statistics.median(v), 1), "p95": pct(v, 0.95)}
                 for k, v in st.items()}
        if args.recall:
            for name, items in (("specific", specific), ("vague", vague)):
                hit = 0
                for it in items:
                    p._rank_cache.clear()
                    res = p.find_papers(it["query"], top_k=10)
                    hit += any(x["paper_id"] == it["gold_paper_id"] for x in res["papers"])
                entry[f"recall_at_10_{name}"] = round(hit / len(items), 4)
        report["depths"][str(depth)] = entry
        print(f"[profile] depth {depth}: " + json.dumps(entry), flush=True)

    # End to end through _ranked, which is what a request runs: full rerank vs
    # the adaptive policy from the config, cache cleared per query.
    policy = cfg["retrieval"].get("early_exit")
    p.retrieve_k = cfg["retrieval"]["top_k"]
    report["end_to_end_ms"] = {}
    for label, ee in (("full_rerank", None), ("adaptive", policy)):
        if label == "adaptive" and not policy:
            continue
        cfg["retrieval"]["early_exit"] = ee
        p.early_exits = 0
        times = []
        for q in timed:
            p._rank_cache.clear()
            a = time.perf_counter()
            p._ranked(q)
            times.append((time.perf_counter() - a) * 1000)
        report["end_to_end_ms"][label] = {
            "median": round(statistics.median(times), 1), "p95": pct(times, 0.95),
            "mean": round(statistics.mean(times), 1), "early_exits": p.early_exits,
            "of": len(times)}
        print(f"[profile] end-to-end {label}: {report['end_to_end_ms'][label]}", flush=True)
    cfg["retrieval"]["early_exit"] = policy

    # The cache: same query twice, as a user retrying or a shared link would.
    p._rank_cache.clear()
    q = timed[0]
    a = time.perf_counter(); p.find_papers(q, 10); cold = (time.perf_counter() - a) * 1000
    a = time.perf_counter(); p.find_papers(q, 10); warm = (time.perf_counter() - a) * 1000
    report["repeat_query_ms"] = {"cold": round(cold, 1), "cached": round(warm, 1)}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
