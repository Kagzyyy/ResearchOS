"""Stage 1 - baseline retrieval.

BM25, dense, and their RRF fusion, evaluated on the SciFact test split.
Writes rows 1 and 2 of the results table and caches the fused run for later
stages.

    python scripts/run_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import device, load_config, report_vram, seed_everything
from src.data.loader import flatten_corpus, load_split
from src.rerank.evaluate import evaluate_run, latency_stats, summarise, write_row
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.protocols import SearchRetriever
from src.retrieval.fusion import Run, reciprocal_rank_fusion


def timed_search(
    retriever: SearchRetriever, queries: dict[str, str], top_k: int
) -> tuple[Run, list[float]]:
    """Search one query at a time so the timings reflect the serving path."""
    run: Run = {}
    timings_ms: list[float] = []
    for qid, text in queries.items():
        start = time.perf_counter()
        single = retriever.search({qid: text}, top_k=top_k)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        run[qid] = single[qid]
    return run, timings_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    print(f"[seed] {seed}")

    corpus, queries, qrels = load_split(
        cfg["dataset"]["url"], cfg["dataset"]["split"], cfg["dataset"]["download_dir"]
    )
    print(f"[data] {len(corpus)} docs | {len(queries)} queries | {len(qrels)} qrels")
    doc_ids, doc_texts = flatten_corpus(corpus)

    top_k = cfg["retrieval"]["top_k"]
    k_values = cfg["eval"]["k_values"]
    cache_dir = cfg["paths"]["cache_dir"]
    results_path = cfg["paths"]["results"]

    # ---- Row 1: BM25 only -------------------------------------------------
    bm25_cfg = cfg["retrieval"]["bm25"]
    bm25 = BM25Retriever(k1=bm25_cfg["k1"], b=bm25_cfg["b"], stemmer_lang=bm25_cfg["stemmer"])
    bm25.index(doc_ids, doc_texts, cache_dir=cache_dir)
    bm25_run, bm25_ms = timed_search(bm25, queries, top_k)
    bm25_metrics = evaluate_run(qrels, bm25_run, k_values)
    print(f"[bm25] {summarise(bm25_metrics)}")
    write_row(
        "bm25",
        {"metrics": bm25_metrics, "latency": latency_stats(bm25_ms), "params": None, "seed": seed},
        results_path,
    )

    # ---- Row 2: BM25 + dense, fused with RRF ------------------------------
    dense_cfg = cfg["retrieval"]["dense"]
    dense = DenseRetriever(
        model_name=dense_cfg["model"],
        batch_size=dense_cfg["batch_size"],
        max_length=dense_cfg["max_length"],
        query_prefix=dense_cfg["query_prefix"],
        device=device(),
    )
    dense.index_corpus(doc_ids, doc_texts, cache_dir=cache_dir)
    dense_run, dense_ms = timed_search(dense, queries, top_k)
    dense_metrics = evaluate_run(qrels, dense_run, k_values)
    print(f"[dense] {summarise(dense_metrics)}")

    rrf_k = cfg["retrieval"]["fusion"]["rrf_k"]
    fuse_start = time.perf_counter()
    fused = reciprocal_rank_fusion([bm25_run, dense_run], k=rrf_k, top_k=top_k)
    fuse_ms_per_query = (time.perf_counter() - fuse_start) * 1000.0 / max(1, len(queries))

    fused_metrics = evaluate_run(qrels, fused, k_values)
    fused_ms = [b + d + fuse_ms_per_query for b, d in zip(bm25_ms, dense_ms)]
    print(f"[rrf ] {summarise(fused_metrics)}")
    write_row(
        "bm25_dense_rrf",
        {
            "metrics": fused_metrics,
            "latency": latency_stats(fused_ms),
            "params": dense_cfg["params"],
            "seed": seed,
            "dense_only_metrics": dense_metrics,
        },
        results_path,
    )

    fused_path = Path(cfg["paths"]["fused_run"])
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    fused_path.write_text(json.dumps(fused), encoding="utf-8")
    print(f"[cache] fused top-{top_k} run -> {fused_path}")

    # Per-query retrieval timings, so later stages can report end-to-end latency.
    latency_path = fused_path.with_name(fused_path.stem + "_latency.json")
    latency_path.write_text(
        json.dumps({qid: ms for qid, ms in zip(queries.keys(), fused_ms)}), encoding="utf-8"
    )
    print(f"[cache] per-query retrieval latency -> {latency_path}")

    report_vram("baseline")


if __name__ == "__main__":
    main()
