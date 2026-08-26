"""Stage 3 - build hard_negatives.jsonl and random_negatives.jsonl.

    python scripts/mine_negatives.py --config configs/mine_negatives.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import device, load_config, report_vram, seed_everything
from src.data.loader import flatten_corpus, load_split
from src.data.negatives import build_groups, write_jsonl
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    rng = random.Random(seed)
    print(f"[seed] {seed}")

    corpus, queries, qrels = load_split(
        cfg["dataset"]["url"], cfg["dataset"]["split"], cfg["dataset"]["download_dir"]
    )
    print(f"[data] train split: {len(queries)} queries | {len(corpus)} docs")
    doc_ids, doc_texts = flatten_corpus(corpus)

    top_k = cfg["retrieval"]["top_k"]
    cache_dir = cfg["paths"]["cache_dir"]
    run_path = Path(cfg["paths"]["train_run"])

    if run_path.exists():
        fused = json.loads(run_path.read_text(encoding="utf-8"))
        print(f"[cache] reusing {run_path}")
    else:
        bm25_cfg = cfg["retrieval"]["bm25"]
        bm25 = BM25Retriever(k1=bm25_cfg["k1"], b=bm25_cfg["b"], stemmer_lang=bm25_cfg["stemmer"])
        bm25.index(doc_ids, doc_texts, cache_dir=cache_dir)
        bm25_run = bm25.search(queries, top_k=top_k)

        dense_cfg = cfg["retrieval"]["dense"]
        dense = DenseRetriever(
            model_name=dense_cfg["model"],
            batch_size=dense_cfg["batch_size"],
            max_length=dense_cfg["max_length"],
            query_prefix=dense_cfg["query_prefix"],
            device=device(),
        )
        dense.index_corpus(doc_ids, doc_texts, cache_dir=cache_dir)
        dense_run = dense.search(queries, top_k=top_k)

        fused = reciprocal_rank_fusion(
            [bm25_run, dense_run], k=cfg["retrieval"]["fusion"]["rrf_k"], top_k=top_k
        )
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(fused), encoding="utf-8")
        print(f"[cache] train run -> {run_path}")

    hard, rand = build_groups(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        run=fused,
        num_negatives=cfg["mining"]["num_negatives"],
        hard_min_rank=cfg["mining"]["hard_min_rank"],
        rng=rng,
    )

    # The ablation is only meaningful if these two datasets are identical
    # everywhere except the negatives.
    assert len(hard) == len(rand), "hard and random datasets must be the same size"
    for h, r in zip(hard, rand):
        assert h["positive_id"] == r["positive_id"], "positives must match across datasets"
        assert len(h["negatives"]) == len(r["negatives"]), "negative counts must match"

    write_jsonl(hard, cfg["paths"]["hard_negatives"])
    write_jsonl(rand, cfg["paths"]["random_negatives"])
    report_vram("mine_negatives")


if __name__ == "__main__":
    main()
