"""Rerank the cached fused top-50 and write one row of the results table.

The same script serves rows 3, 4, 5 and 6 - only the config changes.

    python scripts/run_rerank.py --config configs/rerank_offshelf.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import device, load_config, report_vram, seed_everything
from src.data.loader import load_split
from src.rerank.evaluate import evaluate_run, latency_stats, summarise, write_row


def load_reranker(cfg):
    """Build a torch or onnx reranker depending on the config backend."""
    rc = cfg["reranker"]
    backend = rc.get("backend", "torch")
    if backend == "torch":
        from src.rerank.model import Reranker

        return Reranker(
            model_name_or_path=rc["model"],
            max_length=rc["max_length"],
            batch_size=rc["batch_size"],
            device=device(),
            fp16=rc.get("fp16", True),
        )
    if backend == "onnx":
        from src.rerank.onnx_runtime import OnnxReranker

        return OnnxReranker(
            onnx_path=rc["model"],
            tokenizer_path=rc["tokenizer"],
            max_length=rc["max_length"],
            batch_size=rc["batch_size"],
        )
    raise ValueError(f"Unknown reranker backend {backend!r}")


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

    fused_path = Path(cfg["paths"]["fused_run"])
    if not fused_path.exists():
        raise SystemExit(
            f"{fused_path} not found. Run scripts/run_baseline.py first - stages are sequential."
        )
    fused = json.loads(fused_path.read_text(encoding="utf-8"))

    latency_path = fused_path.with_name(fused_path.stem + "_latency.json")
    retrieval_ms = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else {}

    reranker = load_reranker(cfg)
    print(f"[rerank] model={cfg['reranker']['model']} backend={cfg['reranker'].get('backend','torch')}")

    reranked, rerank_ms = reranker.rerank(
        queries, fused, corpus, top_k=cfg["rerank"]["top_k"]
    )
    metrics = evaluate_run(qrels, reranked, cfg["eval"]["k_values"])
    print(f"[rerank] {summarise(metrics)}")

    qids = list(fused.keys())
    end_to_end_ms = [retrieval_ms.get(qid, 0.0) + ms for qid, ms in zip(qids, rerank_ms)]

    write_row(
        cfg["row_id"],
        {
            "metrics": metrics,
            "latency": latency_stats(end_to_end_ms),
            "rerank_only_latency": latency_stats(rerank_ms),
            "params": cfg["reranker"]["params"],
            "seed": seed,
            "model": cfg["reranker"]["model"],
            "backend": cfg["reranker"].get("backend", "torch"),
        },
        cfg["paths"]["results"],
    )
    report_vram(cfg["row_id"])


if __name__ == "__main__":
    main()
