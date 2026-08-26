"""Stage 6b - benchmark PyTorch vs ONNX fp32 vs ONNX INT8.

Measures per-query rerank latency over a fixed sample of test queries, each with
the same candidate list, so the only variable is the inference backend.

    python scripts/benchmark_latency.py --config configs/onnx.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import device, load_config, seed_everything
from src.data.loader import Corpus, load_split, passage_text
from src.protocols import Scorer
from src.rerank.evaluate import latency_stats


def build_pair_batches(
    queries: dict[str, str],
    fused: dict[str, dict[str, float]],
    corpus: Corpus,
    qids: list[str],
    top_k: int,
) -> list[list[tuple[str, str]]]:
    """One list of (query, passage) pairs per query, identical across backends."""
    batches: list[list[tuple[str, str]]] = []
    for qid in qids:
        ranked = sorted(fused[qid].items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        batches.append([(queries[qid], passage_text(corpus, d)) for d, _ in ranked])
    return batches


def time_backend(
    scorer: Scorer, batches: list[list[tuple[str, str]]], warmup: int
) -> dict[str, Any]:
    """Warm up, then time each query's rerank call."""
    for batch in batches[:warmup]:
        scorer.score(batch)

    timings_ms: list[float] = []
    total_pairs = 0
    start_all = time.perf_counter()
    for batch in batches:
        start = time.perf_counter()
        scorer.score(batch)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
        total_pairs += len(batch)
    wall_s = time.perf_counter() - start_all

    stats = latency_stats(timings_ms)
    stats["throughput_pairs_per_s"] = round(total_pairs / wall_s, 1)
    stats["throughput_queries_per_s"] = round(len(batches) / wall_s, 2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    ec, bc = cfg["export"], cfg["benchmark"]

    corpus, queries, _qrels = load_split(
        cfg["dataset"]["url"], cfg["dataset"]["split"], cfg["dataset"]["download_dir"]
    )
    fused = json.loads(Path(cfg["paths"]["fused_run"]).read_text(encoding="utf-8"))

    qids = list(fused.keys())[: bc["num_queries"]]
    batches = build_pair_batches(queries, fused, corpus, qids, bc["candidates_per_query"])
    print(f"[bench] {len(batches)} queries x {bc['candidates_per_query']} candidates")

    results: dict[str, Any] = {
        "seed": seed,
        "num_queries": len(batches),
        "candidates_per_query": bc["candidates_per_query"],
        "backends": {},
    }

    # --- PyTorch (GPU if available) ---
    from src.rerank.model import Reranker

    torch_model = Reranker(
        model_name_or_path=ec["source_model"],
        max_length=ec["max_length"],
        batch_size=bc["batch_size"],
        device=device(),
        fp16=device() == "cuda",
    )
    results["backends"]["pytorch"] = {
        "device": device(),
        "precision": "fp16" if device() == "cuda" else "fp32",
        **time_backend(torch_model, batches, bc["warmup_queries"]),
    }
    print(f"[bench] pytorch: {results['backends']['pytorch']}")
    del torch_model

    # --- ONNX Runtime, fp32 and int8, both on CPU ---
    from src.rerank.onnx_runtime import OnnxReranker

    for label, path in (("onnx_fp32", ec["fp32_path"]), ("onnx_int8", ec["int8_path"])):
        if not Path(path).exists():
            print(f"[bench] skipping {label}: {path} not found")
            continue
        onnx_model = OnnxReranker(
            onnx_path=path,
            tokenizer_path=ec["tokenizer_dir"],
            max_length=ec["max_length"],
            batch_size=bc["batch_size"],
        )
        results["backends"][label] = {
            "device": "cpu",
            "precision": "int8" if "int8" in label else "fp32",
            "file_size_mb": round(Path(path).stat().st_size / 1e6, 1),
            **time_backend(onnx_model, batches, bc["warmup_queries"]),
        }
        print(f"[bench] {label}: {results['backends'][label]}")
        del onnx_model

    out = Path(cfg["paths"]["latency"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()
