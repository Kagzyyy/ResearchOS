"""Evaluation with the official BEIR evaluator, and the results ledger.

results/metrics.json is the source of truth for the README table. Nothing in
this project is allowed to be print-only.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from beir.retrieval.evaluation import EvaluateRetrieval

Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]

# Row order in the README table. Keys must match what the scripts write.
ROW_ORDER = [
    "bm25",
    "bm25_dense_rrf",
    "offshelf_reranker",
    "finetuned_random_negatives",
    "finetuned_hard_negatives",
    "finetuned_onnx_int8",
]

ROW_LABELS = {
    "bm25": "BM25 only",
    "bm25_dense_rrf": "BM25 + dense (RRF)",
    "offshelf_reranker": "+ off-the-shelf reranker",
    "finetuned_random_negatives": "+ fine-tuned, random negatives",
    "finetuned_hard_negatives": "+ fine-tuned, hard negatives",
    "finetuned_onnx_int8": "Fine-tuned, ONNX INT8",
}


def evaluate_run(qrels: Qrels, run: Run, k_values: list[int]) -> dict[str, float]:
    """Run the BEIR evaluator and flatten its output to a single dict."""
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run, k_values)
    flat: dict[str, float] = {}
    for block in (ndcg, _map, recall, precision):
        for key, value in block.items():
            flat[key.lower().replace("_", "")] = round(float(value), 5)
    return flat


def latency_stats(timings_ms: list[float]) -> dict[str, float]:
    """p50/p95/mean over per-query wall-clock timings."""
    if not timings_ms:
        return {}
    ordered = sorted(timings_ms)
    return {
        "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 2),
        "mean_ms": round(sum(ordered) / len(ordered), 2),
        "n_queries": len(ordered),
    }


def write_row(row_id: str, payload: dict[str, Any], path: str = "results/metrics.json") -> None:
    """Merge one row into the metrics ledger, preserving the other rows."""
    if row_id not in ROW_LABELS:
        raise ValueError(f"Unknown row id {row_id!r}; expected one of {list(ROW_LABELS)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, Any] = {"rows": {}}
    if target.exists():
        ledger = json.loads(target.read_text(encoding="utf-8"))
        ledger.setdefault("rows", {})
    ledger["rows"][row_id] = {"label": ROW_LABELS[row_id], **payload}
    ordered = {rid: ledger["rows"][rid] for rid in ROW_ORDER if rid in ledger["rows"]}
    ledger["rows"] = ordered
    target.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"[results] wrote row {row_id!r} to {target}")


def read_rows(path: str = "results/metrics.json") -> dict[str, Any]:
    """Read the ledger back, or an empty dict if it does not exist yet."""
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8")).get("rows", {})


def summarise(metrics: dict[str, float]) -> str:
    """One-line headline for logs and commit messages."""
    return f"nDCG@10 {metrics.get('ndcg@10', float('nan')):.4f} | Recall@50 {metrics.get('recall@50', float('nan')):.4f}"
