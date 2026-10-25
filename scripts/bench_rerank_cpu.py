"""Cross-encoder cost on CPU, in isolation: how many candidates can afford to be scored?

    python scripts/bench_rerank_cpu.py --threads 2

Loads only the reranker (about 1 GB), so it measures the model rather than a
machine that is paging because a whole index is also in RAM. Reports the median
time to score 10, 15, 20 and 30 query-passage pairs, and the same at a shorter
max_length. Pairs are real abstract passages from the corpus.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ml.cpu.yaml")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/rerank_cpu.json")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from src.rerank.model import Reranker

    cfg = load_config(args.config)
    mc = cfg["reranker"]
    chunks = []
    with open(cfg["paths"]["chunks"], encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))
            if len(chunks) >= 400:
                break
    queries = ["sparse mixture of experts routing for language models",
               "how to reduce hallucination in retrieval augmented generation"]
    report = {"threads": args.threads, "device": "cpu", "results": {}}
    for max_length in (mc["max_length"], 256):
        rr = Reranker(model_name_or_path=mc["model"], max_length=max_length,
                      batch_size=mc["batch_size"], device="cpu", fp16=False)
        rr.score([(queries[0], chunks[0]["text"])])           # warm up
        for n in (10, 15, 20, 30):
            times = []
            for rep in range(args.reps):
                pairs = [(queries[rep % 2], c["title"] + " " + c["text"])
                         for c in chunks[rep * 30: rep * 30 + n]]
                t = time.perf_counter()
                rr.score(pairs)
                times.append((time.perf_counter() - t) * 1000)
            report["results"][f"max_length={max_length},n={n}"] = round(statistics.median(times))
            print(f"[rerank-cpu] max_length={max_length} n={n}: median {statistics.median(times):.0f} ms",
                  flush=True)
        del rr
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
