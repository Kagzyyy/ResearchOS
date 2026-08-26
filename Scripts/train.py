"""Stages 4 and 5 - fine-tune the reranker on one negatives file.

    python scripts/train.py --config configs/train_random.yaml
    python scripts/train.py --config configs/train_hard.yaml

The two configs differ only in which negatives file they point at.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, report_vram, seed_everything
from src.rerank.train import train_reranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    print(f"[seed] {seed}")
    print(f"[train] run={cfg['run']['name']} negatives={cfg['train']['negatives_file']}")

    start = time.perf_counter()
    final_dir = train_reranker(cfg, seed)
    minutes = (time.perf_counter() - start) / 60.0

    peak_gb = report_vram(cfg["run"]["name"])
    print(f"[train] finished in {minutes:.1f} min | peak {peak_gb:.2f} GB | model at {final_dir}")


if __name__ == "__main__":
    main()
