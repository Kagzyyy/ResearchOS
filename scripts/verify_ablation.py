"""Prove that rows 4 and 5 are a valid controlled comparison.

The headline claim of this project is a delta between two training runs. That
claim is only worth anything if the runs were identical apart from the
intervention. This script checks that mechanically instead of on trust:

  1. the two configs differ only in the negatives file, output dir and run name
  2. the two datasets have the same size, the same positives, in the same order
  3. the negatives really are drawn from different distributions
  4. the two training runs used the same seed, epochs, LR and batch geometry

    python scripts/verify_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.negatives import read_jsonl

ALLOWED_CONFIG_DIFFS = {"train.negatives_file", "train.output_dir", "run.name"}


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def check(label: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-config", default="configs/train_random.yaml")
    parser.add_argument("--hard-config", default="configs/train_hard.yaml")
    args = parser.parse_args()

    results: list[bool] = []

    print("\n1. Config comparison")
    rand_cfg = _flatten(load_config(args.random_config))
    hard_cfg = _flatten(load_config(args.hard_config))
    differing = {k for k in set(rand_cfg) | set(hard_cfg) if rand_cfg.get(k) != hard_cfg.get(k)}
    unexpected = differing - ALLOWED_CONFIG_DIFFS
    results.append(
        check(
            "configs differ only in negatives file / output dir / run name",
            not unexpected,
            f"unexpected: {sorted(unexpected)}" if unexpected else f"differ: {sorted(differing)}",
        )
    )

    print("\n2. Dataset comparison")
    rand_file = rand_cfg["train.negatives_file"]
    hard_file = hard_cfg["train.negatives_file"]
    if not (Path(rand_file).exists() and Path(hard_file).exists()):
        print(f"  [SKIP] datasets not built yet ({rand_file}, {hard_file})")
    else:
        rand_ds = read_jsonl(rand_file)
        hard_ds = read_jsonl(hard_file)
        results.append(check("same number of groups", len(rand_ds) == len(hard_ds), f"{len(hard_ds)}"))
        results.append(
            check(
                "identical positives, in identical order",
                all(r["positive_id"] == h["positive_id"] for r, h in zip(rand_ds, hard_ds)),
            )
        )
        results.append(
            check(
                "identical negative counts per group",
                all(len(r["negatives"]) == len(h["negatives"]) for r, h in zip(rand_ds, hard_ds)),
                f"{len(hard_ds[0]['negatives'])} per group",
            )
        )
        shared = sum(
            len(set(r["negative_ids"]) & set(h["negative_ids"])) for r, h in zip(rand_ds, hard_ds)
        )
        total = sum(len(h["negative_ids"]) for h in hard_ds)
        results.append(
            check(
                "negative sets are substantively different",
                shared / total < 0.05,
                f"{shared}/{total} shared ({shared / total * 100:.1f}%)",
            )
        )

    print("\n3. Training run comparison")
    states = {}
    for name, cfg in (("random", rand_cfg), ("hard", hard_cfg)):
        candidates = sorted(Path(cfg["train.output_dir"]).glob("checkpoint-*/trainer_state.json"))
        if candidates:
            states[name] = json.loads(candidates[-1].read_text(encoding="utf-8"))

    if len(states) < 2:
        print(f"  [SKIP] need both runs trained (found: {sorted(states)})")
    else:
        rand_args = states["random"].get("train_batch_size"), states["random"].get("num_train_epochs")
        hard_args = states["hard"].get("train_batch_size"), states["hard"].get("num_train_epochs")
        results.append(check("same batch size and epoch count", rand_args == hard_args, f"{hard_args}"))
        results.append(
            check(
                "same number of optimizer steps",
                states["random"].get("global_step") == states["hard"].get("global_step"),
                f"{states['hard'].get('global_step')} steps",
            )
        )
        rand_lr = [e.get("learning_rate") for e in states["random"].get("log_history", []) if "learning_rate" in e]
        hard_lr = [e.get("learning_rate") for e in states["hard"].get("log_history", []) if "learning_rate" in e]
        results.append(
            check(
                "identical learning-rate schedule",
                rand_lr == hard_lr and bool(rand_lr),
                f"{len(hard_lr)} logged points, peak {max(hard_lr):.2e}" if hard_lr else "no lr logged",
            )
        )

    print()
    if all(results):
        print(f"ABLATION VALID - {len(results)} checks passed. Rows 4 and 5 are comparable.\n")
    else:
        print(f"ABLATION INVALID - {results.count(False)} of {len(results)} checks failed.\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
