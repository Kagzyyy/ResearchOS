"""Render results/metrics.json into the README results table.

results/metrics.json is the source of truth. This script rewrites the block
between the RESULTS markers in README.md so the two can never drift.

    python scripts/render_readme.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rerank.evaluate import ROW_LABELS, ROW_ORDER

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _fmt(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def build_table(rows: dict) -> str:
    """Markdown table in the exact row order the spec defines."""
    lines = [
        "| Setup | nDCG@10 | Recall@50 | p95 latency | Params |",
        "|---|---|---|---|---|",
    ]
    for row_id in ROW_ORDER:
        row = rows.get(row_id)
        label = ROW_LABELS[row_id]
        if not row:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        metrics = row.get("metrics", {})
        latency = row.get("latency", {})
        p95 = latency.get("p95_ms")
        lines.append(
            f"| {label} "
            f"| {_fmt(metrics.get('ndcg@10'))} "
            f"| {_fmt(metrics.get('recall@50'))} "
            f"| {'—' if p95 is None else f'{p95:.0f} ms'} "
            f"| {row.get('params') or '—'} |"
        )
    return "\n".join(lines)


def build_delta(rows: dict) -> str:
    """The headline sentence: hard negatives minus random negatives."""
    hard = rows.get("finetuned_hard_negatives", {}).get("metrics", {})
    rand = rows.get("finetuned_random_negatives", {}).get("metrics", {})
    if not hard or not rand:
        return ""
    delta = hard.get("ndcg@10", 0.0) - rand.get("ndcg@10", 0.0)
    pct = (delta / rand["ndcg@10"] * 100.0) if rand.get("ndcg@10") else 0.0
    direction = "above" if delta >= 0 else "below"
    return (
        f"\n**Headline:** training on hard negatives scores nDCG@10 "
        f"{hard.get('ndcg@10'):.4f}, {abs(delta):.4f} ({abs(pct):.1f}%) {direction} the "
        f"identical run on random negatives ({rand.get('ndcg@10'):.4f}). "
        f"Same seed, same hyperparameters, same epochs - only the negatives file differs.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} not found - run the pipeline stages first.")
    rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", {})

    block = f"{START}\n\n{build_table(rows)}\n{build_delta(rows)}\n{END}"

    readme_path = Path(args.readme)
    if not readme_path.exists():
        raise SystemExit(f"{readme_path} not found.")
    text = readme_path.read_text(encoding="utf-8")

    if START not in text or END not in text:
        raise SystemExit(f"README.md is missing the {START} / {END} markers.")

    before = text.split(START)[0]
    after = text.split(END, 1)[1]
    readme_path.write_text(before + block + after, encoding="utf-8")
    print(f"[readme] rendered {len(rows)} row(s) into {readme_path}")


if __name__ == "__main__":
    main()
