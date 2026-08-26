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
NL = "\n"


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
    return NL.join(lines)


def build_delta(rows: dict, sig_path: str = "results/significance.json") -> str:
    """The headline findings, with the significance test attached.

    A delta reported without its confidence interval is a claim, not a result.
    """
    hard = rows.get("finetuned_hard_negatives", {}).get("metrics", {})
    rand = rows.get("finetuned_random_negatives", {}).get("metrics", {})
    off = rows.get("offshelf_reranker", {}).get("metrics", {})
    if not hard or not rand:
        return ""

    parts: list[str] = []

    if off:
        gain = hard.get("ndcg@10", 0.0) - off.get("ndcg@10", 0.0)
        parts.append(
            f"**Fine-tuning works.** The fine-tuned reranker scores nDCG@10 "
            f"{hard.get('ndcg@10'):.4f} against {off.get('ndcg@10'):.4f} off the shelf "
            f"({gain:+.4f}), and the off-the-shelf model is itself *below* the "
            f"retrieval baseline it is meant to improve."
        )

    delta = hard.get("ndcg@10", 0.0) - rand.get("ndcg@10", 0.0)
    sig_file = Path(sig_path)
    if sig_file.exists():
        sig = json.loads(sig_file.read_text(encoding="utf-8"))
        low, high = sig["bootstrap_95_ci"]
        verdict = (
            "statistically significant"
            if sig["significant_at_0.05"]
            else "NOT statistically significant"
        )
        per_query = sig["per_query"]
        parts.append(
            f"**Hard vs. random negatives: {delta:+.4f} nDCG@10 — {verdict}.** "
            f"Paired permutation p = {sig['permutation_p_value']:.3f}; 95% bootstrap CI "
            f"[{low:+.4f}, {high:+.4f}] over {sig['n_queries']} queries. Per query: "
            f"{per_query['treatment_better']} better, {per_query['control_better']} worse, "
            f"{per_query['tied']} identical. Same seed, same hyperparameters, same epochs — "
            f"only the negatives file differs. Reproduce with `scripts/significance.py`."
        )
    else:
        parts.append(
            f"**Hard vs. random negatives:** {delta:+.4f} nDCG@10 "
            f"(no significance test has been run)."
        )

    return NL + (NL * 2).join(parts) + NL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--significance", default="results/significance.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} not found - run the pipeline stages first.")
    rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", {})

    block = START + NL * 2 + build_table(rows) + NL + build_delta(rows, args.significance) + NL + END

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
