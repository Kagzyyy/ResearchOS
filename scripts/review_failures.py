"""Read the failure log. Run this weekly once people are using the app.

    python scripts/review_failures.py --config configs/ml.yaml

Five things worth looking at, in the order they usually matter:

  errors          exceptions. Anything here is a bug, not a limitation.
  downvoted       someone said the answer was wrong. Read the query.
  out_of_corpus   the retriever declined. If a topic keeps appearing, that is
                  the corpus telling you what to ingest next.
  never_clicked   confident answers nobody opened a citation for. Weak signal
                  individually; a pattern means results look irrelevant.
  slowest         where the latency actually goes.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.api.db import FeedbackLog
from src.config import load_config


def section(title: str, rows: list[dict], columns: list[str], limit: int) -> None:
    print(f"\n{'=' * 78}\n{title}  ({len(rows)})\n{'=' * 78}")
    if not rows:
        print("  nothing recorded")
        return
    for row in rows[:limit]:
        parts = [str(row.get(c, ""))[:96] for c in columns if row.get(c) not in (None, "")]
        print("  " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = FeedbackLog(cfg["paths"]["feedback_db"])

    stats = log.stats()
    print("\nUSAGE")
    for key, value in stats.items():
        print(f"  {key:16s} {value}")

    total = stats["queries"]
    if total:
        low = stats["low_confidence"] / total * 100
        clicked = stats["clicks"] / total
        print(f"\n  {low:.0f}% of queries were out-of-corpus")
        print(f"  {clicked:.2f} citation clicks per query")
        if stats["votes_up"] + stats["votes_down"]:
            up = stats["votes_up"] / (stats["votes_up"] + stats["votes_down"]) * 100
            print(f"  {up:.0f}% of votes were positive")

    data = log.recent_failures(limit=200)
    section("ERRORS - these are bugs", data["errors"],
            ["ts", "endpoint", "error_type", "message"], args.limit)
    section("DOWNVOTED - the answer was wrong", data["downvoted"],
            ["ts", "mode", "query", "comment"], args.limit)
    section("OUT OF CORPUS - what people want that you have not ingested",
            data["out_of_corpus"], ["ts", "mode", "query", "mean_top_score"], args.limit)
    section("CONFIDENT BUT NO CLICKS - results may look irrelevant",
            data["never_clicked"], ["ts", "mode", "query"], args.limit)
    section("SLOWEST", data["slowest"], ["latency_ms", "mode", "query"], args.limit)

    if data["out_of_corpus"]:
        words = collections.Counter()
        for row in data["out_of_corpus"]:
            for word in str(row.get("query", "")).lower().split():
                if len(word) > 4:
                    words[word] += 1
        common = words.most_common(10)
        if common:
            print("\nMost common terms in out-of-corpus queries - candidate ingest topics:")
            print("  " + ", ".join(f"{w} ({n})" for w, n in common))
    print()


if __name__ == "__main__":
    main()
