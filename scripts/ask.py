"""ResearchOS-ML - ask the corpus a question from the terminal.

A CLI rather than a server, because the tool-first phase needs a fast feedback
loop more than it needs HTTP.

    python scripts/ask.py --config configs/ml.yaml "explain JEPA"
    python scripts/ask.py "how does JEPA differ from MAE?" --no-generate
    python scripts/ask.py --interactive
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.ml_pipeline import MLPipeline
from src.config import load_config, seed_everything


def render(result: dict[str, Any], width: int = 96, snippet: int = 320) -> None:
    """Print an answer and its citations."""
    print()
    if result["answer"]:
        for para in result["answer"].split("\n"):
            print(textwrap.fill(para, width=width) if para.strip() else "")
    print()
    print("-" * width)
    flag = "" if result["confident"] else "  [LOW CONFIDENCE]"
    print(
        f"top score {result['top_score']}  |  {result['distinct_papers']} distinct papers  |  "
        f"{result['timing_ms']['total']:.0f} ms{flag}"
    )
    print("-" * width)

    for i, c in enumerate(result["citations"], 1):
        print(f"\n[{i}] {c['score']:+.2f}  {c['title'][:width - 14]}")
        print(f"     {c['arxiv_url']}  ({c['published']}, section: {c['section']})")
        body = c["text"][:snippet] + ("..." if len(c["text"]) > snippet else "")
        print(textwrap.fill(body, width=width - 5, initial_indent="     ", subsequent_indent="     "))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--no-generate", action="store_true", help="passages only")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    pipeline = MLPipeline(cfg)
    generate = not args.no_generate

    if args.interactive:
        print("Ask a question, or an empty line to quit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                break
            render(pipeline.answer(question, generate=generate))
        return

    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("Give a question, or use --interactive.")
    render(pipeline.answer(question, generate=generate))


if __name__ == "__main__":
    main()
