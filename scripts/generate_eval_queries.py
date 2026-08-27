"""Generate natural-language evaluation queries from papers, via Groq.

    python scripts/generate_eval_queries.py --config configs/ml.yaml --n 300

Resumable: re-running tops up an existing file rather than starting over.
Spot-check the output before trusting any metric computed from it - if the
questions read like paper titles rather than something you would type, the
benchmark is worthless and you want to know now.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.api.llm import is_configured
from src.config import load_config, seed_everything
from src.data.arxiv_source import read_papers
from src.data.chunking import read_chunks
from src.data.query_gen import generate_queries

BODY_SECTIONS = {"method", "methods", "approach", "experiments", "model", "architecture"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="data/generated_eval.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--model", default=None, help="override answer.model")
    args = parser.parse_args()

    if not is_configured():
        raise SystemExit("GROQ_API_KEY is not set. Put it in .env (UTF-8, not UTF-16).")

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    rng = random.Random(seed)
    model = args.model or cfg["answer"]["model"]

    papers = read_papers(cfg["paths"]["papers"])
    with_text = {p.paper_id for p in papers if p.full_text}

    # One method-ish chunk per paper gives the model something concrete to work
    # from beyond the abstract's marketing.
    context: dict[str, str] = {}
    for chunk in read_chunks(cfg["paths"]["chunks"]):
        if chunk.paper_id in with_text and chunk.section in BODY_SECTIONS:
            context.setdefault(chunk.paper_id, chunk.text)

    eligible = [p for p in papers if p.paper_id in context]
    print(f"[qgen] {len(eligible)} eligible papers | model={model} | target={args.n}")

    generated = generate_queries(
        papers=eligible,
        context_by_paper=context,
        n_queries=args.n,
        model=model,
        out_path=args.out,
        workers=args.workers,
        rng=rng,
    )

    print("\n[qgen] spot-check - would you type these?\n")
    for item in rng.sample(generated, min(8, len(generated))):
        print(f'  Q: "{item.query}"')
        print(f"     -> {item.gold_title[:70]}\n")


if __name__ == "__main__":
    main()
