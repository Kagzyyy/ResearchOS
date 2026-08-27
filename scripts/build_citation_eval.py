"""Mine a retrieval evaluation set from citation contexts.

    python scripts/build_citation_eval.py --config configs/ml.yaml
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, seed_everything
from src.data.arxiv_source import read_papers
from src.data.citation_eval import build_pairs, write_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--out", default="data/citation_eval.jsonl")
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--max-per-paper", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    papers = read_papers(cfg["paths"]["papers"])
    corpus_papers = {p.paper_id: vars(p) for p in papers}
    with_text = [p for p in papers if p.full_text]
    print(f"[eval] {len(papers)} papers, {len(with_text)} with full text")

    pairs = build_pairs(
        with_text,
        corpus_papers,
        match_threshold=args.threshold,
        max_per_paper=args.max_per_paper,
    )

    if not pairs:
        raise SystemExit(
            "No citation pairs found. The corpus skews recent, so most references "
            "point outside it. Try a lower --threshold or ingest more papers."
        )

    golds = collections.Counter(p.gold_paper_id for p in pairs)
    citers = collections.Counter(p.citing_paper_id for p in pairs)
    print(f"[eval] {len(pairs)} pairs | {len(golds)} distinct gold papers | {len(citers)} citing papers")
    print(f"[eval] median query length: {sorted(len(p.query.split()) for p in pairs)[len(pairs) // 2]} words")
    print("[eval] most-cited golds:")
    for pid, n in golds.most_common(5):
        title = next(p.gold_title for p in pairs if p.gold_paper_id == pid)
        print(f"         {n:>3}x  {pid}  {title[:58]}")

    write_pairs(pairs, args.out)
    print("\n[eval] sample:")
    for pair in pairs[:3]:
        print(f'  Q: "{pair.query[:96]}"')
        print(f"  A: {pair.gold_paper_id}  {pair.gold_title[:64]}\n")


if __name__ == "__main__":
    main()
