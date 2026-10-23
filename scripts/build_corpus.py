"""Merge harvested abstracts into the corpus: v1 full text + bulk abstracts.

    python scripts/build_corpus.py

The v1 corpus is 8,330 papers with full text (40k passages). The harvest
adds hundreds of thousands of abstracts. Both kinds go in one index:

  * a paper already in v1 keeps its full-text passages - the harvested
    abstract is dropped, so the same paper never appears twice
  * a new paper contributes one passage, its abstract - chunked by the same
    chunk_paper() as v1, so passage format and ids are identical
  * duplicate harvest records (a cross-listed paper seen in two sets, or a
    page re-fetched after an interrupted run) collapse to one

The output is written beside the v1 files, never over them, so the switch is
a config change and rolling back is the same change in reverse.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data.arxiv_source import Paper  # noqa: E402
from src.data.chunking import chunk_paper  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-papers", default="data/arxiv_papers.jsonl")
    ap.add_argument("--base-chunks", default="data/arxiv_chunks.jsonl")
    ap.add_argument("--abstracts", default="data/arxiv_abstracts.jsonl")
    ap.add_argument("--out-dir", default="data/corpus_v3")
    ap.add_argument("--min-words", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_ids: set[str] = set()
    with open(args.base_papers, encoding="utf-8") as src, \
            open(out / "papers.jsonl", "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                base_ids.add(json.loads(line)["paper_id"])
                dst.write(line if line.endswith("\n") else line + "\n")

    harvested: dict[str, Paper] = {}
    bad = 0
    with open(args.abstracts, encoding="utf-8") as src:
        for line in src:
            try:
                p = Paper(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                bad += 1           # a line torn by an interrupted harvest
                continue
            if p.paper_id not in base_ids:
                harvested[p.paper_id] = p          # later record wins
    new_papers = sorted(harvested.values(), key=lambda p: p.paper_id)

    n_new_chunks = 0
    with open(out / "papers.jsonl", "a", encoding="utf-8") as papers_out, \
            open(out / "chunks.jsonl", "w", encoding="utf-8") as chunks_out:
        with open(args.base_chunks, encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    chunks_out.write(line if line.endswith("\n") else line + "\n")
        for p in new_papers:
            chunks = chunk_paper(p, min_words=args.min_words)
            if not chunks:
                continue                     # abstract too short to index
            papers_out.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            for c in chunks:
                chunks_out.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n_new_chunks += len(chunks)

    by_cat: dict[str, int] = {}
    for p in new_papers:
        by_cat[p.primary_category] = by_cat.get(p.primary_category, 0) + 1
    report = {
        "base_papers": len(base_ids),
        "harvested_unique_new": len(new_papers),
        "harvest_lines_unreadable": bad,
        "new_passages": n_new_chunks,
        "new_by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
        "date_range": [new_papers[0].published if new_papers else None,
                       max((p.published for p in new_papers), default=None)],
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
