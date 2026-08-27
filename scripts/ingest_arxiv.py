"""ResearchOS-ML stage 1 - build the arXiv corpus.

Fetches metadata for the configured queries, dedups by version, optionally
downloads and extracts full text, chunks everything, and writes both papers and
chunks to disk.

Resumable: PDFs and extracted text are cached by arXiv id, so re-running skips
anything already fetched. Safe to interrupt.

    python scripts/ingest_arxiv.py --config configs/ml.yaml
    python scripts/ingest_arxiv.py --config configs/ml.yaml --no-full-text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, seed_everything
from src.data.arxiv_source import (
    Paper,
    dedup_by_version,
    download_pdf,
    extract_text,
    fetch_metadata,
    write_papers,
)
from src.data.chunking import chunk_papers, write_chunks


def collect_papers(cfg: dict, delay: float) -> list[Paper]:
    """Run every configured query and merge the results."""
    ic = cfg["ingest"]
    everything: list[Paper] = []
    for query in ic["queries"]:
        everything.extend(
            fetch_metadata(
                query=query,
                max_results=ic["max_results_per_query"],
                page_size=ic["page_size"],
                delay_seconds=delay,
            )
        )
    print(f"[ingest] {len(everything)} raw results across {len(ic['queries'])} queries")
    return dedup_by_version(everything)


def attach_full_text(papers: list[Paper], cfg: dict, limit: int) -> int:
    """Download and extract text for up to `limit` papers. Returns the count."""
    ic, paths = cfg["ingest"], cfg["paths"]
    attached = 0
    for i, paper in enumerate(papers):
        if attached >= limit:
            break
        pdf_path = download_pdf(paper, paths["pdf_cache"], ic["pdf_delay_seconds"])
        if pdf_path is None:
            continue
        text = extract_text(pdf_path, paths["text_cache"])
        if text and len(text.split()) > 200:
            paper.full_text = text
            attached += 1
        if (i + 1) % 25 == 0:
            print(f"[ingest] full text {attached}/{limit} (scanned {i + 1}/{len(papers)})")
    return attached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--no-full-text", action="store_true", help="abstracts only")
    parser.add_argument("--max-full-text", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None, help="override API delay")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    ic, paths = cfg["ingest"], cfg["paths"]

    papers = collect_papers(cfg, args.delay or ic["api_delay_seconds"])
    print(f"[ingest] {len(papers)} unique papers after version dedup")

    want_full_text = ic["full_text"] and not args.no_full_text
    if want_full_text:
        limit = args.max_full_text if args.max_full_text is not None else ic["max_full_text"]
        attached = attach_full_text(papers, cfg, limit)
        print(f"[ingest] full text attached to {attached}/{len(papers)} papers")
    else:
        print("[ingest] abstracts only (full text disabled)")

    write_papers(papers, paths["papers"])

    chunks = chunk_papers(
        papers,
        max_words=cfg["chunking"]["max_words"],
        overlap_words=cfg["chunking"]["overlap_words"],
        min_words=cfg["chunking"]["min_words"],
    )
    write_chunks(chunks, paths["chunks"])

    sections: dict[str, int] = {}
    for chunk in chunks:
        sections[chunk.section] = sections.get(chunk.section, 0) + 1
    top = sorted(sections.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("[ingest] chunks by section: " + ", ".join(f"{k}={v}" for k, v in top))


if __name__ == "__main__":
    main()
