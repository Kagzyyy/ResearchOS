"""Publish the corpus and its prebuilt indices to a Hugging Face dataset repo.

    python scripts/publish_corpus.py --repo Kagzy/researchos-corpus

A free Space's own repo holds 1 GB; the 273k-paper corpus with its indices is
~1.6 GB. The Space downloads this dataset on first start instead
(src/api/corpus_fetch.py). Resumable: upload_large_folder hashes and uploads
in pieces and skips what is already there, so an interrupted upload continues
rather than restarting.

The paper records are slimmed first - `full_text` (82 MB of PDF text for the
8k v1 papers) is never read at serving time; chunks.jsonl already holds it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from prepare_deploy import slim_papers  # noqa: E402
from src.config import load_config  # noqa: E402

CARD = """---
license: cc0-1.0
language: [en]
pretty_name: ResearchOS corpus
size_categories: [100K<n<1M]
task_categories: [text-retrieval]
tags: [arxiv, machine-learning, retrieval, rag]
---

# ResearchOS corpus

The search corpus behind [ResearchOS](https://github.com/Kagzyyy/ResearchOS): AI
papers from arXiv, plus the BM25 and dense (BAAI/bge-small-en-v1.5) indices
prebuilt over them, so a server can start in seconds instead of embedding
~{passages:,} passages.

| | |
|---|---|
| papers | {papers:,} |
| passages | {passages:,} |
| coverage | arXiv cs.LG, cs.AI, cs.CL, cs.CV, cs.RO, cs.IR, cs.NE and stat.ML, first submitted 2024-01 onward, plus {full_text:,} earlier papers with full text |
| source | arXiv OAI-PMH metadata, which arXiv releases under CC0 |

```
data/corpus_v3/papers.jsonl    one paper per line (title, abstract, authors, category, dates)
data/corpus_v3/chunks.jsonl    retrievable passages (an abstract, or a section window of a full text)
cache/ml_v3/bm25/              bm25s index   (corpus.json = fingerprint of the exact passage ids)
cache/ml_v3/dense_*.npy        float32 embeddings, row i = passage i of chunks.jsonl
```

The indices are valid only for these exact files: each is fingerprinted to the
ordered passage ids, and the loader rebuilds rather than trust a mismatch.
Rebuild with `scripts/harvest_arxiv_oai.py`, `scripts/build_corpus.py` and
`scripts/build_index.py`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Kagzy/researchos-corpus")
    ap.add_argument("--config", default="configs/ml.yaml")
    ap.add_argument("--stage", default=None, help="staging dir (default: beside the corpus)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    paths = load_config(args.config)["paths"]
    papers, chunks, cache = Path(paths["papers"]), Path(paths["chunks"]), Path(paths["index_cache"])
    stage = Path(args.stage or papers.parent.parent / "_publish_stage")
    if stage.exists():
        shutil.rmtree(stage)
    (stage / papers.parent).mkdir(parents=True)

    print("[publish] slimming papers ...", flush=True)
    n_papers, before, after = slim_papers(papers, stage / papers)
    print(f"[publish] {n_papers:,} papers, {before / 1e6:.0f} -> {after / 1e6:.0f} MB", flush=True)

    def link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)           # same volume: no extra disk
        except OSError:
            shutil.copy2(src, dst)

    link(chunks, stage / chunks)
    for f in cache.rglob("*"):
        if f.is_file():
            link(f, stage / f)

    n_passages = sum(1 for _ in open(chunks, encoding="utf-8"))
    n_full = sum(1 for line in open(papers, encoding="utf-8")
                 if '"full_text": "' in line)
    (stage / "README.md").write_text(
        CARD.format(papers=n_papers, passages=n_passages, full_text=n_full), encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"[publish] uploading {stage} -> {args.repo} (resumable) ...", flush=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(stage))
    print(f"[publish] done: https://huggingface.co/datasets/{args.repo}", flush=True)
    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
