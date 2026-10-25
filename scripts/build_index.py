"""Build the BM25 and dense index caches without starting the whole pipeline.

    python scripts/build_index.py --config configs/ml.yaml

MLPipeline builds these on first start, but it also loads the reranker and
every paper record, which at 270k papers costs several GB of RAM that
indexing never uses. This builds exactly the same caches (same ids, same
text, same fingerprints) in a lean process - the one to run on a laptop -
and the app then finds them ready and starts in seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import device, load_config  # noqa: E402
from src.data.chunking import chunks_to_corpus, read_chunks  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/ml.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths, rc = cfg["paths"], cfg["retrieval"]
    corpus = chunks_to_corpus(read_chunks(paths["chunks"]))
    doc_ids = list(corpus.keys())
    # Must match MLPipeline.__init__ exactly, or the caches would never be hit.
    doc_texts = [f"{corpus[d]['title']} {corpus[d]['text']}".strip() for d in doc_ids]
    del corpus
    print(f"[index] {len(doc_ids):,} passages -> {paths['index_cache']}", flush=True)

    t = time.time()
    BM25Retriever(k1=rc["bm25"]["k1"], b=rc["bm25"]["b"],
                  stemmer_lang=rc["bm25"]["stemmer"]).index(
        doc_ids, doc_texts, cache_dir=paths["index_cache"])
    print(f"[index] bm25 ready in {time.time() - t:.0f}s", flush=True)

    t = time.time()
    DenseRetriever(model_name=rc["dense"]["model"], batch_size=rc["dense"]["batch_size"],
                   max_length=rc["dense"]["max_length"], query_prefix=rc["dense"]["query_prefix"],
                   device=device()).index_corpus(doc_ids, doc_texts, cache_dir=paths["index_cache"])
    print(f"[index] dense ready in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
