"""SciFact loading via BEIR, plus the corpus flattening the retrievers need."""

from __future__ import annotations

from pathlib import Path

# beir is deliberately NOT imported at module level. Serving imports this module
# for Corpus and passage_text only, and the deployed Space does not install the
# evaluation stack - a top-level import took the whole app down on startup.
Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


def download_dataset(url: str, cache_dir: str = "datasets") -> str:
    """Download and unzip a BEIR dataset. Returns the extracted folder path."""
    from beir import util

    name = url.rstrip("/").split("/")[-1].removesuffix(".zip")
    target = Path(cache_dir) / name
    if target.exists():
        return str(target)
    return util.download_and_unzip(url, cache_dir)


def load_split(url: str, split: str, cache_dir: str = "datasets") -> tuple[Corpus, Queries, Qrels]:
    """Load one split of a BEIR dataset."""
    from beir.datasets.data_loader import GenericDataLoader

    data_path = download_dataset(url, cache_dir)
    return GenericDataLoader(data_folder=data_path).load(split=split)


def flatten_corpus(corpus: Corpus) -> tuple[list[str], list[str]]:
    """Turn the corpus dict into parallel id/text lists.

    Index i of both lists always describes the same document. Retrievers work in
    positions; everything else in this project works in doc ids.
    """
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for doc_id, doc in corpus.items():
        doc_ids.append(doc_id)
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        doc_texts.append(f"{title} {text}".strip())
    return doc_ids, doc_texts


def passage_text(corpus: Corpus, doc_id: str) -> str:
    """The exact text representation used for both indexing and reranking."""
    doc = corpus[doc_id]
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return f"{title} {text}".strip()
