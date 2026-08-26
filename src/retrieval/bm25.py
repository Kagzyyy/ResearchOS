"""Lexical retrieval with bm25s, cached to disk."""

from __future__ import annotations

from pathlib import Path

import bm25s
import Stemmer

Run = dict[str, dict[str, float]]


class BM25Retriever:
    """Thin wrapper over bm25s that speaks doc ids instead of positions."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, stemmer_lang: str = "english") -> None:
        self.k1 = k1
        self.b = b
        self.stemmer = Stemmer.Stemmer(stemmer_lang) if stemmer_lang else None
        self.retriever: bm25s.BM25 | None = None
        self.doc_ids: list[str] = []

    def _tokenize(self, texts: list[str]) -> object:
        return bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)

    def index(self, doc_ids: list[str], doc_texts: list[str], cache_dir: str | None = None) -> None:
        """Build (or load) the inverted index for this corpus."""
        self.doc_ids = doc_ids
        cache = Path(cache_dir) / "bm25" if cache_dir else None
        if cache is not None and (cache / "data.csc.index.npy").exists():
            self.retriever = bm25s.BM25.load(str(cache), load_corpus=False)
            return
        retriever = bm25s.BM25(k1=self.k1, b=self.b)
        retriever.index(self._tokenize(doc_texts), show_progress=False)
        self.retriever = retriever
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            retriever.save(str(cache))

    def search(self, queries: dict[str, str], top_k: int = 50) -> Run:
        """Return {query_id: {doc_id: score}} for every query."""
        if self.retriever is None:
            raise RuntimeError("index() must be called before search()")
        qids = list(queries.keys())
        query_tokens = self._tokenize([queries[q] for q in qids])
        positions, scores = self.retriever.retrieve(query_tokens, k=top_k, show_progress=False)
        run: Run = {}
        for row, qid in enumerate(qids):
            run[qid] = {
                self.doc_ids[int(positions[row, col])]: float(scores[row, col])
                for col in range(positions.shape[1])
            }
        return run
