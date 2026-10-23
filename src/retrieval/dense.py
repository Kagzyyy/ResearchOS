"""Dense retrieval: bge-small-en-v1.5 embeddings over a FAISS flat index."""

from __future__ import annotations

import hashlib
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

Run = dict[str, dict[str, float]]

# BGE v1.5 retrieval models expect this instruction on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Encode once, cache the matrix, search with exact inner product."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        max_length: int = 512,
        query_prefix: str = BGE_QUERY_PREFIX,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.index: faiss.Index | None = None
        self.doc_ids: list[str] = []

    def _encode(self, texts: list[str], desc: str) -> np.ndarray:
        verbose = len(texts) > 128
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=verbose,
        )
        if verbose:
            print(f"[dense] encoded {len(texts)} {desc} -> {vectors.shape}")
        return vectors.astype(np.float32)

    def index_corpus(
        self, doc_ids: list[str], doc_texts: list[str], cache_dir: str | None = None
    ) -> None:
        """Embed the corpus (or reuse the cached matrix) and build a flat index."""
        self.doc_ids = doc_ids
        cache_file = None
        if cache_dir:
            slug = self.model_name.replace("/", "_")
            # Keyed on the exact id list, not just its length: two corpora of
            # the same size would otherwise share (wrong) vectors silently.
            ids_sha = hashlib.sha1("\n".join(doc_ids).encode()).hexdigest()[:10]
            cache_file = Path(cache_dir) / f"dense_{slug}_{len(doc_ids)}_{ids_sha}.npy"
            cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file is not None and cache_file.exists():
            vectors = np.load(cache_file)
        else:
            vectors = self._encode(doc_texts, "documents")
            if cache_file is not None:
                np.save(cache_file, vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index

    def search(self, queries: dict[str, str], top_k: int = 50) -> Run:
        """Return {query_id: {doc_id: cosine_score}} for every query."""
        if self.index is None:
            raise RuntimeError("index_corpus() must be called before search()")
        qids = list(queries.keys())
        texts = [self.query_prefix + queries[q] for q in qids]
        vectors = self._encode(texts, "queries")
        scores, positions = self.index.search(vectors, top_k)
        run: Run = {}
        for row, qid in enumerate(qids):
            run[qid] = {
                self.doc_ids[int(positions[row, col])]: float(scores[row, col])
                for col in range(positions.shape[1])
            }
        return run
