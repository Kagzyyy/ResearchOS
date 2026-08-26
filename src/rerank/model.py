"""Cross-encoder reranking: score every (query, passage) pair and reorder."""

from __future__ import annotations

import time

import numpy as np
from sentence_transformers import CrossEncoder

from src.data.loader import Corpus, passage_text

Run = dict[str, dict[str, float]]


class Reranker:
    """Wraps a CrossEncoder so it can reorder a retrieval run in place."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        max_length: int = 384,
        batch_size: int = 32,
        device: str | None = None,
        fp16: bool = True,
    ) -> None:
        self.model = CrossEncoder(
            model_name_or_path,
            num_labels=1,
            max_length=max_length,
            device=device,
        )
        if fp16 and device != "cpu":
            self.model.model.half()
        self.batch_size = batch_size

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Raw relevance logits for a batch of (query, passage) pairs."""
        return np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        ).reshape(-1)

    def rerank(
        self,
        queries: dict[str, str],
        run: Run,
        corpus: Corpus,
        top_k: int = 50,
    ) -> tuple[Run, list[float]]:
        """Rerank each query's candidates. Returns the new run and per-query ms."""
        reranked: Run = {}
        timings_ms: list[float] = []
        for qid, candidates in run.items():
            doc_ids = [d for d, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)]
            doc_ids = doc_ids[:top_k]
            if not doc_ids:
                reranked[qid] = {}
                timings_ms.append(0.0)
                continue
            pairs = [(queries[qid], passage_text(corpus, d)) for d in doc_ids]
            start = time.perf_counter()
            scores = self.score(pairs)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
            reranked[qid] = {d: float(s) for d, s in zip(doc_ids, scores)}
        return reranked, timings_ms
