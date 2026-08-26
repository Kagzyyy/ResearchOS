"""Structural types for the duck-typed interfaces in this project.

Several functions accept "anything that can score pairs" or "anything that can
search" - the torch reranker and the ONNX reranker are interchangeable, and that
interchangeability is the whole point of the row-6 benchmark. Protocols express
that without forcing a shared base class.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.data.loader import Corpus

Run = dict[str, dict[str, float]]


class Scorer(Protocol):
    """Anything that turns (query, passage) pairs into relevance scores."""

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


class RerankerLike(Scorer, Protocol):
    """A Scorer that can also reorder a whole retrieval run.

    Implemented by both src.rerank.model.Reranker (torch) and
    src.rerank.onnx_runtime.OnnxReranker (onnxruntime).
    """

    def rerank(
        self,
        queries: dict[str, str],
        run: Run,
        corpus: Corpus,
        top_k: int = 50,
    ) -> tuple[Run, list[float]]: ...


class SearchRetriever(Protocol):
    """A first-stage retriever: BM25Retriever or DenseRetriever."""

    def search(self, queries: dict[str, str], top_k: int = 50) -> Run: ...
