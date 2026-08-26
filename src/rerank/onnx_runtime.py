"""ONNX Runtime reranker.

Interface-compatible with src.rerank.model.Reranker so the same evaluation and
benchmark scripts can drive either backend and the comparison stays honest.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.data.loader import Corpus, passage_text

Run = dict[str, dict[str, float]]


class OnnxReranker:
    """Cross-encoder inference through ONNX Runtime on CPU."""

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: str,
        max_length: int = 384,
        batch_size: int = 32,
        intra_op_threads: int = 0,
    ) -> None:
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads

        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.batch_size = batch_size

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Relevance logits for a batch of (query, passage) pairs."""
        scores: list[np.ndarray] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [q for q, _ in chunk],
                [p for _, p in chunk],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {name: encoded[name].astype(np.int64) for name in self.input_names}
            logits = self.session.run(["logits"], feed)[0]
            scores.append(np.asarray(logits, dtype=np.float32).reshape(-1))
        return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)

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
