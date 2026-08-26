"""The serving pipeline: retrieve -> fuse -> rerank -> generate.

Shared by the FastAPI app and the Gradio UI so both measure the same thing.
"""

from __future__ import annotations

import time
from typing import Any

from src.api.generate import generate_answer, is_configured
from src.config import device
from src.data.loader import flatten_corpus, load_split, passage_text
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


class ResearchPipeline:
    """Loads every index and model once, then answers questions."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        dc = cfg["dataset"]
        self.corpus, _queries, _qrels = load_split(dc["url"], dc["split"], dc["download_dir"])
        doc_ids, doc_texts = flatten_corpus(self.corpus)

        rc = cfg["retrieval"]
        cache_dir = cfg["paths"]["cache_dir"]

        self.bm25 = BM25Retriever(
            k1=rc["bm25"]["k1"], b=rc["bm25"]["b"], stemmer_lang=rc["bm25"]["stemmer"]
        )
        self.bm25.index(doc_ids, doc_texts, cache_dir=cache_dir)

        self.dense = DenseRetriever(
            model_name=rc["dense"]["model"],
            batch_size=rc["dense"]["batch_size"],
            max_length=rc["dense"]["max_length"],
            query_prefix=rc["dense"]["query_prefix"],
            device=device(),
        )
        self.dense.index_corpus(doc_ids, doc_texts, cache_dir=cache_dir)

        self.reranker, self.backend = self._load_reranker(cfg["reranker"])
        self.rrf_k = rc["fusion"]["rrf_k"]
        self.retrieve_k = rc["top_k"]
        print(f"[pipeline] ready | {len(self.corpus)} docs | reranker backend={self.backend}")

    @staticmethod
    def _load_reranker(rc: dict[str, Any]) -> tuple[Any, str]:
        backend = rc.get("backend", "torch")
        if backend == "onnx":
            from src.rerank.onnx_runtime import OnnxReranker

            return (
                OnnxReranker(
                    onnx_path=rc["model"],
                    tokenizer_path=rc["tokenizer"],
                    max_length=rc["max_length"],
                    batch_size=rc["batch_size"],
                ),
                "onnx",
            )
        from src.rerank.model import Reranker

        return (
            Reranker(
                model_name_or_path=rc["model"],
                max_length=rc["max_length"],
                batch_size=rc["batch_size"],
                device=device(),
                fp16=rc.get("fp16", True) and device() == "cuda",
            ),
            "torch",
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve, fuse, and rerank. Returns passages sorted best-first."""
        retrieve_k = top_k or self.retrieve_k
        pseudo_qid = "q"
        query_map = {pseudo_qid: question}

        bm25_run = self.bm25.search(query_map, top_k=retrieve_k)
        dense_run = self.dense.search(query_map, top_k=retrieve_k)
        fused = reciprocal_rank_fusion([bm25_run, dense_run], k=self.rrf_k, top_k=retrieve_k)

        reranked, _timings = self.reranker.rerank(query_map, fused, self.corpus, top_k=retrieve_k)
        ordered = sorted(reranked[pseudo_qid].items(), key=lambda kv: kv[1], reverse=True)

        return [
            {
                "doc_id": doc_id,
                "score": float(score),
                "title": self.corpus[doc_id].get("title", ""),
                "text": self.corpus[doc_id].get("text", ""),
                "full_text": passage_text(self.corpus, doc_id),
            }
            for doc_id, score in ordered
        ]

    def answer(
        self, question: str, top_k_final: int = 5, generate: bool = True
    ) -> dict[str, Any]:
        """Full pipeline. Returns the answer, its citations, and timings."""
        start = time.perf_counter()
        passages = self.retrieve(question)[:top_k_final]
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        answer_text = None
        generation_ms = 0.0
        if generate:
            gen_start = time.perf_counter()
            answer_text = generate_answer(question, passages)
            generation_ms = (time.perf_counter() - gen_start) * 1000.0

        return {
            "question": question,
            "answer": answer_text,
            "citations": [
                {k: p[k] for k in ("doc_id", "score", "title", "text")} for p in passages
            ],
            "backend": self.backend,
            "generation_configured": is_configured(),
            "timing_ms": {
                "retrieval_and_rerank": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(retrieval_ms + generation_ms, 1),
            },
        }
