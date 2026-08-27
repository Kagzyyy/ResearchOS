"""ResearchOS-ML serving pipeline: retrieve -> fuse -> rerank -> diversify -> answer.

Two things differ from the parent project's pipeline, both forced by the corpus:

FR4.2 diversification. One paper produces dozens of chunks, so the top-scoring
five are routinely five paragraphs of the same paper. `max_per_paper` caps that,
which is what makes "explain" and "compare" answers multi-source.

US5 rejection. The corpus is topic-scoped, so most of the world is out of it.
Below `min_score` the system says it does not know rather than presenting
low-scoring unrelated passages as an answer. The parent project showed the raw
signal is there - an out-of-domain query scored +0.06 against +0.60 in-domain -
it just never acted on it.
"""

from __future__ import annotations

import time
from typing import Any

from src.api.generate import generate_answer, is_configured
from src.config import device
from src.data.chunking import chunks_to_corpus, read_chunks
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.diversify import cap_per_paper, distinct_papers
from src.retrieval.fusion import reciprocal_rank_fusion

PSEUDO_QID = "q"


class MLPipeline:
    """Loads chunk corpus and indices once, then answers questions."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        paths, rc = cfg["paths"], cfg["retrieval"]

        chunks = read_chunks(paths["chunks"])
        self.corpus = chunks_to_corpus(chunks)
        self.papers = {c.paper_id: c.title for c in chunks}

        doc_ids = list(self.corpus.keys())
        doc_texts = [
            f"{self.corpus[d]['title']} {self.corpus[d]['text']}".strip() for d in doc_ids
        ]

        # The lexical arm is mandatory here: bge-small has frozen weights and has
        # never seen "JEPA" or "GRPO", while BM25 gives a rare token a very high
        # IDF weight. It is the only defence against vocabulary coined after the
        # embedding model was trained.
        self.bm25 = BM25Retriever(
            k1=rc["bm25"]["k1"], b=rc["bm25"]["b"], stemmer_lang=rc["bm25"]["stemmer"]
        )
        self.bm25.index(doc_ids, doc_texts, cache_dir=paths["index_cache"])

        self.dense = DenseRetriever(
            model_name=rc["dense"]["model"],
            batch_size=rc["dense"]["batch_size"],
            max_length=rc["dense"]["max_length"],
            query_prefix=rc["dense"]["query_prefix"],
            device=device(),
        )
        self.dense.index_corpus(doc_ids, doc_texts, cache_dir=paths["index_cache"])

        from src.rerank.model import Reranker

        mc = cfg["reranker"]
        self.reranker = Reranker(
            model_name_or_path=mc["model"],
            max_length=mc["max_length"],
            batch_size=mc["batch_size"],
            device=device(),
            fp16=mc.get("fp16", True) and device() == "cuda",
        )

        self.rrf_k = rc["fusion"]["rrf_k"]
        self.retrieve_k = rc["top_k"]
        print(
            f"[ml] ready | {len(self.corpus)} chunks from {len(self.papers)} papers "
            f"| reranker={mc['model']}"
        )

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Hybrid retrieve, fuse, rerank, then cap chunks per paper."""
        query_map = {PSEUDO_QID: question}
        bm25_run = self.bm25.search(query_map, top_k=self.retrieve_k)
        dense_run = self.dense.search(query_map, top_k=self.retrieve_k)
        fused = reciprocal_rank_fusion(
            [bm25_run, dense_run], k=self.rrf_k, top_k=self.retrieve_k
        )

        reranked, _timings = self.reranker.rerank(
            query_map, fused, self.corpus, top_k=self.retrieve_k
        )

        ac = self.cfg["answer"]
        selected = cap_per_paper(
            reranked[PSEUDO_QID],
            self.corpus,
            top_k=ac["top_k_passages"],
            max_per_paper=ac["max_per_paper"],
        )

        results: list[dict[str, Any]] = []
        for chunk_id, score in selected:
            entry = self.corpus[chunk_id]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": chunk_id,  # generate.py speaks doc_id
                    "paper_id": entry["paper_id"],
                    "arxiv_url": f"https://arxiv.org/abs/{entry['paper_id']}",
                    "score": float(score),
                    "title": entry["title"],
                    "section": entry["section"],
                    "published": entry["published"],
                    "text": entry["text"],
                }
            )
        return results

    def answer(self, question: str, generate: bool = True) -> dict[str, Any]:
        """Full pipeline, including the US5 rejection path."""
        start = time.perf_counter()
        passages = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        ac = self.cfg["answer"]
        best = max((p["score"] for p in passages), default=float("-inf"))
        confident = best >= ac["min_score"]

        answer_text: str | None = None
        generation_ms = 0.0
        if not confident:
            answer_text = (
                "Nothing in this corpus answers that. The best passage scored "
                f"{best:.2f}, below the {ac['min_score']:.1f} threshold, which usually "
                "means the topic is outside the ingested slice of arXiv rather than "
                "absent from the literature. The closest matches are listed below."
            )
        elif generate:
            gen_start = time.perf_counter()
            answer_text = generate_answer(
                question, passages, model=ac["model"], max_tokens=ac["max_tokens"]
            )
            generation_ms = (time.perf_counter() - gen_start) * 1000.0

        return {
            "question": question,
            "answer": answer_text,
            "confident": confident,
            "top_score": None if best == float("-inf") else round(best, 3),
            "citations": passages,
            "distinct_papers": distinct_papers(passages),
            "generation_configured": is_configured(),
            "timing_ms": {
                "retrieval_and_rerank": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(retrieval_ms + generation_ms, 1),
            },
        }
