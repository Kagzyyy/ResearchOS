"""ResearchOS-ML pipeline: retrieve -> fuse -> rerank -> diversify -> answer.

Two capabilities:

  ask()              question answering over the ingested corpus
  explain_citation() resolve a citation you saw in a paper, then explain it

Three things differ from the parent project's pipeline, all forced by the corpus:

FR4.2 diversification. One paper produces dozens of chunks, so the top-scoring
five are routinely five paragraphs of the same paper. `max_per_paper` caps that,
which is what makes "explain" and "compare" answers multi-source.

US5 rejection. The threshold is on the MEAN of the top-N reranker scores, not
the maximum, because the maximum is not discriminative. Measured on this corpus:

    query                           max     mean-of-5
    explain JEPA                    0.998     0.995     in corpus
    how does JEPA differ from MAE   0.898     0.671     in corpus
    what causes knee osteoarthritis 0.670     0.190     NOT in corpus
    sourdough hydration ratio       0.001     0.001     NOT in corpus

The osteoarthritis query hits 0.670 on max because the corpus really does
contain a self-supervised bone-densitometry paper - one genuine partial match
defeats a max-based threshold. The mean separates cleanly, because a query that
belongs to this corpus matches *many* passages, not one.

Scale note: CrossEncoder.predict applies a sigmoid, so torch-backend scores are
probabilities in [0,1]. The ONNX backend returns raw logits - rescale if you
switch backends.

Citation resolution is local-first, then arXiv on demand, so the tool is not
limited to whatever was ingested.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from src.api.llm import explain_paper, generate_answer, is_configured
from src.config import device
from src.data.arxiv_source import Paper, read_papers
from src.data.chunking import chunk_paper, chunks_to_corpus, read_chunks
from src.data.citation import resolve
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
        self.chunks_by_paper: dict[str, list[str]] = {}
        for chunk in chunks:
            self.chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk.chunk_id)

        # Full paper records, for citation resolution and metadata display.
        self.papers_by_id = {p.paper_id: vars(p) for p in read_papers(paths["papers"])}

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

        # One GPU, one model, many HTTP threads. FastAPI runs sync endpoints in
        # a threadpool, so without this two concurrent users would drive
        # forward passes through the same CUDA module simultaneously - which
        # risks wrong results and VRAM spikes. Retrieval is fast enough
        # (sub-second) that serialising it costs little.
        self._gpu_lock = threading.Lock()
        self.rrf_k = rc["fusion"]["rrf_k"]
        self.retrieve_k = rc["top_k"]
        print(
            f"[ml] ready | {len(self.corpus)} chunks from {len(self.papers_by_id)} papers "
            f"| reranker={mc['model']} | llm={'groq' if is_configured() else 'disabled'}"
        )

    # ---------------------------------------------------------------- helpers

    def _passage(self, chunk_id: str, score: float) -> dict[str, Any]:
        entry = self.corpus[chunk_id]
        return {
            "chunk_id": chunk_id,
            "doc_id": chunk_id,
            "paper_id": entry["paper_id"],
            "arxiv_url": f"https://arxiv.org/abs/{entry['paper_id']}",
            "score": float(score),
            "title": entry["title"],
            "section": entry["section"],
            "published": entry["published"],
            "text": entry["text"],
        }

    def _confidence(self, scores: list[float]) -> tuple[float, float, bool]:
        ac = self.cfg["answer"]
        best = max(scores, default=0.0)
        n = min(ac.get("confidence_top_n", 5), len(scores))
        mean_top = sum(sorted(scores, reverse=True)[:n]) / n if n else 0.0
        return best, mean_top, mean_top >= ac["min_mean_score"]

    # ------------------------------------------------------------- retrieval

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Hybrid retrieve, fuse, rerank, then cap chunks per paper."""
        query_map = {PSEUDO_QID: question}
        with self._gpu_lock:
            bm25_run = self.bm25.search(query_map, top_k=self.retrieve_k)
            dense_run = self.dense.search(query_map, top_k=self.retrieve_k)
            fused = reciprocal_rank_fusion(
                [bm25_run, dense_run], k=self.rrf_k, top_k=self.retrieve_k
            )
            reranked, _ = self.reranker.rerank(
                query_map, fused, self.corpus, top_k=self.retrieve_k
            )

        ac = self.cfg["answer"]
        selected = cap_per_paper(
            reranked[PSEUDO_QID],
            self.corpus,
            top_k=top_k or ac["top_k_passages"],
            max_per_paper=ac["max_per_paper"],
        )
        return [self._passage(cid, s) for cid, s in selected]

    # -------------------------------------------------------- paper discovery

    def find_papers(self, topic: str, top_k: int = 10) -> dict[str, Any]:
        """Topic in, ranked *papers* out.

        The ask path returns passages, which is right for answering a question
        and wrong for "show me work on X" - you get five paragraphs where you
        wanted five papers. Here the per-paper cap is deliberately skipped, a
        wide candidate set is reranked, and results are then grouped by paper.

        A paper is scored by its single best passage. Summing would reward long
        papers with many mediocre chunks over short papers with one excellent
        one, which inverts what you actually want from a search result.
        """
        start = time.perf_counter()
        query_map = {PSEUDO_QID: topic}

        with self._gpu_lock:
            bm25_run = self.bm25.search(query_map, top_k=self.retrieve_k)
            dense_run = self.dense.search(query_map, top_k=self.retrieve_k)
            fused = reciprocal_rank_fusion(
                [bm25_run, dense_run], k=self.rrf_k, top_k=self.retrieve_k
            )
            reranked, _ = self.reranker.rerank(
                query_map, fused, self.corpus, top_k=self.retrieve_k
            )

        grouped: dict[str, dict[str, Any]] = {}
        for chunk_id, score in reranked[PSEUDO_QID].items():
            entry = self.corpus[chunk_id]
            pid = entry["paper_id"]
            bucket = grouped.setdefault(
                pid, {"score": float("-inf"), "matches": 0, "best_chunk": None, "sections": []}
            )
            bucket["matches"] += 1
            bucket["sections"].append(entry["section"])
            if score > bucket["score"]:
                bucket["score"] = float(score)
                bucket["best_chunk"] = chunk_id

        ordered = sorted(grouped.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]

        papers: list[dict[str, Any]] = []
        for pid, info in ordered:
            meta = self.papers_by_id.get(pid, {})
            best = self.corpus[info["best_chunk"]]
            papers.append(
                {
                    "paper_id": pid,
                    "doc_id": info["best_chunk"],
                    "title": meta.get("title") or best["title"],
                    "authors": (meta.get("authors") or [])[:6],
                    "abstract": meta.get("abstract", ""),
                    "published": meta.get("published") or best["published"],
                    "primary_category": meta.get("primary_category", ""),
                    "arxiv_url": f"https://arxiv.org/abs/{pid}",
                    "score": round(info["score"], 4),
                    "matched_passages": info["matches"],
                    "matched_sections": sorted(set(info["sections"]))[:5],
                    "why": best["text"][:400],
                    "why_section": best["section"],
                    "has_full_text": bool(meta.get("full_text")),
                }
            )

        scores = [p["score"] for p in papers]
        best_score, mean_top, confident = self._confidence(scores)
        return {
            "topic": topic,
            "papers": papers,
            "confident": confident,
            "top_score": round(best_score, 3),
            "mean_top_score": round(mean_top, 3),
            "timing_ms": {"total": round((time.perf_counter() - start) * 1000.0, 1)},
        }

    # ------------------------------------------------------------------- ask

    def answer(self, question: str, generate: bool = True) -> dict[str, Any]:
        """Question answering, including the US5 rejection path."""
        start = time.perf_counter()
        passages = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        ac = self.cfg["answer"]
        best, mean_top, confident = self._confidence([p["score"] for p in passages])

        answer_text: str | None = None
        generation_ms = 0.0
        if not confident:
            answer_text = (
                f"Nothing in this corpus answers that. The top passages averaged "
                f"{mean_top:.3f}, below the {ac['min_mean_score']:.2f} threshold, which "
                "usually means the topic is outside the ingested slice of arXiv rather "
                "than absent from the literature. The closest matches are below."
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
            "top_score": round(best, 3),
            "mean_top_score": round(mean_top, 3),
            "citations": passages,
            "distinct_papers": distinct_papers(passages),
            "generation_configured": is_configured(),
            "timing_ms": {
                "retrieval_and_rerank": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(retrieval_ms + generation_ms, 1),
            },
        }

    # ------------------------------------------------------- citation lookup

    def _passages_for_paper(self, paper: Paper, from_corpus: bool) -> list[dict[str, Any]]:
        """Passages describing one specific paper.

        Indexed papers use their stored chunks. A paper fetched on demand is
        chunked in memory - it is not added to the index, because rebuilding the
        index per lookup would be far slower than the lookup itself.
        """
        if from_corpus and paper.paper_id in self.chunks_by_paper:
            chunk_ids = self.chunks_by_paper[paper.paper_id]
            ordered = sorted(chunk_ids, key=lambda c: self.corpus[c]["section"] != "abstract")
            return [self._passage(cid, 1.0) for cid in ordered[:8]]

        chunks = chunk_paper(
            paper,
            max_words=self.cfg["chunking"]["max_words"],
            overlap_words=self.cfg["chunking"]["overlap_words"],
            min_words=self.cfg["chunking"]["min_words"],
        )[:8]
        return [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.chunk_id,
                "paper_id": c.paper_id,
                "arxiv_url": f"https://arxiv.org/abs/{c.paper_id}",
                "score": 1.0,
                "title": c.title,
                "section": c.section,
                "published": c.published,
                "text": c.text,
            }
            for c in chunks
        ]

    def explain_citation(self, citation: str, generate: bool = True) -> dict[str, Any]:
        """Resolve a citation string to a paper and explain it.

        Returns the explanation plus `related`: what else in the corpus talks
        about this paper, which is where the "and here is the wider context"
        half of the answer comes from.
        """
        start = time.perf_counter()
        resolution = resolve(
            citation,
            self.papers_by_id,
            allow_remote=self.cfg["citation"]["allow_remote"],
            fetch_full_text=self.cfg["citation"]["fetch_full_text"],
            pdf_dir=self.cfg["paths"]["pdf_cache"],
            text_dir=self.cfg["paths"]["text_cache"],
        )

        if resolution.paper is None:
            return {
                "citation": citation,
                "resolved": False,
                "method": resolution.method,
                "query_used": resolution.query_used,
                "candidates": resolution.candidates,
                "explanation": (
                    "Could not resolve that citation to a paper. Try the arXiv id, or the "
                    "paper title on its own."
                ),
                "paper": None,
                "passages": [],
                "related": [],
                "generation_configured": is_configured(),
                "timing_ms": {"total": round((time.perf_counter() - start) * 1000.0, 1)},
            }

        paper = resolution.paper
        passages = self._passages_for_paper(paper, resolution.from_corpus)

        # What else in the corpus discusses this work - the wider-context half.
        related_query = f"{paper.title} {paper.abstract[:300]}"
        related = [
            p
            for p in self.retrieve(related_query, top_k=6)
            if p["paper_id"] != paper.paper_id
        ][:4]

        explanation: str | None = None
        gen_ms = 0.0
        if generate:
            gen_start = time.perf_counter()
            explanation = explain_paper(
                paper.title,
                passages + related,
                model=self.cfg["answer"]["model"],
                max_tokens=self.cfg["citation"]["max_tokens"],
            )
            gen_ms = (time.perf_counter() - gen_start) * 1000.0

        return {
            "citation": citation,
            "resolved": True,
            "method": resolution.method,
            "query_used": resolution.query_used,
            "from_corpus": resolution.from_corpus,
            "candidates": resolution.candidates,
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors[:8],
                "published": paper.published,
                "primary_category": paper.primary_category,
                "arxiv_url": f"https://arxiv.org/abs/{paper.paper_id}",
                "has_full_text": bool(paper.full_text),
            },
            "explanation": explanation,
            "passages": passages,
            "related": related,
            "generation_configured": is_configured(),
            "timing_ms": {
                "total": round((time.perf_counter() - start) * 1000.0, 1),
                "generation": round(gen_ms, 1),
            },
        }
