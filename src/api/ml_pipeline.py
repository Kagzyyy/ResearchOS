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

import re
import threading
import time
from typing import Any

from src.api.llm import explain_paper, generate_answer, is_configured
from src.config import device
from src.data.arxiv_source import Paper, fetch_metadata, read_papers
from src.data.chunking import chunk_paper, chunks_to_corpus, read_chunks
from src.data.citation import looks_like_a_question, resolve
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.diversify import cap_per_paper, distinct_papers
from src.retrieval.fusion import reciprocal_rank_fusion

PSEUDO_QID = "q"

_QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "what", "how", "why", "does", "explain",
    "when", "are", "was", "that", "this", "there", "which", "about", "into",
    "some", "can", "you", "its", "their", "using", "used", "based",
}


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

        # Same live fallback the ask path has. Without it, Find papers is
        # limited to the ingested sample - and the corpus holds roughly 1,400
        # papers per category against the 50-150/day arXiv publishes, so a
        # topic can be well covered on arXiv and absent here. Searching "economic
        # world models" returned unrelated robotics papers for exactly this
        # reason while the paper sat on arXiv.
        source = "corpus"
        ac = self.cfg["answer"]
        if not confident and self.cfg["citation"].get("allow_remote", True):
            live = self.search_arxiv_live(topic, top_k=top_k)
            if live:
                live = self._rescore_live(topic, live)
                floor = ac.get("live_min_score", 0.30)
                kept = [p for p in live if p["score"] >= floor]
                if kept:
                    seen = {p["paper_id"] for p in papers}
                    fetched = [
                        {
                            "paper_id": p["paper_id"],
                            "doc_id": p["doc_id"],
                            "title": p["title"],
                            "authors": [],
                            "abstract": p["text"],
                            "published": p["published"],
                            "primary_category": "",
                            "arxiv_url": p["arxiv_url"],
                            "score": round(p["score"], 4),
                            "matched_passages": 1,
                            "matched_sections": ["abstract"],
                            "why": p["text"][:400],
                            "why_section": "abstract",
                            "has_full_text": False,
                            "from_arxiv": True,
                        }
                        for p in kept
                        if p["paper_id"] not in seen
                    ]
                    if fetched:
                        papers = (fetched + papers)[:top_k]
                        source = "arxiv_live"
                        confident = True
                        mean_top = sum(p["score"] for p in fetched) / len(fetched)
                        best_score = max(p["score"] for p in fetched)

        return {
            "topic": topic,
            "source": source,
            "papers": papers,
            "confident": confident,
            "top_score": round(best_score, 3),
            "mean_top_score": round(mean_top, 3),
            "timing_ms": {"total": round((time.perf_counter() - start) * 1000.0, 1)},
        }

    def _rescore_live(
        self, question: str, live: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score live arXiv abstracts with the same reranker the corpus uses.

        arXiv's own relevance ranking is lexical and generous. Running the
        cross-encoder over the abstracts puts live results on the same scale as
        corpus results, so one confidence threshold governs both.
        """
        pairs = [(question, f"{p['title']} {p['text']}".strip()) for p in live]
        if not pairs:
            return []
        with self._gpu_lock:
            scores = self.reranker.score(pairs)
        for passage, score in zip(live, scores):
            passage["score"] = float(score)
        return sorted(live, key=lambda p: p["score"], reverse=True)

    def answer_from_document(
        self, question: str, passages: list[dict[str, Any]], title: str, generate: bool = True
    ) -> dict[str, Any]:
        """Answer a question against a single uploaded paper.

        Every chunk of the document is scored directly by the cross-encoder -
        no BM25, no embeddings, no index. A paper is only 20-40 chunks, so
        exhaustive scoring is both cheap and strictly better than retrieval,
        which can only approximate it.
        """
        start = time.perf_counter()
        if not passages:
            return {
                "question": question,
                "answer": "That document has no readable passages.",
                "confident": False,
                "source": "upload",
                "top_score": 0.0,
                "mean_top_score": 0.0,
                "citations": [],
                "distinct_papers": 0,
                "generation_configured": is_configured(),
                "timing_ms": {"retrieval_and_rerank": 0.0, "generation": 0.0, "total": 0.0},
            }

        pairs = [(question, f"{p['title']} {p['text']}".strip()) for p in passages]
        with self._gpu_lock:
            scores = self.reranker.score(pairs)
        for passage, score in zip(passages, scores):
            passage["score"] = float(score)
        ranked = sorted(passages, key=lambda p: p["score"], reverse=True)

        ac = self.cfg["answer"]
        top = ranked[: ac["top_k_passages"]]
        best, mean_top, _ = self._confidence([p["score"] for p in top])
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        answer_text: str | None = None
        generation_ms = 0.0
        if generate:
            gen_start = time.perf_counter()
            answer_text = generate_answer(
                f"About the paper '{title}': {question}",
                top,
                model=ac["model"],
                max_tokens=ac["max_tokens"],
            )
            generation_ms = (time.perf_counter() - gen_start) * 1000.0

        return {
            "question": question,
            "answer": answer_text,
            # The user chose this document, so we answer from it even when the
            # match is weak - declining would be second-guessing them.
            "confident": True,
            "source": "upload",
            "top_score": round(best, 3),
            "mean_top_score": round(mean_top, 3),
            "citations": top,
            "distinct_papers": 1,
            "generation_configured": is_configured(),
            "timing_ms": {
                "retrieval_and_rerank": round(retrieval_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(retrieval_ms + generation_ms, 1),
            },
        }

    # ------------------------------------------------------- live arxiv search

    def search_arxiv_live(self, question: str, top_k: int = 6) -> list[dict[str, Any]]:
        """Search arXiv directly, for questions the local corpus cannot answer.

        The corpus is a sample, not coverage: ~1,400 papers per category across
        nine months, against the 50-150/day arXiv actually publishes. Most
        papers a user asks about were never ingested, and telling them "not in
        this corpus" is honest but useless when the paper plainly exists.

        Abstracts only - no PDF fetch - so this stays a couple of seconds.
        """
        terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
                 if w.lower() not in _QUERY_STOPWORDS][:12]
        if not terms:
            return []
        try:
            found = fetch_metadata(
                "all:" + " AND ".join(f'"{t}"' for t in terms[:4]),
                max_results=top_k,
                sort_by_relevance=True,
            )
            if not found:
                found = fetch_metadata(
                    "all:" + " ".join(terms[:8]), max_results=top_k, sort_by_relevance=True
                )
        except Exception:  # noqa: BLE001 - a failed fallback is not a failed request
            return []

        return [
            {
                "chunk_id": f"live:{p.paper_id}",
                "doc_id": f"live:{p.paper_id}",
                "paper_id": p.paper_id,
                "arxiv_url": f"https://arxiv.org/abs/{p.paper_id}",
                "score": 1.0,
                "title": p.title,
                "section": "abstract",
                "published": p.published,
                "text": p.abstract,
            }
            for p in found
        ]

    # ------------------------------------------------------------------- ask

    def answer(self, question: str, generate: bool = True) -> dict[str, Any]:
        """Question answering, including the US5 rejection path."""
        start = time.perf_counter()
        passages = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000.0

        ac = self.cfg["answer"]
        best, mean_top, confident = self._confidence([p["score"] for p in passages])

        # Blend in live arXiv results when the corpus alone is thin.
        #
        # The corpus is a sample (~1,400 papers per category against the 50-150
        # a day arXiv publishes), so a question can be perfectly answerable from
        # the literature and unanswerable from here. Two triggers:
        #
        #   below min_mean_score   the corpus has nothing - live results replace
        #   below enrich_below     the corpus has something thin - live results
        #                          are added, so the answer draws on both
        #
        # A strong corpus hit skips the network entirely and stays sub-second.
        answer_text: str | None = None
        generation_ms = 0.0
        source = "corpus"
        remote_ok = self.cfg["citation"].get("allow_remote", True)
        enrich_below = ac.get("enrich_below", 0.0)

        if remote_ok and (not confident or mean_top < enrich_below):
            live = self.search_arxiv_live(question)
            if live:
                # arXiv always returns *something*, so live results must clear
                # the same confidence bar as the corpus. Without this the
                # honest "I do not know" is replaced by confident nonsense -
                # a sourdough question came back with quantum-physics papers.
                # Filter live results PER PASSAGE, not on their mean.
                #
                # The mean-of-top-5 rule works for the corpus because a query
                # that belongs to it matches many passages. An arXiv keyword
                # search returns a mixed bag by construction - three loose hits
                # would drag the mean under and bury the one paper that is
                # actually the answer. So each live passage clears its own bar.
                live = self._rescore_live(question, live)
                floor = ac.get("live_min_score", 0.30)
                live = [p for p in live if p["score"] >= floor]
                seen = {p["paper_id"] for p in passages}
                extra = [p for p in live if p["paper_id"] not in seen]
                if confident and extra:
                    passages = passages + extra          # enrich
                    source = "corpus+arxiv_live"
                elif extra:
                    passages = extra                     # replace
                    source = "arxiv_live"
                    confident = True
                    mean_top = sum(p["score"] for p in extra) / len(extra)
                elif not confident:
                    passages = []                        # genuinely unanswerable

        if not passages:
            answer_text = (
                "Nothing answers that - not in the local corpus, and not in a live "
                "arXiv search either. If this is an AI/ML topic the paper may simply "
                "be too new or too obscure; if it is not, this tool only covers arXiv."
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
            "source": source,
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

        # A question is not a citation. Trying to resolve one costs two arXiv
        # searches that cannot succeed, and arXiv answers 429 - which is how a
        # concept question turned into a minute of waiting and then a failure.
        if looks_like_a_question(citation):
            answer = self.answer(citation, generate=generate)
            return {
                "citation": citation,
                "resolved": False,
                "method": "question",
                "query_used": citation,
                "candidates": [],
                "paper": None,
                "explanation": answer["answer"],
                "passages": answer["citations"],
                "related": [],
                "answered_as_question": True,
                "source": answer["source"],
                "generation_configured": is_configured(),
                "timing_ms": answer["timing_ms"],
            }

        resolution = resolve(
            citation,
            self.papers_by_id,
            allow_remote=self.cfg["citation"]["allow_remote"],
            fetch_full_text=self.cfg["citation"]["fetch_full_text"],
            pdf_dir=self.cfg["paths"]["pdf_cache"],
            text_dir=self.cfg["paths"]["text_cache"],
            max_cache_gb=self.cfg["citation"].get("max_pdf_cache_gb", 5.0),
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
