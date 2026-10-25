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
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as _futures_wait
from typing import Any

from src.api.diagnose import diagnose
from src.api.llm import (
    explain_paper, general_answer, generate_answer, is_configured, is_not_covered,
)
from src.config import device
from src.data.arxiv_source import (
    ArxivUnavailable, Paper, fetch_metadata_bounded, read_papers,
)
from src.data import openalex_source as openalex
from src.data.chunking import chunk_paper, chunks_to_corpus, read_chunks
from src.data.citation import looks_like_a_question, paper_url, resolve
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.diversify import cap_per_paper, distinct_papers
from src.retrieval.fusion import reciprocal_rank_fusion

PSEUDO_QID = "q"

# Live sources run side by side under one deadline; their own clients bound
# each call, so this pool only has to hold the two waits of one request.
_LIVE_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="live-search")
_SEARCH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="first-stage")
RANK_CACHE_MAX = 512


def _dedup_key(item: dict[str, Any]) -> str:
    """arXiv id when known, else a normalised title: one paper, one entry."""
    pid = item.get("paper_id") or ""
    if pid and not pid.startswith("openalex:"):
        return pid
    return re.sub(r"[^a-z0-9]", "", (item.get("title") or "").lower())[:120]

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

        # An abstract-only paper holds its abstract twice - once in its paper
        # record, once as its single passage - as two equal strings. At 270k
        # papers that is ~400 MB of RAM for nothing. Point both at one object.
        for entry in self.corpus.values():
            if entry["section"] == "abstract":
                record = self.papers_by_id.get(entry["paper_id"])
                if record is not None and record.get("abstract") == entry["text"]:
                    entry["text"] = record["abstract"]

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
        self._rank_cache: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.cache_hits = 0
        self.early_exits = 0
        print(
            f"[ml] ready | {len(self.corpus)} chunks from {len(self.papers_by_id)} papers "
            f"| reranker={mc['model']} | llm={'groq' if is_configured() else 'disabled'}"
        )

    def warmup(self) -> None:
        """Run one throwaway query so the first real user does not pay for it.

        The first call into torch, faiss and the tokenizers allocates buffers
        and compiles kernels; on a CPU container that is seconds the first
        visitor would otherwise sit through. The result is cached, harmlessly.
        """
        t0 = time.perf_counter()
        try:
            self._ranked("how do transformers use attention for language modelling")
            print(f"[ml] warmed up in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - warmup is an optimisation only
            print(f"[ml] warmup skipped ({type(exc).__name__})", flush=True)

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

    def _ranked(self, question: str) -> dict[str, float]:
        """Reranked candidate chunks for a query: {chunk_id: score}.

        The one place a query is searched, so every endpoint shares its two
        latency wins:

        * BM25 and dense search run side by side. They are independent, and
          both spend their time in native code (numpy, faiss, torch) that
          releases the GIL, so the cost is the slower of the two, not the sum.
        * Results are cached. The corpus is static and every stage is
          deterministic, so a repeated query - a demo, a retry, /papers then
          /ask on the same text, a shared link - skips search AND the
          reranker, which is the dominant cost on CPU.
        """
        key = " ".join(question.lower().split())
        with self._cache_lock:
            hit = self._rank_cache.get(key)
            if hit is not None:
                self._rank_cache.move_to_end(key)
                self.cache_hits += 1
                return dict(hit)
        query_map = {PSEUDO_QID: question}
        with self._gpu_lock:
            bm25_future = _SEARCH_POOL.submit(self.bm25.search, query_map, self.retrieve_k)
            dense_run = self.dense.search(query_map, top_k=self.retrieve_k)
            fused = reciprocal_rank_fusion(
                [bm25_future.result(), dense_run], k=self.rrf_k, top_k=self.retrieve_k
            )
            ranked = self._rerank_adaptive(query_map, fused[PSEUDO_QID])
        with self._cache_lock:
            self._rank_cache[key] = dict(ranked)
            while len(self._rank_cache) > RANK_CACHE_MAX:
                self._rank_cache.popitem(last=False)
        return ranked

    def _rerank_adaptive(self, query_map: dict[str, str], fused: dict[str, float]) -> dict[str, float]:
        """Cross-encoder scores for the fused candidates, stopping early when sure.

        The cross-encoder is the dominant cost of a search and scales with the
        candidates scored (seconds on a 2-vCPU CPU). Score the best `first`
        fused candidates; if their top-5 mean already clears `tau`, the answer
        is obvious and the remaining candidates are not worth scoring.
        Otherwise score the rest, so a hard query still sees every candidate.

        Measured offline (scripts/eval_early_exit.py): recall@10 unchanged,
        ~27% fewer candidates scored.
        """
        ee = self.cfg["retrieval"].get("early_exit")
        if not ee or len(fused) <= ee["first"]:
            reranked, _ = self.reranker.rerank(query_map, {PSEUDO_QID: fused}, self.corpus,
                                               top_k=self.retrieve_k)
            return reranked[PSEUDO_QID]
        order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        head = dict(order[: ee["first"]])
        scores, _ = self.reranker.rerank(query_map, {PSEUDO_QID: head}, self.corpus,
                                         top_k=len(head))
        scores = scores[PSEUDO_QID]
        top = sorted(scores.values(), reverse=True)[:5]
        if sum(top) / len(top) >= ee["tau"]:
            self.early_exits += 1
            return scores
        tail = dict(order[ee["first"]:])
        rest, _ = self.reranker.rerank(query_map, {PSEUDO_QID: tail}, self.corpus,
                                       top_k=len(tail))
        return {**scores, **rest[PSEUDO_QID]}

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Hybrid retrieve, fuse, rerank, then cap chunks per paper."""
        reranked = {PSEUDO_QID: self._ranked(question)}

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
        reranked = {PSEUDO_QID: self._ranked(topic)}

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
            live = self.search_live(topic, top_k=top_k)
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
                            "authors": p.get("authors", []),
                            "abstract": p["text"],
                            "published": p["published"],
                            "primary_category": "",
                            "arxiv_url": p["arxiv_url"],
                            "venue": p.get("venue", ""),
                            "cited_by_count": p.get("cited_by_count"),
                            "live_source": p.get("live_source", "arxiv"),
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
                        # Corpus papers only fill remaining slots if they clear
                        # the same bar - otherwise a 0.08 match sat under a
                        # result labelled confident.
                        papers = (fetched + [p for p in papers if p["score"] >= floor])[:top_k]
                        source = "arxiv_live"
                        confident = True
                        mean_top = sum(p["score"] for p in fetched) / len(fetched)
                        best_score = max(p["score"] for p in fetched)

        # v2: nothing convincing in the corpus or on arXiv - diagnose why.
        diagnosis, diagnosis_ms = None, 0.0
        if not confident:
            diagnosis, diagnosis_ms = self._diagnose(topic, papers, None)

        return {
            "topic": topic,
            "source": source,
            "papers": papers,
            "confident": confident,
            "top_score": round(best_score, 3),
            "mean_top_score": round(mean_top, 3),
            "diagnosis": diagnosis,
            "timing_ms": {"total": round((time.perf_counter() - start) * 1000.0, 1),
                          "diagnosis": round(diagnosis_ms, 1)},
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
                model=self._answer_models(),
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

    # ------------------------------------------------------------ live search

    def search_live(self, question: str, top_k: int = 6) -> list[dict[str, Any]]:
        """Search arXiv and OpenAlex live, for questions the corpus cannot answer.

        The corpus is a sample, not coverage: ~1,400 papers per category across
        nine months, against the 50-150/day arXiv actually publishes. Most
        papers a user asks about were never ingested, and telling them "not in
        this corpus" is honest but useless when the paper plainly exists.

        Two sources, because each misses what the other has. arXiv is the
        freshest for preprints; OpenAlex adds conference and journal papers
        that never had one. Both run in parallel under ONE wall-clock budget
        (`citation.live_budget_s`), so adding a source never adds latency:
        whatever has answered when the budget expires is used, and a source
        that is down or paused costs nothing. The same paper found by both
        collapses to one entry, keyed on its arXiv id or normalised title.

        Abstracts only - no PDF fetch. Results are rescored by the corpus
        reranker afterwards, so one confidence bar governs every source.
        """
        sources = self.cfg["citation"].get("live_sources", ["arxiv", "openalex"])
        budget = float(self.cfg["citation"].get("live_budget_s", 5.0))
        deadline = time.monotonic() + budget
        jobs = {}
        if "arxiv" in sources:
            jobs["arxiv"] = _LIVE_POOL.submit(self._live_arxiv, question, top_k, deadline)
        if "openalex" in sources:
            jobs["openalex"] = _LIVE_POOL.submit(self._live_openalex, question, top_k, deadline)

        # Do not wait for the slowest source. Once one has results, the other
        # gets a short grace and is then left behind: arXiv often takes 3-5 s
        # while OpenAlex answers in ~1 s, and its extra papers are not worth
        # holding the whole response. A source that returns nothing (down,
        # paused, empty) does not start the clock - the other keeps its full
        # budget.
        grace = float(self.cfg["citation"].get("live_grace_s", 0.75))
        found: dict[str, list[dict[str, Any]]] = {}
        name_of = {job: name for name, job in jobs.items()}
        pending = set(jobs.values())
        cutoff = deadline
        while pending and time.monotonic() < cutoff:
            done, pending = _futures_wait(
                pending, timeout=max(cutoff - time.monotonic(), 0.01),
                return_when=FIRST_COMPLETED)
            for job in done:
                try:
                    found[name_of[job]] = job.result()
                except Exception:  # noqa: BLE001 - a failed source is an empty source
                    found[name_of[job]] = []
                if found[name_of[job]]:
                    cutoff = min(cutoff, time.monotonic() + grace)

        merged: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        # arXiv first: on a duplicate, its record wins (fresher version), and
        # OpenAlex contributes the venue and citation count it lacks.
        for name in ("arxiv", "openalex"):
            for item in found.get(name, []):
                key = _dedup_key(item)
                if key in seen:
                    kept = seen[key]
                    kept["venue"] = kept.get("venue") or item.get("venue", "")
                    kept["cited_by_count"] = max(kept.get("cited_by_count") or 0,
                                                 item.get("cited_by_count") or 0)
                    kept["live_source"] = "arxiv+openalex"
                    continue
                seen[key] = item
                merged.append(item)
        return merged

    # Kept for callers that predate OpenAlex.
    search_arxiv_live = search_live

    def _live_arxiv(self, question: str, top_k: int, deadline: float) -> list[dict[str, Any]]:
        terms = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
                 if w.lower() not in _QUERY_STOPWORDS][:12]
        if not terms:
            return []
        queries = [
            "all:" + " AND ".join(f'"{t}"' for t in terms[:4]),  # precise first
            "all:" + " ".join(terms[:8]),                         # then looser
        ]
        found: list[Paper] = []
        for query in queries:
            remaining = deadline - time.monotonic()
            if remaining <= 0.25:
                break
            try:
                found = fetch_metadata_bounded(query, max_results=top_k,
                                               budget_s=remaining)
            except ArxivUnavailable:
                return []  # a failed enrichment is not a failed request
            if found:
                break
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
                "authors": p.authors[:8],
                "venue": "arXiv",
                "cited_by_count": None,
                "live_source": "arxiv",
            }
            for p in found
        ]

    def _live_openalex(self, question: str, top_k: int, deadline: float) -> list[dict[str, Any]]:
        remaining = deadline - time.monotonic()
        if remaining <= 0.25:
            return []
        try:
            works = openalex.search_works(question, max_results=top_k, budget_s=remaining)
        except openalex.OpenAlexUnavailable:
            return []
        return [
            {
                "chunk_id": f"live:{w['paper_id']}",
                "doc_id": f"live:{w['paper_id']}",
                "paper_id": w["paper_id"],
                # Named arxiv_url for compatibility with every client; for a
                # paper with no arXiv copy it is the DOI or landing page.
                "arxiv_url": w["url"],
                "score": 1.0,
                "title": w["title"],
                "section": "abstract",
                "published": w["published"],
                "text": w["abstract"],
                "authors": w["authors"][:8],
                "venue": w["venue"],
                "cited_by_count": w["cited_by_count"],
                "doi": w["doi"],
                "live_source": "openalex",
            }
            for w in works
        ]

    # ------------------------------------------------------------------- ask

    def answer(self, question: str, generate: bool = True) -> dict[str, Any]:
        """Question answering, including the US5 rejection path."""
        start = time.perf_counter()
        passages = self.retrieve(question)
        retrieval_ms = (time.perf_counter() - start) * 1000.0
        # Kept for diagnosis: if everything below fails, these weak corpus hits
        # are exactly what the critic needs to see to say what went wrong.
        corpus_passages = passages

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

        # Timed on its own: this was the one stage outside every timer, which is
        # how a 45s arXiv stall reported itself as "retrieval_and_rerank: 1783ms".
        live_ms = 0.0
        if remote_ok and (not confident or mean_top < enrich_below):
            live_start = time.perf_counter()
            live = self.search_live(question)
            live_ms = (time.perf_counter() - live_start) * 1000.0
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

        # Answer from the papers when they cover the question. When they do
        # not - nothing retrieved, or the answerer replies NOT_COVERED - fall
        # back to the model's own knowledge, but only for AI questions and
        # labelled as not grounded in the papers. Everything else is declined.
        grounded = True
        uncovered = not passages
        if passages and generate:
            gen_start = time.perf_counter()
            answer_text = generate_answer(
                question, passages, model=self._answer_models(), max_tokens=ac["max_tokens"]
            )
            generation_ms = (time.perf_counter() - gen_start) * 1000.0
            if is_not_covered(answer_text):
                uncovered, answer_text = True, None
        if uncovered and generate and ac.get("general_knowledge", True):
            gen_start = time.perf_counter()
            try:
                general = general_answer(question, model=self._answer_models(),
                                         max_tokens=ac["max_tokens"])
            except Exception as exc:  # noqa: BLE001 - fall back to the plain decline
                print(f"[ml] general answer failed ({type(exc).__name__})", flush=True)
                general = None
            generation_ms += (time.perf_counter() - gen_start) * 1000.0
            if general:
                answer_text, source, grounded, confident = general, "general_knowledge", False, False
        if uncovered and answer_text is None:
            answer_text = (
                "Nothing answers that - not in the local corpus, and not in a live "
                "arXiv search either. If this is an AI/ML topic the paper may simply "
                "be too new or too obscure; if it is not, this tool only covers AI research."
                if not passages else
                "The papers found for this do not contain an answer to it."
            )

        # v2: the gate fired and nothing rescued it, so say what went wrong.
        diagnosis, diagnosis_ms = None, 0.0
        if not passages and source != "general_knowledge":
            diagnosis, diagnosis_ms = self._diagnose(question, corpus_passages, None)

        return {
            "question": question,
            "answer": answer_text,
            "confident": confident,
            "source": source,
            "grounded": grounded,
            "top_score": round(best, 3),
            "mean_top_score": round(mean_top, 3),
            "citations": passages,
            "distinct_papers": distinct_papers(passages),
            "generation_configured": is_configured(),
            "diagnosis": diagnosis,
            "timing_ms": {
                "retrieval_and_rerank": round(retrieval_ms, 1),
                "arxiv_live": round(live_ms, 1),
                "generation": round(generation_ms, 1),
                "diagnosis": round(diagnosis_ms, 1),
                "total": round(retrieval_ms + live_ms + generation_ms + diagnosis_ms, 1),
            },
        }

    def _answer_models(self) -> list[str]:
        """Primary answer model, then fallbacks for when its daily cap is spent."""
        ac = self.cfg["answer"]
        return [ac["model"], *ac.get("fallback_models", [])]

    def _diagnose(self, question: str, passages: list[dict[str, Any]],
                  answer: str | None) -> tuple[dict[str, Any] | None, float]:
        """Run the v2 failure diagnosis, if enabled. Never raises."""
        dc = self.cfg.get("diagnosis", {})
        if not dc.get("enabled", True):
            return None, 0.0
        t0 = time.perf_counter()
        d = diagnose(question, passages, answer,
                     model=dc.get("models") or dc.get("model", self.cfg["answer"]["model"]),
                     budget_s=float(dc.get("budget_s", 6.0)))
        return (d.to_json() if d else None), (time.perf_counter() - t0) * 1000.0

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
                "arxiv_url": paper_url(paper),
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
                "diagnosis": answer.get("diagnosis"),
                # The complete /ask payload, so the client can render it
                # directly. Without this the web client discarded the answer
                # computed here and called /ask, re-running retrieval,
                # reranking and generation - every Explain-mode question cost
                # two full pipeline passes and two Groq calls.
                "answer_result": answer,
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
                model=self._answer_models(),
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
                "arxiv_url": paper_url(paper),
                "venue": (paper.metadata or {}).get("venue", ""),
                "doi": (paper.metadata or {}).get("doi", ""),
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
