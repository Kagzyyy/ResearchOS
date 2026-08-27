"""Evaluate ResearchOS-ML retrieval, and compare configurations.

    python scripts/eval_retrieval.py --config configs/ml.yaml

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
arXiv ships no relevance judgements. Citation-context mining (the principled
approach, see src/data/citation_eval.py) yielded only 12 usable pairs on this
corpus, because papers from 2025-11 onward cite mostly older work that is not
indexed. Twelve queries is noise, not an evaluation.

So the primary signal here is **passage-to-paper retrieval**: take a sentence
from the body of a paper, use it as the query, and ask whether the system
returns that paper. Gold labels are free and exact - the sentence provably came
from that paper - and there are thousands of them.

This is a real retrieval test and it is the right shape for the Find-papers
use case: you read a claim, you want the paper it came from.

It is also EASIER than a natural-language query, because the query is verbatim
text from the target document. That favours lexical matching, so:

  - absolute numbers here are optimistic; do not compare them to SciFact or to
    published BEIR results;
  - the value is in the COMPARISON between configurations, which all face the
    same advantage, so the ranking between them is informative even though the
    absolute level is inflated.

Queries are drawn from method/experiment sections, never the abstract, because
abstracts are indexed verbatim as their own chunk and would make the task
trivial rather than merely easy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import device, load_config, seed_everything
from src.data.arxiv_source import read_papers
from src.data.chunking import chunks_to_corpus, read_chunks
from src.rerank.evaluate import evaluate_run, latency_stats
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
BODY_SECTIONS = {"method", "methods", "methodology", "approach", "experiments",
                 "experimental setup", "results", "analysis", "model", "architecture"}
MIN_WORDS, MAX_WORDS = 12, 45


def build_queries(
    chunks_path: str, papers_path: str, n_queries: int, rng: random.Random
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    """Sample body sentences as queries; gold is the paper they came from.

    Also returns query_id -> source chunk_id. That chunk is excluded at scoring
    time: otherwise the query is verbatim text inside it, BM25 matches it
    exactly, and the benchmark saturates at nDCG 0.99 without discriminating
    between configurations. Holding it out forces the system to identify the
    paper from its OTHER passages, which is the thing actually worth measuring.
    """
    chunks = read_chunks(chunks_path)
    full_text_papers = {p.paper_id for p in read_papers(papers_path) if p.full_text}

    by_paper: dict[str, list[tuple[str, str]]] = {}
    for chunk in chunks:
        if chunk.paper_id in full_text_papers and chunk.section in BODY_SECTIONS:
            by_paper.setdefault(chunk.paper_id, []).append((chunk.chunk_id, chunk.text))

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    sources: dict[str, str] = {}
    candidates = sorted(by_paper)
    rng.shuffle(candidates)

    for paper_id in candidates:
        if len(queries) >= n_queries:
            break
        # Only usable if the paper has more than one body chunk - otherwise
        # holding out the source leaves nothing of the paper to find.
        if len(by_paper[paper_id]) < 2:
            continue
        sentences = [
            (cid, s.strip())
            for cid, text in by_paper[paper_id]
            for s in SENTENCE_RE.split(text)
            if MIN_WORDS <= len(s.split()) <= MAX_WORDS
        ]
        sentences = [
            (cid, s) for cid, s in sentences
            if not s.lower().startswith(("in this section", "table", "figure"))
        ]
        if not sentences:
            continue
        qid = f"q{len(queries)}"
        source_chunk, sentence = rng.choice(sentences)
        queries[qid] = sentence
        qrels[qid] = {paper_id: 1}
        sources[qid] = source_chunk

    return queries, qrels, sources


def to_paper_run(
    chunk_run: dict[str, dict[str, float]],
    corpus: dict[str, dict[str, Any]],
    top_k: int,
    sources: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Collapse a chunk-level run to a paper-level run, scoring by best chunk.

    `sources` maps query -> the chunk its sentence was taken from; that chunk is
    dropped so the query cannot match itself verbatim.
    """
    paper_run: dict[str, dict[str, float]] = {}
    for qid, hits in chunk_run.items():
        best: dict[str, float] = {}
        held_out = (sources or {}).get(qid)
        for chunk_id, score in hits.items():
            if chunk_id == held_out:
                continue
            pid = corpus[chunk_id]["paper_id"]
            if score > best.get(pid, float("-inf")):
                best[pid] = float(score)
        paper_run[qid] = dict(
            sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        )
    return paper_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--n-queries", type=int, default=300)
    parser.add_argument("--out", default="results/ml_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    rng = random.Random(seed)
    paths, rc = cfg["paths"], cfg["retrieval"]

    queries, qrels, sources = build_queries(
        paths["chunks"], paths["papers"], args.n_queries, rng
    )
    print(
        f"[eval] {len(queries)} queries | gold = the paper each sentence came from "
        "| source chunk held out"
    )

    chunks = read_chunks(paths["chunks"])
    corpus = chunks_to_corpus(chunks)
    doc_ids = list(corpus)
    doc_texts = [f"{corpus[d]['title']} {corpus[d]['text']}".strip() for d in doc_ids]
    print(f"[eval] corpus: {len(doc_ids)} chunks")

    k_values = [1, 5, 10, 50]
    top_k = rc["top_k"]
    results: dict[str, Any] = {"seed": seed, "n_queries": len(queries), "source_chunk_held_out": True, "rows": {}}

    def record(name: str, label: str, chunk_run, ms: list[float], extra: dict | None = None) -> None:
        paper_run = to_paper_run(chunk_run, corpus, top_k, sources)
        metrics = evaluate_run(qrels, paper_run, k_values)
        results["rows"][name] = {
            "label": label,
            "metrics": metrics,
            "latency": latency_stats(ms),
            **(extra or {}),
        }
        print(
            f"[eval] {label:34s} nDCG@10={metrics['ndcg@10']:.4f}  "
            f"R@10={metrics['recall@10']:.4f}  MAP={metrics['map@10']:.4f}"
        )

    # --- BM25 ---
    bm25 = BM25Retriever(k1=rc["bm25"]["k1"], b=rc["bm25"]["b"], stemmer_lang=rc["bm25"]["stemmer"])
    bm25.index(doc_ids, doc_texts, cache_dir=paths["index_cache"])
    t0 = time.perf_counter()
    bm25_run = bm25.search(queries, top_k=top_k)
    bm25_ms = [(time.perf_counter() - t0) * 1000.0 / len(queries)] * len(queries)
    record("bm25", "BM25 only", bm25_run, bm25_ms)

    # --- Dense ---
    dense = DenseRetriever(
        model_name=rc["dense"]["model"],
        batch_size=rc["dense"]["batch_size"],
        max_length=rc["dense"]["max_length"],
        query_prefix=rc["dense"]["query_prefix"],
        device=device(),
    )
    dense.index_corpus(doc_ids, doc_texts, cache_dir=paths["index_cache"])
    t0 = time.perf_counter()
    dense_run = dense.search(queries, top_k=top_k)
    dense_ms = [(time.perf_counter() - t0) * 1000.0 / len(queries)] * len(queries)
    record("dense", "Dense only (bge-small)", dense_run, dense_ms, {"params": "33M"})

    # --- RRF ---
    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion([bm25_run, dense_run], k=rc["fusion"]["rrf_k"], top_k=top_k)
    fuse_ms = [(time.perf_counter() - t0) * 1000.0 / len(queries) + b + d
               for b, d in zip(bm25_ms, dense_ms)]
    record("rrf", "BM25 + dense (RRF)", fused, fuse_ms, {"params": "33M"})

    # --- + reranker ---
    from src.rerank.model import Reranker

    mc = cfg["reranker"]
    reranker = Reranker(
        model_name_or_path=mc["model"],
        max_length=mc["max_length"],
        batch_size=mc["batch_size"],
        device=device(),
        fp16=mc.get("fp16", True) and device() == "cuda",
    )
    reranked, rerank_ms = reranker.rerank(queries, fused, corpus, top_k=top_k)
    end_to_end = [f + r for f, r in zip(fuse_ms, rerank_ms)]
    record("rrf_rerank", "+ cross-encoder reranker", reranked, end_to_end, {"params": "278M"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
