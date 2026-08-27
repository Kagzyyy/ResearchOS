"""Evaluate ResearchOS-ML retrieval, and compare configurations.

    python scripts/eval_retrieval.py --config configs/ml.yaml
    python scripts/eval_retrieval.py --benchmark generated --judge

TWO BENCHMARKS, AND WHY BOTH
----------------------------
arXiv ships no relevance judgements, so a test set had to be manufactured.
Citation-context mining - the principled route, kept in src/data/citation_eval.py
- yielded 12 usable pairs on this corpus, because papers from 2025-11 onward cite
mostly older work that is not indexed. Twelve queries is noise.

  sentence   A verbatim sentence from a paper's body is the query; gold is that
             paper. The chunk it came from is HELD OUT, so the paper must be
             found through its other passages. Free, exact labels - but the
             query is literal text from the target, which hands BM25 an exact
             lexical match. Absolute numbers are optimistic.

  generated  A model reads the paper and writes the question a researcher would
             type to find it (scripts/generate_eval_queries.py). The label is
             still structural - the question was written from that paper - but
             the phrasing is natural, which is what users actually type. The
             prompt forbids reusing the title, and queries that copy it anyway
             are rejected.

The `generated` set is the better proxy for the Find-papers feature. The
`sentence` set is kept because it is label-noise-free and shows how much of any
result is a lexical artifact.

FALSE NEGATIVES, AND --judge
----------------------------
Both benchmarks mark exactly one paper relevant per query, so a genuinely good
result that is not the source paper counts as a miss. `--judge` pools the top-k
from every configuration, has a model judge only that pool, and adds the
positives. That recovers most of the loss - at the cost of being a SYNTHETIC
metric, so it is reported as a secondary row, never the headline.
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
K_VALUES = [1, 5, 10, 50]


# ------------------------------------------------------------------ benchmarks

def sentence_benchmark(
    chunks_path: str, papers_path: str, n_queries: int, rng: random.Random
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    """Body sentence as query; gold is its paper; the source chunk is held out."""
    chunks = read_chunks(chunks_path)
    full_text = {p.paper_id for p in read_papers(papers_path) if p.full_text}

    by_paper: dict[str, list[tuple[str, str]]] = {}
    for chunk in chunks:
        if chunk.paper_id in full_text and chunk.section in BODY_SECTIONS:
            by_paper.setdefault(chunk.paper_id, []).append((chunk.chunk_id, chunk.text))

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    sources: dict[str, str] = {}
    candidates = sorted(by_paper)
    rng.shuffle(candidates)

    for paper_id in candidates:
        if len(queries) >= n_queries:
            break
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
        qid = f"s{len(queries)}"
        source_chunk, sentence = rng.choice(sentences)
        queries[qid], qrels[qid], sources[qid] = sentence, {paper_id: 1}, source_chunk

    return queries, qrels, sources


def generated_benchmark(
    path: str, n_queries: int
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    """Model-written natural-language queries; gold is the source paper."""
    from src.data.query_gen import read_generated

    items = read_generated(path)[:n_queries]
    queries = {i.query_id: i.query for i in items}
    qrels = {i.query_id: {i.gold_paper_id: 1} for i in items}
    return queries, qrels, {}


# ------------------------------------------------------------------- machinery

def to_paper_run(
    chunk_run: dict[str, dict[str, float]],
    corpus: dict[str, dict[str, Any]],
    top_k: int,
    sources: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Collapse chunk-level hits to paper-level, scoring each paper by its best chunk."""
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
        paper_run[qid] = dict(sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
    return paper_run


def run_all_configs(
    queries: dict[str, str],
    corpus: dict[str, dict[str, Any]],
    doc_ids: list[str],
    doc_texts: list[str],
    cfg: dict[str, Any],
    sources: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Run every retrieval configuration over one query set."""
    rc, paths = cfg["retrieval"], cfg["paths"]
    top_k = rc["top_k"]
    out: dict[str, dict[str, Any]] = {}

    bm25 = BM25Retriever(k1=rc["bm25"]["k1"], b=rc["bm25"]["b"], stemmer_lang=rc["bm25"]["stemmer"])
    bm25.index(doc_ids, doc_texts, cache_dir=paths["index_cache"])
    t0 = time.perf_counter()
    bm25_run = bm25.search(queries, top_k=top_k)
    bm25_ms = [(time.perf_counter() - t0) * 1000.0 / len(queries)] * len(queries)
    out["bm25"] = {"label": "BM25 only", "run": to_paper_run(bm25_run, corpus, top_k, sources),
                   "ms": bm25_ms, "params": None}

    dense = DenseRetriever(
        model_name=rc["dense"]["model"], batch_size=rc["dense"]["batch_size"],
        max_length=rc["dense"]["max_length"], query_prefix=rc["dense"]["query_prefix"],
        device=device(),
    )
    dense.index_corpus(doc_ids, doc_texts, cache_dir=paths["index_cache"])
    t0 = time.perf_counter()
    dense_run = dense.search(queries, top_k=top_k)
    dense_ms = [(time.perf_counter() - t0) * 1000.0 / len(queries)] * len(queries)
    out["dense"] = {"label": "Dense only (bge-small)", "run": to_paper_run(dense_run, corpus, top_k, sources),
                    "ms": dense_ms, "params": "33M"}

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion([bm25_run, dense_run], k=rc["fusion"]["rrf_k"], top_k=top_k)
    fuse_ms = [(time.perf_counter() - t0) * 1000.0 / len(queries) + b + d
               for b, d in zip(bm25_ms, dense_ms)]
    out["rrf"] = {"label": "BM25 + dense (RRF)", "run": to_paper_run(fused, corpus, top_k, sources),
                  "ms": fuse_ms, "params": "33M"}

    from src.rerank.model import Reranker

    mc = cfg["reranker"]
    reranker = Reranker(
        model_name_or_path=mc["model"], max_length=mc["max_length"],
        batch_size=mc["batch_size"], device=device(),
        fp16=mc.get("fp16", True) and device() == "cuda",
    )
    reranked, rerank_ms = reranker.rerank(queries, fused, corpus, top_k=top_k)
    out["rrf_rerank"] = {
        "label": "+ cross-encoder reranker",
        "run": to_paper_run(reranked, corpus, top_k, sources),
        "ms": [f + r for f, r in zip(fuse_ms, rerank_ms)],
        "params": "278M",
    }
    return out


def score(configs: dict[str, dict[str, Any]], qrels: dict[str, dict[str, int]], banner: str) -> dict[str, Any]:
    print(f"\n--- {banner} ---")
    rows: dict[str, Any] = {}
    for name, data in configs.items():
        metrics = evaluate_run(qrels, data["run"], K_VALUES)
        rows[name] = {"label": data["label"], "metrics": metrics,
                      "latency": latency_stats(data["ms"]), "params": data["params"]}
        print(f"  {data['label']:32s} nDCG@10={metrics['ndcg@10']:.4f}  "
              f"R@10={metrics['recall@10']:.4f}  MAP={metrics['map@10']:.4f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--benchmark", choices=["sentence", "generated", "both"], default="both")
    parser.add_argument("--n-queries", type=int, default=300)
    parser.add_argument("--generated-path", default="data/generated_eval.jsonl")
    parser.add_argument("--judge", action="store_true", help="pool + LLM-judge to recover false negatives")
    parser.add_argument("--judge-depth", type=int, default=10)
    parser.add_argument("--out", default="results/ml_metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = seed_everything(cfg["seed"])
    rng = random.Random(seed)
    paths = cfg["paths"]

    corpus = chunks_to_corpus(read_chunks(paths["chunks"]))
    doc_ids = list(corpus)
    doc_texts = [f"{corpus[d]['title']} {corpus[d]['text']}".strip() for d in doc_ids]
    print(f"[eval] corpus: {len(doc_ids)} chunks")

    results: dict[str, Any] = {"seed": seed, "benchmarks": {}}

    wanted = ["sentence", "generated"] if args.benchmark == "both" else [args.benchmark]
    for kind in wanted:
        if kind == "sentence":
            queries, qrels, sources = sentence_benchmark(
                paths["chunks"], paths["papers"], args.n_queries, rng
            )
            note = "verbatim body sentence; source chunk held out; favours lexical matching"
        else:
            gen_path = Path(args.generated_path)
            if not gen_path.exists():
                print(f"[eval] skipping generated: {gen_path} not found "
                      "(run scripts/generate_eval_queries.py)")
                continue
            queries, qrels, sources = generated_benchmark(str(gen_path), args.n_queries)
            note = "model-written natural-language query; label is structural"

        if not queries:
            continue
        print(f"\n[eval] benchmark '{kind}': {len(queries)} queries")
        configs = run_all_configs(queries, corpus, doc_ids, doc_texts, cfg, sources)
        block: dict[str, Any] = {
            "n_queries": len(queries),
            "note": note,
            "rows": score(configs, qrels, f"{kind}: structural labels (primary)"),
        }

        if args.judge and kind == "generated":
            from src.data.judge import build_pool, judge_pool

            papers_by_id = {p.paper_id: vars(p) for p in read_papers(paths["papers"])}
            pool = build_pool({k: v["run"] for k, v in configs.items()}, depth=args.judge_depth)
            merged = judge_pool(
                pool, queries, papers_by_id, qrels,
                model=cfg["answer"]["model"],
                out_path="data/pooled_judgements.jsonl",
            )
            block["rows_pooled"] = score(
                configs, merged, f"{kind}: + pooled LLM judgements (SECONDARY, synthetic)"
            )
            block["pooled_note"] = (
                "Relevance judged by an LLM over the pooled top-k of every configuration. "
                "Synthetic signal - secondary to the structural rows above."
            )

        results["benchmarks"][kind] = block

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
