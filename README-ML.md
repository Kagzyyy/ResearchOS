# ResearchOS-ML — v1

Question answering over recent arXiv ML/AI papers. Phase 1 of the plan in
[REQUIREMENTS-ML.md](REQUIREMENTS-ML.md): **the usable tool**. The evaluation harness
(phase 2) is deliberately not built yet.

Shares its retrieval, reranking, generation and logging with the parent project
[ResearchOS](README.md). What is new here is everything the corpus forced.

---

## Why this exists

Ask ResearchOS about JEPA and it returns *"The Bone and Joint Decade 2000–2010"* — it
matched the word "joint" against a corpus of 5,183 biomedical abstracts containing zero
ML papers. That failure is what this project fixes.

---

## What v1 does

```
arXiv API ──► version dedup ──► PDF ──► section-aware ──► BM25 + dense
 (topic       (FR6.2)          text     chunking          hybrid index
  queries)                     (PyMuPDF) (FR2)                │
                                                              ▼
                                                    RRF fusion (k=60)
                                                              │
                                                              ▼
                                              cross-encoder reranker
                                                              │
                                                              ▼
                                              cap 2 chunks/paper (FR4.2)
                                                              │
                                                              ▼
                                         score ≥ threshold?  ──no──► "not in corpus"
                                                              │yes
                                                              ▼
                                            Claude, answer with citations
```

---

## Three decisions worth defending

### 1. Off-the-shelf reranker, not a fine-tuned one

The parent project's headline ablation was a **null result** (p = 0.806), and most of its
measured +0.048 came from *domain adaptation*, not negative selection. Adapting from
biomedical to ML text would likely repeat that gain — but it needs labels, and arXiv has
none. Spending that time on chunking instead buys more for a tool you actually use.

Phase 2 revisits this once tier-1 citation labels exist. Deferring is not abandoning.

### 2. BM25 is load-bearing, not a baseline

`bge-small-en-v1.5` has frozen weights and has never seen "JEPA", "GRPO" or "Mamba" — it
embeds them as near-noise. BM25 gives a rare token a very high IDF weight, so exact
matching on novel terminology is exactly its strength.

**For a corpus whose vocabulary is coined monthly, the lexical arm is the only defence
against words the embedding model was never trained on.** Dropping it as an optimisation
would be the single worst change available.

### 3. Full text, not abstracts only

Abstracts-only would have been dramatically cheaper — no PDF parsing, no chunking, no
per-paper capping, an order of magnitude less storage. But methods live in the body of a
paper, and both "explain" and "compare" queries need method detail. Abstracts-only would
have reduced the system to comparing marketing summaries.

The cost is real: PDF download and parsing dominates ingestion time.

---

## What the corpus forced that SciFact never did

| | SciFact | Here |
|---|---|---|
| Retrievable unit | one abstract = one document | one paper = dozens of chunks |
| Result diversity | automatic | **must be enforced** (`cap_per_paper`) |
| Duplicates | none | **v1/v2/v3 of the same paper** (`dedup_by_version`) |
| Out-of-corpus queries | rare | common — needs a rejection threshold |
| Vocabulary | stable | drifts monthly |

**Per-paper capping** is the one that surprises. The five best-scoring chunks for
"explain JEPA" are routinely five consecutive paragraphs of one paper — correct by score,
useless as an answer. `answer.max_per_paper` forces multi-source results.

**Version dedup** matters more than it looks. Two near-identical copies of a paper in the
index split its retrieval signal and would corrupt any future evaluation over it.

**The rejection threshold** acts on a signal the parent project measured but never used:
an out-of-domain query scored +0.06 where an in-domain one scored +0.60, an order of
magnitude apart. Below `answer.min_score` the system says it does not know.

---

## Usage

Ingest (resumable — PDFs and extracted text are cached by arXiv id):

```bash
python scripts/ingest_arxiv.py --config configs/ml.yaml
```

Ask:

```bash
python scripts/ask.py --config configs/ml.yaml "explain JEPA"
```

Interactive:

```bash
python scripts/ask.py --config configs/ml.yaml --interactive
```

Without `ANTHROPIC_API_KEY` the retrieval and reranking still run and you get ranked
passages with no generated answer.

Serve the HTTP API instead:

```bash
RESEARCHOS_ML_CONFIG=configs/ml.yaml python -m uvicorn src.api.ml_app:app --port 8100
```

### Changing what it knows about

`configs/ml.yaml` → `ingest.queries` is a list of arXiv API query strings. v1 is
**topic-scoped on purpose**: a few thousand full-text papers on subjects you actually ask
about beats a thin random slice of all of cs.LG. Widen the queries to widen coverage; the
cost is ingest time and disk.

---

## Refresh

`ingest.queries` sorts by submission date, so re-running picks up new papers. Version
dedup means a re-run that finds `v3` of an already-indexed paper replaces `v2` rather
than adding a near-duplicate.

Delete `cache/ml/` after ingesting to force index rebuild — v1 rebuilds indices from
scratch rather than updating them incrementally, which is fine at this corpus size and is
the first thing to change if the corpus grows an order of magnitude.

---

## Limitations

Honest bounding, same as the parent project.

- **No evaluation. No metrics. No results table.** This is phase 1. There is no measured
  claim here, and nothing in this README should be read as one. Phase 2 builds the
  two-tier label set specified in [REQUIREMENTS-ML.md §5](REQUIREMENTS-ML.md).
- **The corpus is topic-scoped, not comprehensive.** It covers what `ingest.queries` asks
  for. Questions outside that return the rejection message — which is correct behaviour,
  but it is a narrow slice of arXiv, not arXiv.
- **Refresh is manual.** Re-running the script is the refresh mechanism. There is no
  scheduler, and index rebuild is full rather than incremental.
- **PDF extraction is imperfect.** Two-column layouts sometimes interleave, and papers
  whose headings do not match the expected pattern fall back to a single unnamed section.
  Equations, tables and figures are dropped entirely.
- **The rejection threshold is uncalibrated.** `answer.min_score` is a hand-set default,
  not a value derived from labelled data. It is the first thing phase 2 should calibrate.
- **"Survey what's new" is not supported.** Date-filtered multi-document summarisation was
  explicitly deferred during requirements discovery.
- **Generation is unevaluated**, as in the parent project. Retrieval is the component
  under scrutiny; answer faithfulness is not measured.
