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

## The corpus as built

| | |
|---|---|
| Papers | **903** unique, after version dedup |
| With full text | 500 (PDF cap; the rest are abstract-only) |
| Chunks | **18,882** (20.9 per paper) |
| Newest paper | same-day — the API returns submissions from hours ago |
| Query latency | **~700–800 ms** end to end, retrieval + rerank, on an RTX 4060 |
| Disk | 2.7 GB PDFs, 37 MB extracted text |

Chunks by section — the chunker is finding real structure, not guessing:

```
introduction 3691 · method 3035 · related work 2194 · model 1910
experiments 1155 · abstract 903 · results 818 · conclusion 779
```

`abstract = 903` is exactly one per paper, which is the check that the
abstract-deduplication in `chunking.py` is working.

### Behaviour on real queries

*"explain JEPA"* → **5 distinct papers** in 6 passages at 792 ms, including a critique
(*"Asleep at the Wheel: JEPA's Limitations"*) and a theory paper
(*"A Generalization Theory for JEPA-Based World Models"*) alongside the primary ones.

*"how does JEPA differ from masked autoencoders"* → **all 6 passages mention both**
JEPA and MAE, across 5 papers, drawn from the `related work` and `preliminaries`
sections where papers actually contrast methods. That is the FR4.2 cap and full-text
chunking earning their keep together.

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

### The rejection threshold, and why it is a mean and not a maximum

The obvious design — reject when the best passage scores below a threshold — **does not
work**, and measuring it is what showed why:

| Query | max | mean of top-5 | in corpus? |
|---|---|---|---|
| explain JEPA | 0.998 | 0.995 | yes |
| how does JEPA differ from MAE | 0.898 | 0.671 | yes |
| what causes knee osteoarthritis | **0.670** | 0.190 | **no** |
| vitamin D and infection | 0.150 | 0.071 | no |
| sourdough hydration ratio | 0.001 | 0.001 | no |
| who won the 1998 football world cup | 0.348 | 0.190 | no |

The osteoarthritis query hits **0.670 on max** — high enough to defeat any max-based
threshold — because the corpus genuinely contains a self-supervised *bone densitometry*
paper. One real partial match is all it takes.

The mean separates cleanly, because a question that belongs to this corpus matches
*many* passages rather than one. Threshold: **mean of top-5 ≥ 0.35**, which classifies
all eight probe queries correctly.

**A scale trap worth knowing:** `CrossEncoder.predict` applies a sigmoid, so torch-backend
scores are probabilities in `[0, 1]`. The ONNX backend returns **raw logits**. A threshold
tuned on one is meaningless on the other. The first version of this config used `-6.0`,
which could never fire against a sigmoid output — the rejection path was silently dead
code until it was measured.

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
- **The rejection threshold is tuned on eight probe queries, not labelled data.**
  `answer.min_mean_score` separates those eight correctly, but eight is not a validation
  set. Phase 2 should calibrate it properly and report a false-accept / false-reject rate.
- **"Survey what's new" is not supported.** Date-filtered multi-document summarisation was
  explicitly deferred during requirements discovery.
- **Generation is unevaluated**, as in the parent project. Retrieval is the component
  under scrutiny; answer faithfulness is not measured.
