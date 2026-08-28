# ResearchOS-ML

Search AI research papers, and de-reference the citations inside them.

You are reading a paper. You hit *"...following [23] (Assran et al., 2023)..."*. Paste
that reference in and get back: what the paper is, what it actually does, why it gets
cited, and an explanation in plain language — grounded in the paper's own text, with
every claim linked to a passage.

Built on the retrieval stack from the parent project, [ResearchOS](README.md).

## The corpus as shipped

| | |
|---|---|
| Papers | **8,330** unique, after version dedup |
| Chunks | **40,379** |
| With full text | 900 (PDF cap; the rest are abstract-indexed) |
| Date range | 2025-11-17 → 2026-08-26 |
| Query latency | **270 ms – 1.1 s** end to end on an RTX 4060 |

Categories: `cs.LG` 1517 · `cs.CV` 1268 · `cs.RO` 1146 · `cs.CL` 926 · `cs.AI` 832 ·
`cs.NE` 730 · `stat.ML` 513

Live checks against this corpus: **8/8** in-corpus vs out-of-corpus queries classified
correctly, **5/5** citations resolved to the right paper.

## Evaluation

arXiv ships no relevance judgements, so test sets had to be manufactured. Three attempts,
and the failures are as informative as the numbers.

**Citation contexts — abandoned.** Mine "...following [23]..." from full text, resolve the
reference, use the sentence as the query. Yielded **12 usable pairs**: the corpus starts
2025-11 and those papers cite older work that is not indexed. The miner is kept in
`src/data/citation_eval.py`; it needs a corpus with historical depth.

**Sentence benchmark.** A verbatim sentence from a paper's body is the query; gold is that
paper; the chunk it came from is held out. Free exact labels, but the query is literal text
from the target.

**Generated benchmark.** A model reads the paper and writes the question a researcher would
type to find it (`scripts/generate_eval_queries.py`). The label stays *structural* - the
question came from that paper - so the model generates the query, never the verdict. The
prompt forbids reusing the title and a 4-gram check rejects anything that does anyway.

150 queries each (`results/ml_metrics.json`):

### Sentence benchmark — favours lexical matching by construction

| Setup | nDCG@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| **BM25 only** | **0.8929** | 0.9267 | **0.8821** |
| Dense only (bge-small) | 0.5146 | 0.6333 | 0.4778 |
| BM25 + dense (RRF) | 0.7698 | **0.9333** | 0.7171 |
| + cross-encoder reranker | 0.8773 | 0.9267 | 0.8615 |

### Generated benchmark — natural-language queries

| Setup | nDCG@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| BM25 only | 0.9951 | 1.0000 | 0.9933 |
| Dense only (bge-small) | 0.9582 | 0.9933 | 0.9465 |
| BM25 + dense (RRF) | 0.9836 | 1.0000 | 0.9780 |
| + cross-encoder reranker | 0.9864 | 1.0000 | 0.9821 |

### Generated + pooled LLM judgements — secondary, synthetic

Top-10 of every configuration pooled and judged by a model, recovering the false negatives
that one-gold-per-query causes. An LLM's opinion, so it is never the headline.

| Setup | nDCG@10 | Recall@10 | MAP@10 |
|---|---|---|---|
| BM25 only | 0.9581 | 0.9650 | 0.9373 |
| Dense only (bge-small) | 0.9317 | 0.9594 | 0.9065 |
| BM25 + dense (RRF) | 0.9567 | 0.9744 | 0.9360 |
| **+ cross-encoder reranker** | **0.9612** | **0.9800** | **0.9403** |

### What these actually say

**BM25's dominance was an artifact of the sentence benchmark.** On the sentence set it
beats the full pipeline by 0.016 nDCG@10 and destroys dense retrieval (0.89 vs 0.51). On
natural-language queries that collapses: dense goes from **0.5146 to 0.9582**, and the four
configurations land within 0.04 of each other. The earlier conclusion that "the fusion
weighting is miscalibrated" does not survive contact with realistic queries.

**Under the most realistic conditions the full pipeline wins.** With pooled judgements -
natural queries, and credit for every relevant paper rather than only the source - the
reranker is top on nDCG@10 *and* Recall@10. That is the first setting in which the
architecture justifies itself.

**But the margin is 0.003 over BM25, on 150 queries, with no significance test.** Treat it
as "no longer losing", not as "wins". A paired bootstrap would almost certainly straddle
zero.

### The generated benchmark is saturated, and that is a flaw

Every configuration scores **0.958 or above**. A benchmark where everything is nearly
perfect cannot rank anything. The cause is the prompt: asking for 8-20 words describing
"the problem and the approach" produces a compressed abstract, and finding the source paper
among 8,330 from a compressed abstract is easy.

Real queries are shorter, vaguer and more ambiguous - "world models for robotics", not
"how to ground latent dynamics predictions with physical state supervision for control".
Discriminating between these configurations needs harder queries: fewer words, no method
description, deliberate ambiguity. That is the next iteration, and until it exists none of
the differences above should be treated as established.

---

## For daily use

### Start everything

```
start.cmd
```

Backend on :8100, web app on :3000. The backend needs 2-3 minutes to load its
models on a cold start; the UI says "Backend not reachable" until then, which is
expected rather than broken.

### Check it works

```bash
python scripts/healthcheck.py
```

Thirteen checks against the live HTTP endpoints - retrieval, the live arXiv
fallback, honest refusal, both citation formats, question routing, PDF upload,
document QA, generation, feedback, and whether the server logged any errors.
Exits non-zero on any failure, so it can gate a deploy.

### The browser extension

`extension/` — load it unpacked at `chrome://extensions` (Developer mode → Load
unpacked). Select a citation on any page and press **Alt+Shift+E**, or
right-click → *Explain with ResearchOS*. A panel opens in place with the paper
identified and explained.

This is the feature worth having. Everything else asks you to leave the paper
you are reading; this one does not.

Set the backend URL and access token in the extension popup. It talks to the
same `/explain` endpoint the web app uses, so nothing in the backend changes.

---

## Quick start

```bash
uv venv --python 3.11 && .venv\Scripts\activate.bat
uv pip install torch==2.6.0 --torch-backend=cu124
uv pip install -r requirements.txt
```

Get a **free** Groq key at [console.groq.com/keys](https://console.groq.com/keys), then:

```bash
copy .env.example .env
```

Put the key in `.env` as `GROQ_API_KEY=...`. Build the corpus (slow, resumable):

```bash
python scripts/ingest_arxiv.py --config configs/ml.yaml
```

Run the backend:

```bash
python scripts/serve_ml.py --config configs/ml.yaml
```

Run the web app:

```bash
cd web && npm install && npm run dev
```

Open `http://localhost:3000`. Or skip the browser entirely:

```bash
python scripts/ask.py --config configs/ml.yaml --interactive
```

**Without a Groq key everything still runs** — you get ranked passages, just no written
answer. Nothing silently fails.

---

## What it does

### Ask a question

Hybrid retrieval over the corpus, reranked by a cross-encoder, answered by Groq with
inline citations that link back to arXiv.

### Explain a citation

Paste any of these and it resolves to the actual paper:

```
[23] M. Assran et al. Self-Supervised Learning from Images with a
     Joint-Embedding Predictive Architecture. CVPR 2023.
arXiv:2301.08243
https://arxiv.org/abs/2301.08243
Masked Autoencoders Are Scalable Vision Learners
```

Resolution is **local-first, then remote**: the ingested corpus is checked by arXiv id
and then by title overlap; anything not found is fetched live from the arXiv API and its
PDF parsed on demand. **The tool is therefore not limited to what was ingested** — any
arXiv paper can be explained, indexed or not.

The explanation comes back in four parts — *What it is · The idea · Why it gets cited ·
In simpler terms* — plus the paper's own passages and **what other papers in the corpus
say about it**, which is where the wider context comes from.

### Ask your own PDF

Attach a paper in the Explain tab and every question is answered from that
document instead of the corpus.

Uploads are **not** added to the index. Rebuilding BM25 and FAISS over 40,000
chunks to accommodate one 30-chunk document would take minutes and be thrown
away; instead the cross-encoder scores every chunk of the paper directly. A
paper is only 20-40 chunks, so exhaustive scoring is both cheaper than
re-indexing and strictly better than retrieval, which could only approximate it.

Titles come from **font size**, not content. Content heuristics lose to licence
blocks - "Attention Is All You Need" opens with a multi-line Google permission
notice, so both "first substantial line" and keyword filtering name the paper
after the legal text. The title is the largest horizontal text on page 1 in
every paper format. Rotated spans are skipped, because arXiv stamps the id down
the left margin in a larger font than anything else on the page.

Limits: 25 MB, text-based PDFs only (a scan is rejected with an explanation
rather than silently producing nothing), 200 documents kept, content-addressed
so re-uploading the same file is free.

---

## Architecture

```
                    ┌─────────────────────── Next.js 16 · React 19 · Tailwind v4
                    │                        (web/)
                    ▼
              FastAPI :8100 ──── SQLite (queries, citations, clicks)
                    │
      ┌─────────────┴──────────────┐
      ▼                            ▼
  ASK path                    EXPLAIN path
      │                            │
      │                     resolve citation
      │                     ├─ arXiv id → local corpus
      │                     ├─ title overlap → local corpus
      │                     └─ miss → arXiv API + PDF on demand
      │                            │
      ▼                            ▼
  BM25 + bge-small ──► RRF ──► bge-reranker-base ──► cap 2 chunks/paper
                                       │
                                       ▼
                         mean-of-top-5 ≥ 0.35 ?  ──no──► "outside this corpus"
                                       │yes
                                       ▼
                          Groq · llama-3.3-70b-versatile
                          answer with [arXiv-id] citations
```

---

## Design decisions

### Groq, not a paid API

Free tier, OpenAI-compatible, fast. Model id lives in `configs/ml.yaml` under
`answer.model` because Groq rotates its catalogue — if you see a decommissioned-model
error, change that line, not the code.

### BM25 is load-bearing, not a baseline

`bge-small-en-v1.5` has frozen weights and has never seen "JEPA", "GRPO" or "Mamba" — it
embeds them as near-noise. BM25 gives a rare token a very high IDF weight, so exact
matching on novel terminology is precisely its strength.

**For a corpus whose vocabulary is coined monthly, the lexical arm is the only defence
against words the embedding model was never trained on.** Removing it as an
"optimisation" would be the single worst change available.

### Off-the-shelf reranker, not a fine-tuned one

The parent project's headline ablation was a **null result** (p = 0.806), and most of its
+0.048 came from *domain adaptation*, not negative selection. Adapting biomedical → ML
would likely repeat that gain, but it needs labels and arXiv ships none. That time went
into chunking and citation resolution instead.

### Full text, not abstracts only

Methods live in the body of a paper. "Explain" and "compare" over abstracts alone
compares marketing summaries. The cost is real — PDF download and parsing dominates
ingest time — so abstracts are indexed for *everything* and full text is capped, with
on-demand fetching for any paper you look up directly.

### Rejection is a mean, not a maximum

The obvious design — reject when the best passage scores below a threshold — **does not
work**:

| Query | max | mean of top-5 | in corpus? |
|---|---|---|---|
| explain JEPA | 0.998 | 0.995 | yes |
| how does JEPA differ from MAE | 0.898 | 0.671 | yes |
| what causes knee osteoarthritis | **0.670** | 0.190 | **no** |
| vitamin D and infection | 0.150 | 0.071 | no |
| sourdough hydration ratio | 0.001 | 0.001 | no |
| who won the 1998 football world cup | 0.348 | 0.190 | no |

The osteoarthritis query hits **0.670 on max** — enough to defeat any max-based threshold
— because the corpus genuinely contains a self-supervised *bone densitometry* paper. One
real partial match is all it takes. The mean separates cleanly, because a question that
belongs to this corpus matches *many* passages rather than one. Threshold: **mean of
top-5 ≥ 0.35**, which classified all eight probe queries correctly.

**Scale trap:** `CrossEncoder.predict` applies a sigmoid, so torch-backend scores are
probabilities in `[0,1]`. The ONNX backend returns **raw logits**. A threshold tuned on
one is meaningless on the other — the first version of this config used `-6.0`, which
could never fire against a sigmoid, so the rejection path was silently dead code until it
was measured.

### Per-paper capping

On SciFact one document was one abstract, so a top-5 was five distinct sources. Here a
paper yields dozens of chunks and the five best-scoring are routinely five consecutive
paragraphs of the same paper — correct by score, useless as an answer.
`answer.max_per_paper` forces multi-source results, which is what makes "compare"
queries work at all.

### Prompt injection

Retrieved passages are untrusted. Instructions are fixed in the system prompt before any
passage is seen; passages arrive in the user turn wrapped in `<passage>` tags with
tag-like sequences stripped so a passage cannot close its own wrapper and start issuing
orders. **No tools are passed to the API call**, so retrieved text has no tool-calling
path to reach.

---

## Frontend

`web/` — Next.js 16 (App Router), React 19, TypeScript, Tailwind v4, shadcn-compatible
layout with components in `src/components/ui/`.

The animated hero (`woven-light-hero.tsx`) was integrated with three fixes that matter
once it ships rather than demos:

1. **The animation loop allocated 150,000 `Vector3` objects per frame** (3 per particle ×
   50,000). That is a GC stall every few frames. Scratch values are hoisted and the maths
   runs on the raw `Float32Array` — identical output, zero allocation.
2. **The renderer, geometry and material were never disposed**, leaking a WebGL context
   on every unmount. Next.js navigation unmounts constantly.
3. **`prefers-reduced-motion` is respected** — the scene renders one still frame instead
   of animating.

Text is prop-driven rather than hardcoded, so the hero serves the product rather than the
other way round. The original demo lives at `/demo`.

Point the frontend elsewhere with `NEXT_PUBLIC_API_BASE` in `web/.env.local`.

---

## Configuration

Everything lives in `configs/ml.yaml`.

**Corpus breadth** — `ingest.queries` covers `cs.LG`, `cs.AI`, `cs.CL`, `cs.CV`, `cs.NE`,
`cs.RO`, `stat.ML`, sorted newest-first. Raise `max_results_per_query` for more papers;
raise `max_full_text` for more depth. Both cost ingest time.

**Refresh** — re-run `ingest_arxiv.py`. Version dedup means a paper that has moved from
`v2` to `v3` is *replaced*, not duplicated. Delete `cache/ml/` afterwards to force an
index rebuild.

---

## Deployment

The corpus and indices are built **offline** and mounted in. Nothing downloads papers at
request time except the citation-lookup path, which fetches a single PDF on demand.

### Backend

```bash
docker build -f Dockerfile.ml -t researchos-ml .
```

```bash
docker run --rm -p 8100:8100 -e GROQ_API_KEY=$GROQ_API_KEY -v "$PWD/data:/app/data:ro" -v "$PWD/cache/ml:/app/cache/ml" researchos-ml
```

CPU-only by design — a CUDA base triples the image for a workload that is already
network-bound on the Groq call. Reranking is ~10x slower on CPU; if you need the GPU,
run `scripts/serve_ml.py` on the host instead of containerising.

`data/` is mounted **read-only** and holds `arxiv_papers.jsonl` and
`arxiv_chunks.jsonl`. `cache/ml/` holds the BM25 and FAISS indices — leave it writable
so the first boot can build them (a few minutes over 40k chunks), or bake them into the
image to make startup instant. The `HEALTHCHECK` allows a 600s start period for exactly
that reason.

The model weights are baked at build time, so the first request is not a 500 MB download.

### Frontend

`output: "standalone"` is set, so `npm run build` emits a self-contained server:

```bash
cd web && npm run build && node .next/standalone/server.js
```

`NEXT_PUBLIC_API_BASE` is inlined **at build time**, not read at runtime — rebuild when
the backend URL changes. It is a public value; never put a secret in a `NEXT_PUBLIC_*`
variable, they ship to the browser.

### Access control

Set these on the backend before exposing it to anyone:

| Variable | Effect |
|---|---|
| `RESEARCHOS_API_TOKEN` | When set, every endpoint except `/health` requires `Authorization: Bearer <token>`. Verified: no token and wrong token both return **401**. |
| `RESEARCHOS_ALLOWED_ORIGINS` | Comma-separated CORS origins. Defaults to `localhost:3000` — set it to your real frontend origin or the browser blocks every call. |
| `RESEARCHOS_RATE_LIMIT` | Requests/minute per client IP, default 20. Verified: the 6th request in a 5/min window returns **429** with `Retry-After`. |

The frontend passes the token via `NEXT_PUBLIC_API_TOKEN`. That is **inlined into
the browser bundle** — it deters drive-by abuse of your Groq quota, it does not
protect anything. Anyone who loads the page can read it. Never put a real secret
in a `NEXT_PUBLIC_*` variable.

The rate limiter is in-process: it resets on restart and does not coordinate
across replicas. With one process serving a few people, that is what is wanted.

### Secrets

`GROQ_API_KEY` is read from the environment or from `.env` (gitignored). It is
never baked into an image or a bundle. On Windows write `.env` as **UTF-8** —
PowerShell's `>` redirection produces UTF-16, which the loader survives with a
warning but should not have to.

### Disk

The on-demand PDF cache is capped by `citation.max_pdf_cache_gb` (default 5 GB)
and evicts least-recently-used files. Extracted text is kept regardless — it is
~1% of the size and is what retrieval actually uses.

### Watching it fail

```bash
python scripts/review_failures.py --config configs/ml.yaml
```

Errors, downvotes, out-of-corpus queries (with their most common terms, as ingest
candidates), confident-but-never-clicked results, and the slowest requests. Run it
weekly once people are using it.

---

## Limitations

Honest bounding, same as the parent project.

- **No evaluation. No metrics. No results table.** This is a tool, not an experiment.
  Nothing here is a measured claim. The two-tier label scheme that would fix this —
  citation contexts for train/dev, hand-written questions for test — is specified in
  [REQUIREMENTS-ML.md §5](REQUIREMENTS-ML.md) and not built.
- **The rejection threshold is tuned on eight probe queries**, which is a sanity check,
  not a validation set. No false-accept/false-reject rate has been measured.
- **Generation is untested end to end.** The Groq path is written and typechecked but was
  never run against a live key during development — there was none available. The
  retrieval half is exercised thoroughly; the generation half is not.
- **Refresh is manual.** Re-running the script is the mechanism. No scheduler, and index
  rebuild is full rather than incremental.
- **Citation resolution is title-overlap, not a citation graph.** It matches on token
  overlap against titles. It will miss papers whose reference string is mangled, and it
  cannot resolve non-arXiv work (journal-only papers, books) at all.
- **PDF extraction is imperfect.** Two-column layouts sometimes interleave; papers whose
  headings do not match the expected pattern fall back to one unnamed section. Equations,
  tables and figures are dropped entirely.
- **Answer quality is unmeasured.** Retrieval is the component under scrutiny.
  Faithfulness and citation correctness are not evaluated.
- **Corpus is recent-biased.** `ingest.queries` sorts by submission date, so the corpus
  skews to the last few months. A foundational 2017 paper may not be indexed — though the
  citation-lookup path will still fetch it from arXiv on demand.
