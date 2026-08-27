# ResearchOS-ML

Search AI research papers, and de-reference the citations inside them.

You are reading a paper. You hit *"...following [23] (Assran et al., 2023)..."*. Paste
that reference in and get back: what the paper is, what it actually does, why it gets
cited, and an explanation in plain language — grounded in the paper's own text, with
every claim linked to a passage.

Built on the retrieval stack from the parent project, [ResearchOS](README.md).

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
