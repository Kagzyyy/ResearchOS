---
title: ResearchOS
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
python_version: "3.12.12"
pinned: false
license: agpl-3.0
short_description: Search AI papers and de-reference the citations inside them
---

# ResearchOS

Search AI research by what a paper *does*, and de-reference the citations
inside it. Hybrid retrieval — BM25 and dense embeddings fused with reciprocal
rank fusion — over 8,330 arXiv papers, reranked by a cross-encoder, with
answers written by Groq and grounded in the passages that were actually
retrieved.

This Space is both halves at once: the interface at `/`, and the API the
browser extension and the Next.js front end call.

## Endpoints

| | |
|---|---|
| `GET /health` | corpus size, models, whether generation is on |
| `POST /papers` | a topic in, ranked papers out |
| `POST /ask` | a question, answered from the corpus |
| `POST /explain` | a citation or concept, explained |
| `POST /upload` | attach a PDF, then ask questions about it |
| `GET /docs` | interactive API docs |

## Configuration

Under **Settings → Variables and secrets**:

| Name | Kind | Purpose |
|---|---|---|
| `GROQ_API_KEY` | secret | Written answers. Without it retrieval still works and you get ranked passages instead of prose. |
| `RESEARCHOS_ALLOWED_ORIGINS` | variable | Your front-end origin, e.g. `https://your-app.vercel.app`. Missing it is the usual cause of "works in curl, fails in the browser". |
| `RESEARCHOS_API_TOKEN` | secret | Optional. When set, every REST endpoint except `/health` requires it. The Gradio interface stays open either way — it is the public front door. |
| `RESEARCHOS_RATE_LIMIT` | variable | Requests per minute per visitor, default 20. Both doors share one budget. |

## Notes on speed

Nothing here uses the GPU, deliberately. Retrieval and reranking are a few
seconds of CPU, and the writing happens on Groq's servers. Claiming a ZeroGPU
slot would spend a visitor's daily quota — two minutes for anyone not signed in
— and fail once it ran out, to save time nobody would notice. `configs/ml.cpu.yaml`
halves the candidate set so the cross-encoder keeps up on 2 vCPU.

Free Spaces sleep when idle. The search indices ship with the repo rather than
being rebuilt on boot, so waking costs seconds; only the first boot pays for
downloading the model weights.

## What it does when it does not know

The corpus is a sample of arXiv, not a mirror. When nothing in it scores well
enough, the query goes to arXiv live and the results come back marked as such.
When even that is weak, it says so rather than returning five unrelated papers.
