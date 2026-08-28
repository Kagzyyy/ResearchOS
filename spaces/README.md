---
title: ResearchOS
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Search AI papers and de-reference the citations inside them
---

# ResearchOS

Backend for [ResearchOS](https://github.com/) — hybrid retrieval over arXiv with
a cross-encoder reranker and grounded generation.

This Space serves the API only. The interface lives on Vercel and calls it.

## Endpoints

| | |
|---|---|
| `GET /health` | corpus size, models, whether generation is on |
| `POST /papers` | a topic in, ranked papers out |
| `POST /explain` | a citation or concept, explained |
| `POST /upload` | attach a PDF, then ask questions about it |
| `GET /docs` | interactive API docs |

## Configuration

Set these under **Settings → Variables and secrets**:

| Name | Kind | Purpose |
|---|---|---|
| `GROQ_API_KEY` | secret | Written answers. Without it retrieval still works and you get ranked passages. |
| `RESEARCHOS_ALLOWED_ORIGINS` | variable | Your front-end origin, e.g. `https://your-app.vercel.app` |
| `RESEARCHOS_API_TOKEN` | secret | Optional shared token; when set, every endpoint except `/health` requires it |
| `RESEARCHOS_RATE_LIMIT` | variable | Requests per minute per IP, default 20 |

## Notes

Running on 2 vCPU, so the cross-encoder is the bottleneck — expect a few seconds
per query rather than the sub-second a GPU gives. `configs/ml.cpu.yaml` halves
the candidate set to compensate.

Free Spaces sleep when idle. The first request after a sleep takes longer while
the container wakes; model weights are baked into the image and the search
indices ship with the repo, so it is seconds rather than minutes.
