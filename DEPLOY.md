# Deploying ResearchOS

The front end goes on Vercel. **The backend cannot.**

It loads 40,379 passages and two models into memory, takes two to three minutes
to start, and keeps that state between requests. Vercel runs serverless
functions — no persistent memory, hard execution limits, no GPU. There is no
arrangement of this backend that fits, so it stays on a machine you control and
Vercel talks to it over the network.

---

## The two-part shape

```
  visitor's browser
        │  HTTPS
        ▼
  Vercel  ── your Next.js front end (static, no secrets)
        │  HTTPS
        ▼
  Hugging Face Space  ── FastAPI + BM25 + FAISS + reranker + your Groq key
```

Vercel serves the interface; the Space does the work. Neither depends on your
laptop being on.

---

## 1. Give the backend a public HTTPS address

Three options, in the order I would try them.

### Hugging Face Spaces — free, permanent, no card (recommended)

The only free tier that actually fits: 16 GB RAM, 2 vCPU, a persistent process,
Docker, and a permanent HTTPS URL. The models need ~2.1 GB, so there is room.

```bash
python scripts/prepare_deploy.py
```

That writes `deploy/` — the papers file slimmed from 86 MB to 16 MB by dropping
full text the chunks already hold, plus the chunks and the prebuilt indices.
About 182 MB total.

Then create a **Docker** Space at huggingface.co/new-space and push:

```
Dockerfile          <- Dockerfile.space, renamed
README.md           <- spaces/README.md (the YAML header is required)
src/  configs/  scripts/  requirements.txt
data/  cache/       <- from deploy/
```

`data/` and `cache/` are over 10 MB, so `git lfs track "data/*.jsonl" "cache/**"`
before committing them.

Set under **Settings → Variables and secrets**:

| Name | Kind |
|---|---|
| `GROQ_API_KEY` | secret |
| `RESEARCHOS_ALLOWED_ORIGINS` | variable — your Vercel origin |
| `RESEARCHOS_API_TOKEN` | secret, optional |

Your backend is then `https://<user>-<space>.hf.space`, permanently, whether or
not your laptop is on.

**The tradeoff is speed.** Two vCPU instead of a 4060 means the cross-encoder
takes seconds rather than milliseconds — `configs/ml.cpu.yaml` halves the
candidate set to compensate, and the Space uses it automatically. Expect a few
seconds per query. Free Spaces also sleep when idle; the weights are baked into
the image and the indices ship with the repo, so waking is seconds, not the
three minutes a cold rebuild would cost.

### Tunnel to your machine — fastest to set up, laptop must stay on

```bash
cloudflared tunnel --url http://127.0.0.1:8100
```

It prints a `https://<random>.trycloudflare.com` address. That is your
`NEXT_PUBLIC_API_BASE`.

The quick tunnel needs no account and changes its address each restart — fine
for showing people, annoying as a permanent home. A named tunnel on a free
Cloudflare account keeps a stable hostname.

**It only works while your machine is awake and the tunnel is running.** When
the laptop sleeps, the site shows "waiting for the backend" to everyone.

### Small server — permanent, you pay

`Dockerfile.ml` builds a CPU-only image. Any VPS with 4 GB RAM will run it.
Reranking is roughly ten times slower without the GPU — acceptable, not
pleasant. Mount `data/` (built offline by `scripts/ingest_arxiv.py`) read-only
and give `cache/ml/` somewhere writable for the indices.

---

## 2. Set a token before anyone else has the link

Your Groq key is on the backend, so deploying the front end never exposes it.
But an open backend is an open invitation to spend your free tier.

In `.env` on the backend machine — **written as UTF-8**, because PowerShell's
`>` produces UTF-16:

```
GROQ_API_KEY=gsk_...
RESEARCHOS_API_TOKEN=pick-something-long-and-random
RESEARCHOS_ALLOWED_ORIGINS=https://your-project.vercel.app
RESEARCHOS_RATE_LIMIT=20
```

`RESEARCHOS_ALLOWED_ORIGINS` must contain your exact Vercel domain. Miss it and
every request fails CORS in the browser while working fine from curl — which is
a confusing hour if you do not know to look for it.

Restart the backend after changing `.env`. A running process cannot see a
variable set after it started.

---

## 3. Deploy the front end

Point Vercel at this repository with **Root Directory** set to `web`.

Environment variables (Project → Settings → Environment Variables):

| Name | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE` | your HTTPS backend address, no trailing slash |
| `NEXT_PUBLIC_API_TOKEN` | the same token as `RESEARCHOS_API_TOKEN` |

Both are inlined into the browser bundle **at build time**. Changing either
requires a redeploy, not just a restart. Neither may hold a real secret — the
token is a deterrent against casual abuse, not protection.

---

## 4. Check it before sharing

```bash
python scripts/healthcheck.py --base https://your-backend.example --token YOUR_TOKEN
```

Thirteen checks against the live endpoints. If they pass from your machine
against the public address, they will pass from a visitor's browser too.

Then open the Vercel URL in a private window. The header should read
*"8,330 papers · 40,379 passages"*. If it says the deployment isn't finished,
`NEXT_PUBLIC_API_BASE` did not get set — that message exists specifically to
tell you so rather than showing a generic network error.

---

## What visitors get

Anyone with the link can search, explain citations, and upload a PDF. There are
no accounts; the rate limit is per IP and resets on backend restart. Search
history is per-browser, in `localStorage`, and never leaves their machine.

## What to expect to go wrong

**Everything works locally, nothing works on Vercel.** Almost always
`RESEARCHOS_ALLOWED_ORIGINS` missing the Vercel domain. The browser console says
CORS; `curl` says fine.

**It worked, then stopped.** The quick tunnel's address changes on restart, and
`NEXT_PUBLIC_API_BASE` was baked in at build time. Use a named tunnel, or
redeploy with the new address.

**Answers stop but search keeps working.** Groq's free tier allows 8,000 tokens
per minute across every user. Several people asking at once will queue. The
backend waits and retries rather than failing, so it reads as slow, not broken.
