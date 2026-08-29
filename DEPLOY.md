# Deploying ResearchOS

## Read this first, because the free tier moved

Hugging Face now bills **both** Docker and Gradio Spaces as compute. A free
personal account has exactly two options:

| | What you get | Cost |
|---|---|---|
| **Static Space** | Front-end files only. No Python, no backend. | Free for everyone |
| **Gradio Space on ZeroGPU** | Full Python process. Up to **2** of them. | Free, with conditions |

The conditions on the second are real: your account needs a **verified email**
and must be **older than 30 days**. If it is newer than that, no amount of
configuration will let you create the Space — skip to
[If your account does not qualify yet](#if-your-account-does-not-qualify-yet).

CPU Basic is greyed out for you because it is now a paid tier for Gradio Spaces.
That is not a mistake on your end.

---

## The shape this takes

Because a Gradio Space runs a real Python process, the Space is the whole
product — not just an API the front end talks to:

```
  visitor's browser
        │  HTTPS
        ▼
  Hugging Face Space  ─────────────────────────────
        /            the interface (Gradio)
        /papers /explain /upload /health   the API
        │
        └── BM25 + FAISS + reranker + your Groq key
```

Vercel becomes **optional**. Deploy it if you want the Next.js interface at your
own domain; it calls the same Space over the network. Skip it and the Space's
own interface is already a working, shareable product.

### About ZeroGPU

The Space asks for ZeroGPU hardware but **never claims a GPU**, deliberately.
BM25, a flat FAISS index over 40k passages and a 278M cross-encoder over 30
candidates are a few seconds of CPU; the writing happens on Groq's servers.
Taking a ZeroGPU slot would spend the *visitor's* daily quota — two minutes for
anyone not signed in — and start failing once it ran out, to save latency nobody
would notice. Staying on CPU costs no quota, never queues, and cannot run out.

Two consequences worth knowing: `spaces/requirements.txt` pins **torch 2.8.0**,
because ZeroGPU rejects the 2.6.0 the rest of the project uses; and the README
pins Python 3.12.12, one of the two versions ZeroGPU supports.

---

## 1. Build the Space repo

```bash
python scripts/prepare_deploy.py
```

That writes `deploy/` — about 182 MB, complete and pushable:

| | |
|---|---|
| `app.py` | Gradio UI mounted onto the FastAPI app |
| `README.md` | the YAML header is what tells Spaces this is a Gradio app |
| `requirements.txt` | serving dependencies only, not the training stack |
| `src/ configs/ scripts/` | the application |
| `data/` | papers slimmed from 86 MB to 16 MB, plus the chunks |
| `cache/ml/` | prebuilt BM25 and FAISS indices |

The indices ship with the repo on purpose. Building them takes about three
minutes, and free Spaces sleep when idle — without them, every wake would pay
that cost.

## 2. Create the Space

At [huggingface.co/new-space](https://huggingface.co/new-space):

| Field | Value |
|---|---|
| SDK | **Gradio** → Blank |
| Hardware | **ZeroGPU** (the only free option) |
| Storage bucket | None — the repo carries its own data |
| Visibility | Public |
| License | AGPL-3.0 |

## 3. Push

```bash
cd deploy
git init -b main
git lfs install
git add -A
git commit -m "ResearchOS"
git remote add origin https://huggingface.co/spaces/USER/SPACE
git push -u origin main --force
```

`--force` is needed and is safe here. Creating a Space through the web form
initialises the repo with its own `README.md` and `.gitattributes`, so a fresh
local history has no common ancestor and a plain push is rejected as
non-fast-forward. You are overwriting two placeholder files on a Space that has
never run.

`prepare_deploy.py` writes the `.gitattributes` that puts `data/` and
`cache/ml/` through LFS. Spaces rejects files over 10 MB otherwise, and both are
far over.

The first build takes several minutes: it installs torch and downloads the model
weights. Watch the **Logs** tab. It is ready when the log reads
`Application startup complete`.

## 4. Set the secrets

**Settings → Variables and secrets:**

| Name | Kind | Value |
|---|---|---|
| `GROQ_API_KEY` | secret | your Groq key |
| `RESEARCHOS_ALLOWED_ORIGINS` | variable | your Vercel origin — only if you deploy the front end |
| `RESEARCHOS_RATE_LIMIT` | variable | requests per minute per visitor, default 20 |
| `RESEARCHOS_API_TOKEN` | secret | optional; when set the REST API needs it, the Gradio UI stays open |

Without `GROQ_API_KEY` the Space still runs — search and retrieval work, you get
ranked passages instead of written answers.

Setting a variable restarts the Space. A running process cannot see a variable
set after it started.

## 5. Check it

```bash
python scripts/healthcheck.py --base https://USER-SPACE.hf.space
```

Thirteen checks against the live endpoints. All thirteen pass against this build
locally; if they pass against the public address they will pass from a
visitor's browser.

Then open `https://USER-SPACE.hf.space` and search for something. The header
should read *"8,330 papers · 40,379 passages"*.

---

## 6. The front end, if you want it

Point Vercel at this repository with **Root Directory** set to `web`.

| Name | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE` | `https://USER-SPACE.hf.space`, no trailing slash |
| `NEXT_PUBLIC_API_TOKEN` | the same value as `RESEARCHOS_API_TOKEN`, if you set one |

Both are inlined into the browser bundle **at build time**. Changing either
needs a redeploy, not a restart. Neither may hold a real secret — the token
deters casual abuse, it does not protect anything.

Then add that Vercel origin to `RESEARCHOS_ALLOWED_ORIGINS` on the Space, or
every request will fail CORS in the browser while working fine from curl.

---

## If your account does not qualify yet

ZeroGPU needs a verified email and an account older than 30 days. Until then:

**Tunnel to your machine.** Free, works today, laptop must stay awake:

```bash
cloudflared tunnel --url http://127.0.0.1:8100
```

It prints a `https://<random>.trycloudflare.com` address — that is your
`NEXT_PUBLIC_API_BASE`. The address changes on restart; a named tunnel on a free
Cloudflare account keeps it stable. When the laptop sleeps, the site is down for
everyone.

**A small server.** `Dockerfile.space` builds a CPU image that runs on any VPS
with 4 GB of RAM. Permanent, and you pay for it.

---

## Adding the licence text

The README declares AGPL-3.0 and the app footer links back to the Space repo,
which is what AGPL asks for when software is used over a network. To ship the
full text too:

```bash
curl -o deploy/LICENSE https://www.gnu.org/licenses/agpl-3.0.txt
```

PyMuPDF is the reason for AGPL rather than something permissive — it is
AGPL-licensed unless you buy a commercial licence from Artifex.

---

## What was tested here, and what could not be

The whole of `deploy/` was run locally exactly as the Space runs it — `app.py`
from the `deploy/` directory, with the key supplied as an environment variable —
and all thirteen healthcheck checks pass against it, plus the Gradio interface
driven by hand through both tabs.

Two things can only be confirmed on the Space itself, because they are
properties of that environment:

**Python 3.12.12.** This machine has 3.11, so the build ran there. Every pinned
package publishes 3.12 wheels, so this should be uneventful. If a wheel is
missing, drop `python_version` from the README header and let Spaces choose.

**Running on ZeroGPU without claiming a GPU.** Nothing in Hugging Face's
documentation requires a `@spaces.GPU` function to exist, and this app has none
by design. If the Space refuses to start for that reason, the fix is three lines
in `app.py` — `import spaces`, a trivial `@spaces.GPU(duration=5)` function, and
adding `spaces` to `requirements.txt`. The retrieval path still would not need
to call it.

## What to expect to go wrong

**The build fails installing a package.** `spaces/requirements.txt` is
deliberately not the repo's `requirements.txt`: the training stack includes
`pytrec-eval-terrier`, which compiles C at install time and has no business in a
serving build. If you swapped the files, swap back.

**"No such file or directory: data/arxiv_chunks.jsonl".** LFS did not push. Run
`git lfs install` *before* the first commit, or the files are committed as
pointer text. `git lfs ls-files` in `deploy/` should list them.

**The Space builds, then dies on startup.** Check the Logs tab for the real
error. If it complains about torch and ZeroGPU, the pin in `requirements.txt`
drifted from the versions ZeroGPU supports (2.8.0 and above).

**Everything works on the Space, nothing works on Vercel.** Almost always
`RESEARCHOS_ALLOWED_ORIGINS` missing the exact Vercel domain. The browser
console says CORS; `curl` says fine.

**Answers stop but search keeps working.** Groq's free tier allows 8,000 tokens
per minute across every user of your key. Several people asking at once queue
up. The backend waits and retries rather than failing, so it reads as slow
rather than broken.

**The first request after a quiet spell is slow.** Free Spaces sleep. The
indices ship with the repo so waking is seconds, not the three minutes a rebuild
would cost.
