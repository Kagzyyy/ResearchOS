"""ResearchOS on Hugging Face Spaces — Gradio UI and REST API in one process.

Spaces entrypoint. In the deployed repo this file sits at the root next to
`src/`, `configs/`, `data/` and `cache/`; `scripts/prepare_deploy.py` puts it
there.

Why this shape
--------------
Gradio must own the server here, and the API is attached to it after launch.

That is forced, not chosen. ZeroGPU's runtime holds port 7860 with a placeholder
server and only releases it inside `demo.launch()` — running our own uvicorn
dies with "address already in use" after the whole pipeline has loaded, which is
exactly what happened on the first deployment. And the attachment must come
*after* launch: `launch()` builds a fresh FastAPI app, so anything added to
`demo.app` beforehand is silently discarded (verified: routes added pre-launch
404).

The FastAPI app keeps its identity by being mounted whole onto Gradio's server.
A mounted ASGI app runs its own middleware stack — Starlette builds it lazily on
first request — so CORS for the Vercel front end survives. Its lifespan does not
run when mounted, so the state it would have built is built directly via
`build_state()` first. The mount is appended after Gradio's routes, and route
matching is first-match, so the UI wins at `/` and everything else — `/papers`,
`/explain`, `/upload`, `/health` — falls through to the API.

Why there is no GPU code
------------------------
ZeroGPU is the only hardware a free personal account can run a Gradio Space on,
but this workload does not want it. BM25, a flat FAISS index over 40k passages
and a 278M cross-encoder over 30 candidates are a few seconds of CPU; the
writing is done by Groq, over the network. Taking the GPU would mean draining a
visitor's daily ZeroGPU quota — two minutes for anyone not signed in — and
failing once it ran out, in exchange for latency nobody would notice. Staying on
CPU costs no quota, never queues, and cannot run out.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

# Must be set before src.api.ml_app is imported: it reads the config path at
# module level. ml.cpu.yaml halves the candidate set for 2 vCPU and binds 7860.
os.environ.setdefault("RESEARCHOS_ML_CONFIG", "configs/ml.cpu.yaml")
# ZeroGPU patches torch.cuda.is_available() to True (CUDA emulation), which
# tricks device autodetection into putting models on a GPU this app never uses.
# Force CPU - see the module docstring for why CPU is the right call here.
os.environ["RESEARCHOS_DEVICE"] = "cpu"

# ZeroGPU refuses to serve an app with no @spaces.GPU function at all
# ("No @spaces.GPU function detected during startup" -> RUNTIME_ERROR), so one
# must exist even though nothing ever calls it. Imported before torch, as the
# spaces package requires. Locally the package is absent and this is a no-op.
try:
    import spaces  # noqa: E402

    @spaces.GPU(duration=10)
    def _zerogpu_requirement() -> str:
        """Never called. Exists because ZeroGPU requires one decorated function."""
        return "This Space runs on CPU by design; see the module docstring."
except ImportError:
    pass

# In the deployed Space this file sits beside `src/`; in this repo it sits in
# `spaces/`, one level below it. Find the root either way so the same file runs
# locally and on the Space without editing.
_here = Path(__file__).resolve().parent
_root = next((p for p in (_here, _here.parent) if (p / "src").is_dir()), _here)
sys.path.insert(0, str(_root))
os.chdir(_root)  # every path in the config is relative to the project root

import gradio as gr  # noqa: E402
from starlette.routing import Mount  # noqa: E402

from src.api.llm import is_configured  # noqa: E402
from src.api.ml_app import app as api  # noqa: E402  (carries routes + CORS)
from src.api.ml_app import build_state, state  # noqa: E402
from src.api.security import _limiter  # noqa: E402
from src.api.uploads import UploadError  # noqa: E402

MAX_QUESTION = 1000

# AGPL-3.0 requires that people using this over a network can reach the source.
# The Space repo is public and is the source, so link to it; SPACE_ID is set by
# the Spaces runtime and absent when running locally.
_space = os.environ.get("SPACE_ID", "").strip()
SOURCE_URL = (
    f"https://huggingface.co/spaces/{_space}/tree/main" if _space
    else "https://huggingface.co/spaces"
)


# --------------------------------------------------------------------------
# The Gradio handlers call the pipeline directly rather than looping back
# through HTTP, so they skip the `guard` dependency and with it the rate limit.
# Sharing the limiter instance keeps one budget across both doors.
# --------------------------------------------------------------------------
def _throttle(request: gr.Request | None) -> str | None:
    """None if the caller may proceed, otherwise a message to show them."""
    who = "unknown"
    if request is not None:
        forwarded = dict(request.headers).get("x-forwarded-for", "")
        who = forwarded.split(",")[0].strip() if forwarded else (request.client.host or "unknown")
    allowed, retry_after = _limiter.check(who)
    if allowed:
        return None
    return (
        f"### Slow down a moment\n\nThis Space allows {_limiter.per_minute} requests a minute "
        f"per visitor, so one person cannot drain the shared answer budget. "
        f"Try again in {retry_after} seconds."
    )


def _ready() -> bool:
    return bool(state.get("pipeline"))


def _authors(names: list[str]) -> str:
    if not names:
        return ""
    shown = ", ".join(names[:3])
    return f"{shown} et al." if len(names) > 3 else shown


# Every spelling of maths a model or an arXiv abstract uses; Gradio only typesets
# the ones listed here.
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "$", "right": "$", "display": False},
]
# gr.Markdown with the delimiters pre-set, so every call site stays as it was.
_Markdown = functools.partial(gr.Markdown, latex_delimiters=LATEX_DELIMITERS)


def _render_diagnosis(d: dict[str, Any] | None) -> str:
    """v2: why the search struggled and what to try - CRITIC-R1 format."""
    if not d:
        return ""
    blamed = ("the query itself" if d.get("location") == "search"
              else f"“{d['location_title']}”" if d.get("location_title")
              else d.get("location", ""))
    out = ["### Why this search struggled",
           f"**Weakest evidence:** {blamed}", "", d.get("reason", "")]
    if d.get("suggested_query"):
        out += ["", f"**Try searching:** `{d['suggested_query']}`"]
    out += ["", "<sub>Structured diagnosis in the CRITIC-R1 format</sub>", "", "---", ""]
    return "\n".join(out)


def _render_papers(result: dict[str, Any]) -> str:
    papers = result.get("papers") or []
    if not papers:
        return (
            "### Nothing convincing\n\nNothing in the corpus or on arXiv scored well enough "
            "to be worth showing. That is the honest answer rather than five unrelated "
            "papers — try naming the method or the task more specifically."
        )

    lines: list[str] = [diag] if diag else []
    if result.get("source") == "arxiv_live":
        lines.append(
            "> Not in the indexed corpus, so these came from a live search of arXiv and "
            "OpenAlex just now, rescored by the same reranker.\n"
        )
    elif not result.get("confident"):
        lines.append("> Weak matches — treat these as leads rather than answers.\n")

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "").strip()
        url = paper.get("arxiv_url") or ""
        head = f"**{i}. [{title}]({url})**" if url else f"**{i}. {title}**"
        lines.append(head)

        meta = [m for m in (
            _authors(paper.get("authors") or []),
            (paper.get("published") or "")[:10],
            (paper.get("venue") if paper.get("venue") not in (None, "", "arXiv")
             else paper.get("primary_category")) or "",
            f"{paper['cited_by_count']:,} citations" if paper.get("cited_by_count") else "",
            f"live · {paper['live_source']}" if paper.get("live_source") else "",
            f"score {paper.get('score')}",
        ) if m]
        lines.append(" · ".join(meta))

        why = (paper.get("why") or "").strip().replace("\n", " ")
        if why:
            section = paper.get("why_section") or "text"
            lines.append(f"\n> *Matched in {section}:* {why}…")
        lines.append("")

    timing = result.get("timing_ms", {}).get("total")
    if timing:
        lines.append(f"\n<sub>{len(papers)} papers in {timing/1000:.1f}s</sub>")
    return "\n".join(lines)


def _render_explanation(result: dict[str, Any]) -> str:
    diag = _render_diagnosis(result.get("diagnosis"))
    lines: list[str] = [diag] if diag else []
    paper = result.get("paper")
    if paper and paper.get("title"):
        url = paper.get("arxiv_url") or ""
        title = paper["title"]
        lines.append(f"### {f'[{title}]({url})' if url else title}")
        meta = [m for m in (
            _authors(paper.get("authors") or []),
            (paper.get("published") or "")[:10],
            paper.get("primary_category") or "",
        ) if m]
        if meta:
            lines.append(" · ".join(meta) + "\n")

    if result.get("source") == "general_knowledge":
        lines.append("<sub>general knowledge &middot; no citations</sub>\n")
    explanation = (result.get("explanation") or "").strip()
    lines.append(explanation or "_No explanation was produced._")

    passages = result.get("passages") or []
    if passages:
        grounded = result.get("source") != "general_knowledge"
        lines.append("\n---\n**" + ("Grounded in these passages" if grounded
                                     else "Related papers")
                     + "**\n")
        for p in passages[:6]:
            title = (p.get("title") or "").strip()
            pid = p.get("paper_id") or ""
            url = f"https://arxiv.org/abs/{pid}" if pid and "." in str(pid) else ""
            label = f"[{title}]({url})" if url else title
            snippet = (p.get("text") or "").strip().replace("\n", " ")[:220]
            lines.append(f"- {label} — {snippet}…")

    related = result.get("related") or []
    if related:
        lines.append("\n**Papers that build on it**\n")
        for r in related[:4]:
            lines.append(f"- {r.get('title', '')}")

    timing = result.get("timing_ms", {}).get("total")
    if timing:
        lines.append(f"\n<sub>answered in {timing/1000:.1f}s</sub>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------
def find_papers(topic: str, top_k: int, request: gr.Request = None) -> str:
    topic = (topic or "").strip()
    if len(topic) < 3:
        return "Type a topic — a method, a task, a problem."
    if not _ready():
        return "The Space is still loading its index. Give it a few seconds and try again."
    blocked = _throttle(request)
    if blocked:
        return blocked
    try:
        result = state["pipeline"].find_papers(topic[:MAX_QUESTION], top_k=int(top_k))
    except Exception as exc:  # noqa: BLE001 — a stack trace in the UI helps nobody
        return f"That search failed: `{type(exc).__name__}: {exc}`"
    return _render_papers(result)


def attach_pdf(path: str | None) -> tuple[str | None, str]:
    """Parse an uploaded PDF once; later questions reuse the document id."""
    if not path:
        return None, ""
    if not _ready():
        return None, "The Space is still loading. Try the upload again in a moment."
    try:
        data = Path(path).read_bytes()
        meta = state["uploads"].add(data, Path(path).name)
    except UploadError as exc:
        return None, f"That file was rejected: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read that PDF: `{type(exc).__name__}: {exc}`"
    return meta.document_id, (
        f"Attached **{meta.title}** — {meta.n_chunks} passages. "
        "Questions below are now answered from this paper."
    )


def explain(question: str, document_id: str | None, request: gr.Request = None) -> str:
    question = (question or "").strip()
    if len(question) < 3:
        return "Paste a citation, an arXiv id, or ask a question."
    if not _ready():
        return "The Space is still loading its index. Give it a few seconds and try again."
    blocked = _throttle(request)
    if blocked:
        return blocked

    pipeline = state["pipeline"]
    try:
        if document_id:
            store = state["uploads"]
            meta = store.get_meta(document_id)
            if meta is None:
                return "That upload has expired. Attach the PDF again."
            answered = pipeline.answer_from_document(
                question[:MAX_QUESTION], store.passages(document_id), meta.title, generate=True
            )
            return _render_explanation(
                {"explanation": answered["answer"], "passages": answered["citations"],
                 "timing_ms": answered["timing_ms"],
                 "paper": {"title": meta.title, "authors": [], "published": "",
                           "primary_category": "your upload", "arxiv_url": ""}}
            )
        result = pipeline.explain_citation(question[:MAX_QUESTION], generate=True)
    except Exception as exc:  # noqa: BLE001
        return f"That request failed: `{type(exc).__name__}: {exc}`"
    return _render_explanation(result)


def corpus_line() -> str:
    if not _ready():
        return "Loading the index…"
    pipeline = state["pipeline"]
    generation = (
        state.get("cfg", {}).get("answer", {}).get("model", "Groq") if is_configured()
        else "retrieval only — no GROQ_API_KEY set, so you get passages, not prose"
    )
    return (
        f"**{len(pipeline.papers_by_id):,} papers · {len(pipeline.corpus):,} passages** "
        f"· answers by {generation}"
    )


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
with gr.Blocks(title="ResearchOS", analytics_enabled=False) as demo:
    _Markdown(
        "# ResearchOS\n"
        "Search AI research by what a paper *does*, and de-reference the citations inside it."
    )
    status = _Markdown("Loading the index…")

    with gr.Tab("Find papers"):
        _Markdown("Describe the work you are looking for — a method, a task, a problem.")
        with gr.Row():
            topic = gr.Textbox(
                label="Topic", scale=4, max_lines=2,
                placeholder="joint-embedding predictive architectures for video",
            )
            count = gr.Slider(3, 20, value=8, step=1, label="Results", scale=1)
        find_button = gr.Button("Search", variant="primary")
        papers_out = _Markdown()
        gr.Examples(
            [["world models for robot manipulation", 8],
             ["mixture of experts routing", 8],
             ["why do transformers need positional encodings", 8]],
            inputs=[topic, count],
        )
        find_button.click(find_papers, [topic, count], papers_out)
        topic.submit(find_papers, [topic, count], papers_out)

    with gr.Tab("Explain"):
        _Markdown(
            "Paste a reference you hit while reading — `arXiv:2312.00752`, or "
            "`[23] M. Assran et al. Self-Supervised Learning…` — and get the paper it "
            "points to, explained. Plain questions work too. Attach a PDF to ask about "
            "that paper instead of the corpus."
        )
        document_id = gr.State(None)
        pdf = gr.File(label="Optional: a paper to ask about", file_types=[".pdf"],
                      type="filepath")
        pdf_status = _Markdown()
        citation = gr.Textbox(
            label="Citation, arXiv id, or question", max_lines=4,
            placeholder="How do JEPAs avoid representation collapse?",
        )
        explain_button = gr.Button("Explain", variant="primary")
        explain_out = _Markdown()

        pdf.change(attach_pdf, pdf, [document_id, pdf_status])
        pdf.clear(lambda: (None, ""), None, [document_id, pdf_status])
        explain_button.click(explain, [citation, document_id], explain_out)
        citation.submit(explain, [citation, document_id], explain_out)

    _Markdown(
        "<sub>Answers are grounded in retrieved passages and cite them. A weak match is "
        "reported as a weak match rather than dressed up. "
        f"[API docs](/docs) · [source]({SOURCE_URL}) · AGPL-3.0</sub>"
    )

    demo.load(corpus_line, None, status)


if __name__ == "__main__":
    # Launch first: on ZeroGPU this is the handshake that takes over port 7860
    # from the platform's placeholder server. The UI is up immediately; its
    # handlers answer "still loading" until build_state() finishes below.
    # launch() reads GRADIO_SERVER_NAME/GRADIO_SERVER_PORT from the environment,
    # which Spaces sets. Gradio 6 takes `theme` here, not on the constructor.
    demo.launch(
        prevent_thread_lock=True,
        theme=gr.themes.Soft(),
        max_file_size="25mb",
        quiet=False,
        # On Spaces, Gradio 6 defaults to SSR: a Node server owns the public
        # port and proxies only Gradio's own endpoints to Python - every other
        # path gets the SPA page, which silently swallowed /health, /papers and
        # the rest of the REST API. Serving straight from Python restores them.
        ssr_mode=False,
    )

    build_state()

    # Attach the whole FastAPI app to the live server. Appended last, so
    # Gradio's own routes keep winning where they exist and the rest falls
    # through to the API — and a mounted app builds its own middleware stack
    # (CORS included) lazily on first request.
    demo.app.router.routes.append(Mount("/", app=api))
    print(f"[space] API attached | UI and REST on one server", flush=True)

    demo.block_thread()
