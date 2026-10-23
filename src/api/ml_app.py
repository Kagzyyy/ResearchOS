"""FastAPI backend for ResearchOS-ML.

    RESEARCHOS_ML_CONFIG=configs/ml.yaml uvicorn src.api.ml_app:app --port 8100

Endpoints:
    GET  /health            corpus size, model ids, whether generation is enabled
    POST /papers            topic in, ranked papers out
    POST /ask               question answering over the corpus
    POST /explain           resolve a citation and explain the paper it points to
    POST /search            retrieval only, no generation
    POST /feedback/click    log which citation a user opened
    GET  /stats             feedback counts

CORS is open to localhost origins so the Next.js dev server can call it.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api import persist
from src.api.db import FeedbackLog
from src.api.llm import is_configured
from src.api.ml_pipeline import MLPipeline
from src.api.security import allowed_origins, api_token, guard, rate_limit_per_minute
from src.api.uploads import UploadError, UploadStore
from src.config import load_config
from src.data.arxiv_source import breaker_open as arxiv_breaker_open
from src.data.openalex_source import status as openalex_status

CONFIG_PATH = os.environ.get("RESEARCHOS_ML_CONFIG", "configs/ml.yaml")

state: dict[str, Any] = {}


def build_state() -> None:
    """Load the pipeline and its supporting stores into `state`.

    A plain function rather than lifespan-only code because the Hugging Face
    Space cannot run this app's lifespan: on ZeroGPU the platform holds the
    port until Gradio's own launch() is called, so there the API is attached to
    Gradio's already-running server and this is called directly instead.
    """
    cfg = load_config(CONFIG_PATH)
    state["cfg"] = cfg
    state["pipeline"] = MLPipeline(cfg)
    persist.restore(cfg["paths"]["feedback_db"])      # before the DB is opened
    state["log"] = FeedbackLog(cfg["paths"]["feedback_db"])
    persist.start_sync(cfg["paths"]["feedback_db"])
    state["uploads"] = UploadStore(cfg["paths"].get("upload_dir", "cache/uploads"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    build_state()
    yield
    state.clear()


app = FastAPI(
    title="ResearchOS-ML",
    description="Search, explain and de-reference AI research papers.",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(guard)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_probe(http: Request) -> bool:
    """Monitoring traffic (scripts/healthcheck.py) is logged but kept out of
    the v2 metrics - otherwise every health check counts as a user retry."""
    return http.headers.get("X-ResearchOS-Probe") == "1"


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)
    generate: bool = True
    # Set when this query came from a v2 diagnosis's "Try: ..." button: the id
    # of the query whose diagnosis suggested it. Drives the live v2 funnel.
    retry_of: int | None = Field(default=None, ge=1)


class PapersRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=30)
    retry_of: int | None = Field(default=None, ge=1)


class ExplainRequest(BaseModel):
    citation: str = Field(min_length=3, max_length=2000)
    generate: bool = True
    # When set, the question is answered against this uploaded PDF instead of
    # the corpus.
    document_id: str | None = Field(default=None, max_length=64)
    retry_of: int | None = Field(default=None, ge=1)


class ClickRequest(BaseModel):
    query_id: int
    doc_id: str


class VoteRequest(BaseModel):
    query_id: int
    helpful: bool
    comment: str | None = Field(default=None, max_length=2000)


@app.exception_handler(Exception)
async def log_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Record every unhandled failure. Otherwise they are invisible."""
    log: FeedbackLog | None = state.get("log")
    if log is not None:
        try:
            log.log_error(endpoint=request.url.path, error=exc)
        except Exception:  # noqa: BLE001 - logging must never mask the original
            pass
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline: MLPipeline | None = state.get("pipeline")
    cfg = state.get("cfg", {})
    return {
        "status": "ok" if pipeline else "loading",
        "chunks": len(pipeline.corpus) if pipeline else 0,
        "papers": len(pipeline.papers_by_id) if pipeline else 0,
        "reranker_model": cfg.get("reranker", {}).get("model"),
        "llm_model": cfg.get("answer", {}).get("model"),
        "generation_enabled": is_configured(),
        "auth_required": api_token() is not None,
        "rate_limit_per_minute": rate_limit_per_minute(),
        # Live sources are optional enrichments; this says which are usable
        # right now and how much of OpenAlex's daily budget is left.
        "live_sources": {
            "arxiv": {"available": not arxiv_breaker_open()},
            "openalex": openalex_status(),
        },
    }


@app.post("/search")
def search(request: AskRequest) -> dict[str, Any]:
    """Retrieval and reranking only."""
    pipeline: MLPipeline = state["pipeline"]
    return {"query": request.query, "results": pipeline.retrieve(request.query, request.top_k)}


@app.post("/papers")
def papers(request: PapersRequest, http: Request) -> dict[str, Any]:
    """Topic in, ranked papers out (not passages)."""
    pipeline: MLPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]
    result = pipeline.find_papers(request.topic, top_k=request.top_k)
    query_id = log.log_query(
        query=request.topic,
        answer=None,
        latency_ms=result["timing_ms"]["total"],
        backend="retrieval",
        citations=[
            {"doc_id": p["doc_id"], "score": p["score"], "title": p["title"]}
            for p in result["papers"]
        ],
        mode="papers",
        confident=result["confident"],
        top_score=result["top_score"],
        mean_top_score=result["mean_top_score"],
        generated=False,
        source=result.get("source"),
        diagnosis=result.get("diagnosis"),
        retry_of=request.retry_of,
        probe=is_probe(http),
    )
    return {"query_id": query_id, **result}


@app.post("/ask")
def ask(request: AskRequest, http: Request) -> dict[str, Any]:
    """Question answering, logged."""
    pipeline: MLPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]
    result = pipeline.answer(request.query, generate=request.generate)
    query_id = log.log_query(
        query=request.query,
        answer=result["answer"],
        latency_ms=result["timing_ms"]["total"],
        backend="groq" if result["generation_configured"] else "retrieval-only",
        citations=result["citations"],
        mode="ask",
        confident=result["confident"],
        top_score=result["top_score"],
        mean_top_score=result["mean_top_score"],
        generated=bool(result["answer"]) and result["generation_configured"],
        source=result.get("source"),
        diagnosis=result.get("diagnosis"),
        retry_of=request.retry_of,
        probe=is_probe(http),
    )
    return {"query_id": query_id, **result}


@app.post("/explain")
def explain(request: ExplainRequest, http: Request) -> dict[str, Any]:
    """Resolve a citation to its paper and explain it in plain terms."""
    pipeline: MLPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]
    if request.document_id:
        store: UploadStore = state["uploads"]
        meta = store.get_meta(request.document_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="That document is no longer stored.")
        answer = pipeline.answer_from_document(
            request.citation, store.passages(request.document_id), meta.title,
            generate=request.generate,
        )
        query_id = log.log_query(
            query=request.citation, answer=answer["answer"],
            latency_ms=answer["timing_ms"]["total"],
            backend="groq" if answer["generation_configured"] else "retrieval-only",
            citations=answer["citations"], mode="upload",
            confident=True, top_score=answer["top_score"],
            mean_top_score=answer["mean_top_score"],
            generated=bool(answer["answer"]) and answer["generation_configured"],
        )
        return {
            "query_id": query_id, "citation": request.citation, "resolved": True,
            "method": "uploaded_document", "query_used": request.citation,
            "candidates": [], "from_corpus": False,
            "paper": {
                "paper_id": meta.document_id, "title": meta.title, "abstract": "",
                "authors": [], "published": "", "primary_category": "your upload",
                "arxiv_url": "", "has_full_text": True,
            },
            "explanation": answer["answer"], "passages": answer["citations"],
            "related": [], "source": "upload",
            "generation_configured": answer["generation_configured"],
            "timing_ms": answer["timing_ms"],
        }

    result = pipeline.explain_citation(request.citation, generate=request.generate)
    query_id = log.log_query(
        query=request.citation,
        answer=result.get("explanation"),
        latency_ms=result["timing_ms"]["total"],
        backend="groq" if result["generation_configured"] else "retrieval-only",
        citations=result.get("passages", []),
        mode="explain",
        confident=result["resolved"],
        generated=bool(result.get("explanation")) and result["generation_configured"],
        source=result.get("source"),
        diagnosis=result.get("diagnosis"),
        retry_of=request.retry_of,
        probe=is_probe(http),
    )
    return {"query_id": query_id, **result}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a PDF, then ask questions about it via /explain?document_id=..."""
    store: UploadStore = state["uploads"]
    try:
        data = await file.read()
        meta = store.add(data, file.filename or "upload.pdf")
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return meta.to_public()


@app.post("/feedback/click")
def feedback_click(request: ClickRequest) -> dict[str, Any]:
    log: FeedbackLog = state["log"]
    try:
        log.log_click(request.query_id, request.doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"logged": True, **log.stats()}


@app.post("/feedback/vote")
def feedback_vote(request: VoteRequest) -> dict[str, Any]:
    """Explicit thumbs up/down - the scarcest and most useful signal."""
    log: FeedbackLog = state["log"]
    try:
        log.log_vote(request.query_id, request.helpful, request.comment)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"logged": True, **log.stats()}


@app.get("/failures")
def failures(limit: int = 30) -> dict[str, Any]:
    """Everything worth reading when asking 'how is this failing?'."""
    return state["log"].recent_failures(limit=min(limit, 200))


@app.get("/stats")
def stats() -> dict[str, int]:
    return state["log"].stats()


@app.get("/v2/metrics")
def v2_metrics() -> dict[str, Any]:
    """Live v2 funnel: diagnoses shown, retries taken, retries that led to a click."""
    return state["log"].v2_metrics()


@app.get("/v2/labels")
def v2_labels(limit: int = 200) -> dict[str, Any]:
    """Retried-then-clicked pairs, exported as training data for a failure detector."""
    rows = state["log"].v2_labels(limit=min(max(limit, 1), 1000))
    return {"count": len(rows), "labels": rows}
