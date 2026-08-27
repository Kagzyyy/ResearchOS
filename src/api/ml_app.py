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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.db import FeedbackLog
from src.api.llm import is_configured
from src.api.ml_pipeline import MLPipeline
from src.config import load_config

CONFIG_PATH = os.environ.get("RESEARCHOS_ML_CONFIG", "configs/ml.yaml")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = load_config(CONFIG_PATH)
    state["cfg"] = cfg
    state["pipeline"] = MLPipeline(cfg)
    state["log"] = FeedbackLog(cfg["paths"]["feedback_db"])
    yield
    state.clear()


app = FastAPI(
    title="ResearchOS-ML",
    description="Search, explain and de-reference AI research papers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)
    generate: bool = True


class PapersRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=30)


class ExplainRequest(BaseModel):
    citation: str = Field(min_length=3, max_length=2000)
    generate: bool = True


class ClickRequest(BaseModel):
    query_id: int
    doc_id: str


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
    }


@app.post("/search")
def search(request: AskRequest) -> dict[str, Any]:
    """Retrieval and reranking only."""
    pipeline: MLPipeline = state["pipeline"]
    return {"query": request.query, "results": pipeline.retrieve(request.query, request.top_k)}


@app.post("/papers")
def papers(request: PapersRequest) -> dict[str, Any]:
    """Topic in, ranked papers out (not passages)."""
    pipeline: MLPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]
    result = pipeline.find_papers(request.topic, top_k=request.top_k)
    query_id = log.log_query(
        query=f"[papers] {request.topic}",
        answer=None,
        latency_ms=result["timing_ms"]["total"],
        backend="retrieval",
        citations=[
            {"doc_id": p["doc_id"], "score": p["score"], "title": p["title"]}
            for p in result["papers"]
        ],
    )
    return {"query_id": query_id, **result}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
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
    )
    return {"query_id": query_id, **result}


@app.post("/explain")
def explain(request: ExplainRequest) -> dict[str, Any]:
    """Resolve a citation to its paper and explain it in plain terms."""
    pipeline: MLPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]
    result = pipeline.explain_citation(request.citation, generate=request.generate)
    query_id = log.log_query(
        query=f"[explain] {request.citation}",
        answer=result.get("explanation"),
        latency_ms=result["timing_ms"]["total"],
        backend="groq" if result["generation_configured"] else "retrieval-only",
        citations=result.get("passages", []),
    )
    return {"query_id": query_id, **result}


@app.post("/feedback/click")
def feedback_click(request: ClickRequest) -> dict[str, Any]:
    log: FeedbackLog = state["log"]
    try:
        log.log_click(request.query_id, request.doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"logged": True, **log.stats()}


@app.get("/stats")
def stats() -> dict[str, int]:
    return state["log"].stats()
