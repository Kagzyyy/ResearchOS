"""FastAPI app: retrieve -> rerank -> Claude, with citation-click logging.

    uvicorn src.api.app:app --reload
    RESEARCHOS_CONFIG=configs/app.yaml uvicorn src.api.app:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.db import FeedbackLog
from src.api.pipeline import ResearchPipeline
from src.config import load_config

CONFIG_PATH = os.environ.get("RESEARCHOS_CONFIG", "configs/app.yaml")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build indices and load models once, at startup."""
    cfg = load_config(CONFIG_PATH)
    state["cfg"] = cfg
    state["pipeline"] = ResearchPipeline(cfg)
    state["log"] = FeedbackLog(cfg["paths"]["feedback_db"])
    yield
    state.clear()


app = FastAPI(
    title="ResearchOS",
    description="Scientific literature retrieval with a fine-tuned cross-encoder reranker.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    generate: bool = True


class ClickRequest(BaseModel):
    query_id: int
    doc_id: str


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus what the pipeline is actually running."""
    pipeline: ResearchPipeline | None = state.get("pipeline")
    return {
        "status": "ok" if pipeline else "loading",
        "documents": len(pipeline.corpus) if pipeline else 0,
        "reranker_backend": pipeline.backend if pipeline else None,
        "reranker_model": state["cfg"]["reranker"]["model"] if "cfg" in state else None,
    }


@app.post("/search")
def search(request: AskRequest) -> dict[str, Any]:
    """Retrieval and reranking only - no generation."""
    pipeline: ResearchPipeline = state["pipeline"]
    passages = pipeline.retrieve(request.query)[: request.top_k]
    return {"query": request.query, "results": passages}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """Full pipeline. Logs the query and the citations shown."""
    pipeline: ResearchPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]

    result = pipeline.answer(request.query, top_k_final=request.top_k, generate=request.generate)
    query_id = log.log_query(
        query=request.query,
        answer=result["answer"],
        latency_ms=result["timing_ms"]["total"],
        backend=result["backend"],
        citations=result["citations"],
    )
    return {"query_id": query_id, **result}


@app.post("/feedback/click")
def feedback_click(request: ClickRequest) -> dict[str, Any]:
    """Record that a user opened a cited passage. This is the retraining signal."""
    log: FeedbackLog = state["log"]
    try:
        log.log_click(request.query_id, request.doc_id)
    except Exception as exc:  # noqa: BLE001 - surface storage failures to the caller
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"logged": True, **log.stats()}


@app.get("/stats")
def stats() -> dict[str, int]:
    """Counts from the feedback log."""
    return state["log"].stats()
