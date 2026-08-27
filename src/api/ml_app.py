"""FastAPI app for ResearchOS-ML.

Same surface as the parent project's app so the Gradio UI works against either,
plus the fields this corpus needs: paper ids, arXiv links, section names and the
US5 confidence flag.

    RESEARCHOS_ML_CONFIG=configs/ml.yaml uvicorn src.api.ml_app:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.db import FeedbackLog
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
    description="Question answering over recent arXiv ML/AI papers.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)
    generate: bool = True


class ClickRequest(BaseModel):
    query_id: int
    doc_id: str


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline: MLPipeline | None = state.get("pipeline")
    return {
        "status": "ok" if pipeline else "loading",
        "chunks": len(pipeline.corpus) if pipeline else 0,
        "papers": len(pipeline.papers) if pipeline else 0,
        "reranker_model": state["cfg"]["reranker"]["model"] if "cfg" in state else None,
    }


@app.post("/search")
def search(request: AskRequest) -> dict[str, Any]:
    """Retrieval and reranking only."""
    pipeline: MLPipeline = state["pipeline"]
    return {"query": request.query, "results": pipeline.retrieve(request.query)[: request.top_k]}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """Full pipeline, logged."""
    pipeline: MLPipeline = state["pipeline"]
    log: FeedbackLog = state["log"]

    result = pipeline.answer(request.query, generate=request.generate)
    query_id = log.log_query(
        query=request.query,
        answer=result["answer"],
        latency_ms=result["timing_ms"]["total"],
        backend="torch",
        citations=result["citations"],
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
