"""FastAPI application – AI Research Intelligence RAG API.

Provides endpoints for health checks, queries, topic listing, and metrics.
Serves the lightweight frontend from the /frontend directory.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import RequestLoggingMiddleware
from api.schemas import QueryRequest, QueryResponse
from ingestion.topic_mapper import get_all_topics
from pipeline.rag_pipeline import run_query
from pipeline.retriever import ensure_schema

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------
_startup_time = time.time()
_query_count = 0
_total_latency = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan – initialize database schema on startup."""
    try:
        ensure_schema()
        logger.info("Database schema verified on startup")
    except Exception as e:
        logger.warning("Could not verify database schema on startup: %s", e)
    yield


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Research Intelligence RAG",
    description=(
        "A RAG-based assistant for exploring recent AI research trends "
        "and translating them into strategic insights for investors and innovation teams."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
app.add_middleware(RequestLoggingMiddleware)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "uptime_s": round(time.time() - _startup_time, 1)}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Execute a RAG query against the AI research corpus.

    Accepts a natural-language question with optional filters and returns
    a synthesized answer with source citations.
    """
    global _query_count, _total_latency
    try:
        response = run_query(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
        )
        _query_count += 1
        _total_latency += response.latency_ms
        return response
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")


@app.get("/topics")
async def topics() -> list[dict]:
    """Return the full AI topic taxonomy with English and Hungarian labels."""
    return get_all_topics()


@app.get("/metrics")
async def metrics() -> dict:
    """Return basic operational metrics."""
    avg_latency = (_total_latency / _query_count) if _query_count > 0 else 0.0
    return {
        "total_queries": _query_count,
        "avg_latency_ms": round(avg_latency, 1),
        "uptime_s": round(time.time() - _startup_time, 1),
    }


@app.get("/")
async def root() -> FileResponse:
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return FileResponse(str(FRONTEND_DIR / "index.html"))
