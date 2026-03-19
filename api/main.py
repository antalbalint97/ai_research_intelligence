"""FastAPI application entrypoint for the AI Research Intelligence service."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RequestLoggingMiddleware
from api.schemas import HealthResponse, MetricsResponse, QueryRequest, QueryResponse
from ingestion.embedder_optimized import warmup_embedder
from pipeline.rag_pipeline import run_query
from pipeline.retriever_faiss import warmup as warmup_retriever

TOTAL_REQUESTS = 0
START_TIME = time.time()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Intelligence API",
    description="Local RAG API over curated arXiv AI research abstracts",
    version="0.1.0",
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.on_event("startup")
# def startup_event() -> None:
#     """Warm up heavy resources once to avoid first-request latency."""
#     startup_start = time.perf_counter()

#     try:
#         warmup_embedder()
#     except Exception as exc:
#         logger.warning("Embedder warmup failed: %s", exc)

#     try:
#         warmup_retriever()
#     except Exception as exc:
#         logger.warning("Retriever warmup failed: %s", exc)

#     logger.info(
#         "API startup warmup finished in %.1f ms",
#         (time.perf_counter() - startup_start) * 1000,
#     )

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/warmup")
def warmup():
    warmup_embedder()
    warmup_retriever()
    return {"status": "warmed"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        service="ai-research-intelligence",
        status="ok",
        uptime_seconds=time.time() - START_TIME,
        total_requests=TOTAL_REQUESTS,
    )


@app.get("/ready")
def ready():
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    global TOTAL_REQUESTS
    TOTAL_REQUESTS += 1

    return run_query(
        query=request.query,
        filters=request.filters,
        top_k=request.top_k,
        retrieval_k=request.retrieval_k,
        mode=request.mode,
    )