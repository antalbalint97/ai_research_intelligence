"""RAG pipeline – orchestrates the online query flow.

Coordinates: query embedding → retrieval → reranking → prompt assembly → generation.
Supports:
- fast mode: shorter, cheaper responses for business Q&A
- full mode: richer synthesis with higher latency
"""

from __future__ import annotations

import logging
import time

from ingestion.embedder import embed_query
from pipeline.generator import generate
from pipeline.models import QueryFilters, QueryResponse, SourceCitation
from pipeline.prompt import (
    build_prompt,
    build_prompt_fast,
    get_system_prompt,
    get_system_prompt_fast,
)
from pipeline.reranker import rerank
from pipeline.retriever_faiss import search_documents

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVAL_K = 20
FAST_RETRIEVAL_K = 8
DEFAULT_TOP_K = 5
FAST_TOP_K = 3


def _normalize_mode(mode: str | None) -> str:
    return "fast" if (mode or "").strip().lower() == "fast" else "full"


def _resolve_top_k(mode: str, requested_top_k: int) -> int:
    default_value = FAST_TOP_K if mode == "fast" else DEFAULT_TOP_K
    limit = FAST_TOP_K if mode == "fast" else 50
    value = requested_top_k or default_value
    return max(1, min(value, limit))


def _resolve_retrieval_k(mode: str, requested_retrieval_k: int | None) -> int:
    default_value = FAST_RETRIEVAL_K if mode == "fast" else DEFAULT_RETRIEVAL_K
    limit = FAST_RETRIEVAL_K if mode == "fast" else 100
    value = requested_retrieval_k or default_value
    return max(1, min(value, limit))


def run_query(
    query: str,
    filters: QueryFilters | None = None,
    top_k: int = DEFAULT_TOP_K,
    retrieval_k: int | None = None,
    mode: str = "full",
) -> QueryResponse:
    """Execute the full RAG pipeline for a user query."""
    total_start = time.perf_counter()
    mode = _normalize_mode(mode)
    resolved_top_k = _resolve_top_k(mode, top_k)
    resolved_retrieval_k = _resolve_retrieval_k(mode, retrieval_k)

    timings: dict[str, float] = {
        "embed_ms": 0.0,
        "retrieve_ms": 0.0,
        "rerank_ms": 0.0,
        "prompt_build_ms": 0.0,
        "generate_ms": 0.0,
        "total_ms": 0.0,
    }

    t0 = time.perf_counter()
    logger.info("Embedding query | mode=%s | query=%s", mode, query[:80])
    query_embedding = embed_query(query)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    primary_topic = filters.primary_topic if filters else None
    category = filters.category if filters else None
    date_from = filters.date_from if filters else None

    t0 = time.perf_counter()
    results = search_documents(
        query_embedding=query_embedding,
        top_k=resolved_retrieval_k,
        primary_topic=primary_topic,
        category=category,
        date_from=date_from,
    )
    timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    retrieval_count = len(results)

    if not results:
        timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 1)
        return QueryResponse(
            answer="No relevant documents were found for your query. Try broadening your search or adjusting filters.",
            sources=[],
            latency_ms=timings["total_ms"],
            model="none",
            retrieval_count=0,
            reranked_count=0,
            mode=mode,
            timings=timings,
            prompt_chars=0,
            answer_chars=0,
        )

    t0 = time.perf_counter()
    reranked = rerank(query, results, top_k=resolved_top_k)
    timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    reranked_count = len(reranked)

    t0 = time.perf_counter()
    if mode == "fast":
        prompt = build_prompt_fast(query, reranked)
        system_prompt = get_system_prompt_fast()
    else:
        prompt = build_prompt(query, reranked)
        system_prompt = get_system_prompt()

    timings["prompt_build_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    prompt_chars = len(prompt)

    t0 = time.perf_counter()
    answer, model_name = generate(prompt, system_prompt, mode=mode)
    timings["generate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    answer_chars = len((answer or "").strip())

    sources = [
        SourceCitation(
            title=doc.get("title", ""),
            url=doc.get("url"),
            published_date=str(doc.get("published_date", "")),
            primary_topic=doc.get("primary_topic", ""),
            relevance_score=round(doc.get("rerank_score", doc.get("similarity", 0.0)), 4),
        )
        for doc in reranked
    ]

    timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 1)

    logger.info(
        "run_query timings | mode=%s embed=%sms retrieve=%sms rerank=%sms prompt=%sms generate=%sms total=%sms prompt_chars=%d answer_chars=%d top_k=%d retrieval_k=%d",
        mode,
        timings["embed_ms"],
        timings["retrieve_ms"],
        timings["rerank_ms"],
        timings["prompt_build_ms"],
        timings["generate_ms"],
        timings["total_ms"],
        prompt_chars,
        answer_chars,
        resolved_top_k,
        resolved_retrieval_k,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=timings["total_ms"],
        model=model_name,
        retrieval_count=retrieval_count,
        reranked_count=reranked_count,
        mode=mode,
        timings=timings,
        prompt_chars=prompt_chars,
        answer_chars=answer_chars,
    )
