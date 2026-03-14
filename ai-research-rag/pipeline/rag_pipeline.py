"""RAG pipeline – orchestrates the online query flow.

Coordinates: query embedding → retrieval → reranking → prompt assembly → generation.
"""

from __future__ import annotations

import logging
import time

from ingestion.embedder import embed_query
from pipeline.generator import generate
from pipeline.models import QueryFilters, QueryResponse, SourceCitation
from pipeline.prompt import build_prompt, get_system_prompt
from pipeline.reranker import rerank
from pipeline.retriever import search_documents

logger = logging.getLogger(__name__)

# Default retrieval parameters
DEFAULT_RETRIEVAL_K = 20


def run_query(
    query: str,
    filters: QueryFilters | None = None,
    top_k: int = 5,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
) -> QueryResponse:
    """Execute the full RAG pipeline for a user query.

    Online flow:
      1. Embed the query
      2. Retrieve top-K candidates from pgvector
      3. Rerank with cross-encoder
      4. Assemble prompt with context
      5. Generate answer
      6. Format response

    Args:
        query: Natural-language research question.
        filters: Optional metadata filters (topic, category, date).
        top_k: Number of sources to return after reranking.
        retrieval_k: Number of candidates to retrieve before reranking.

    Returns:
        QueryResponse with answer, sources, and metadata.
    """
    start = time.time()

    # Step 1: Embed query
    logger.info("Embedding query: %s", query[:80])
    query_embedding = embed_query(query)

    # Step 2: Retrieve
    primary_topic = filters.primary_topic if filters else None
    category = filters.category if filters else None
    date_from = filters.date_from if filters else None

    results = search_documents(
        query_embedding=query_embedding,
        top_k=retrieval_k,
        primary_topic=primary_topic,
        category=category,
        date_from=date_from,
    )
    retrieval_count = len(results)
    logger.info("Retrieved %d candidates", retrieval_count)

    if not results:
        elapsed = (time.time() - start) * 1000
        return QueryResponse(
            answer="No relevant documents were found for your query. "
            "Try broadening your search or adjusting filters.",
            sources=[],
            latency_ms=round(elapsed, 1),
            model="none",
            retrieval_count=0,
            reranked_count=0,
        )

    # Step 3: Rerank
    reranked = rerank(query, results, top_k=top_k)
    reranked_count = len(reranked)

    # Step 4: Build prompt
    prompt = build_prompt(query, reranked)
    system_prompt = get_system_prompt()

    # Step 5: Generate
    answer, model_name = generate(prompt, system_prompt)

    # Step 6: Format sources
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

    elapsed = (time.time() - start) * 1000

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round(elapsed, 1),
        model=model_name,
        retrieval_count=retrieval_count,
        reranked_count=reranked_count,
    )
