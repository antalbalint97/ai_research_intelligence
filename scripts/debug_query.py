#!/usr/bin/env python3
"""
test_query.py

Simple CLI smoke test for the AI Research Intelligence RAG pipeline.

Supports two modes:
1. retrieval-only test
2. full RAG test (retrieval + rerank + generation)
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from ingestion.embedder_optimized import embed_query
from pipeline.retriever_faiss import search_documents, warmup
from pipeline.reranker import rerank

logger = logging.getLogger("test_query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FAISS retrieval and optional full RAG flow")
    parser.add_argument("--query", required=True, help="Natural-language query to test")
    parser.add_argument("--top-k", type=int, default=5, help="Final number of results to display (default: 5)")
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=20,
        help="Initial number of retrieval candidates before reranking (default: 20)",
    )
    parser.add_argument("--topic", default=None, help="Optional primary topic filter")
    parser.add_argument("--category", default=None, help="Optional arXiv category filter")
    parser.add_argument("--date-from", default=None, help="Optional lower publication date bound")
    parser.add_argument("--full-rag", action="store_true", help="Run full RAG pipeline after retrieval test")
    parser.add_argument("--json", action="store_true", help="Print results as JSON instead of formatted text")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def format_result(doc: dict[str, Any], index: int) -> str:
    title = doc.get("title", "Untitled")
    topic = doc.get("primary_topic", "")
    published = doc.get("published_date", "")
    score = doc.get("rerank_score", doc.get("similarity", 0.0))
    url = doc.get("url") or doc.get("arxiv_abs_url") or ""
    abstract = (doc.get("abstract") or "").strip()
    if len(abstract) > 500:
        abstract = abstract[:500].rstrip() + "..."

    lines = [
        f"[{index}] {title}",
        f"    topic: {topic}",
        f"    published: {published}",
        f"    score: {score:.4f}",
    ]
    if url:
        lines.append(f"    url: {url}")
    if abstract:
        lines.append(f"    abstract: {abstract}")
    return "\n".join(lines)


def run_retrieval_test(
    query: str,
    top_k: int,
    retrieval_k: int,
    topic: str | None,
    category: str | None,
    date_from: str | None,
) -> list[dict[str, Any]]:
    logger.info("Warming up retriever")
    warmup()

    logger.info("Embedding query")
    query_vector = embed_query(query)

    logger.info("Running FAISS retrieval")
    results = search_documents(
        query_embedding=query_vector,
        top_k=retrieval_k,
        primary_topic=topic,
        category=category,
        date_from=date_from,
    )

    logger.info("Retrieved %d candidates", len(results))
    if not results:
        return []

    logger.info("Running reranker")
    reranked = rerank(query=query, documents=results, top_k=top_k)
    logger.info("Returning %d reranked results", len(reranked))
    return reranked


def print_formatted_results(query: str, results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    if not results:
        print("No results found.")
        return

    for i, doc in enumerate(results, start=1):
        print(format_result(doc, i))
        print("-" * 100)


def print_json_results(query: str, results: list[dict[str, Any]]) -> None:
    payload = {"query": query, "results": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_full_rag(
    query: str,
    top_k: int,
    retrieval_k: int,
    topic: str | None,
    category: str | None,
    date_from: str | None,
) -> None:
    from pipeline.models import QueryFilters
    from pipeline.rag_pipeline import run_query

    filters = QueryFilters(
        primary_topic=topic,
        category=category,
        date_from=date_from,
    )

    response = run_query(
        query=query,
        filters=filters,
        top_k=top_k,
        retrieval_k=retrieval_k,
    )

    print("\n" + "=" * 100)
    print("FULL RAG RESPONSE")
    print("=" * 100)
    print(f"Model: {response.model}")
    print(f"Latency (ms): {response.latency_ms}")
    print(f"Retrieved: {response.retrieval_count}")
    print(f"Reranked: {response.reranked_count}")
    print(f"Prompt chars: {response.prompt_chars}")
    print(f"Answer chars: {response.answer_chars}")
    print(f"Timings: {response.timings}")
    print("\nANSWER:\n")
    print(response.answer)

    if response.sources:
        print("\nSOURCES:\n")
        for i, src in enumerate(response.sources, start=1):
            print(
                f"[{i}] {src.title} | topic={src.primary_topic} | "
                f"published={src.published_date} | score={src.relevance_score:.4f}"
            )
            if src.url:
                print(f"    {src.url}")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    results = run_retrieval_test(
        query=args.query,
        top_k=args.top_k,
        retrieval_k=args.retrieval_k,
        topic=args.topic,
        category=args.category,
        date_from=args.date_from,
    )

    if args.json:
        print_json_results(args.query, results)
    else:
        print_formatted_results(args.query, results)

    if args.full_rag:
        run_full_rag(
            query=args.query,
            top_k=args.top_k,
            retrieval_k=args.retrieval_k,
            topic=args.topic,
            category=args.category,
            date_from=args.date_from,
        )


if __name__ == "__main__":
    main()
