"""Ingestion orchestrator – runs the full offline pipeline.

Steps:
  1. Load raw arXiv data from a local file
  2. Filter to last N months
  3. Filter to AI categories / keywords
  4. Assign topics
  5. Build normalized documents
  6. Chunk documents
  7. Embed documents
  8. Upsert into pgvector
"""

from __future__ import annotations

import logging
import os
import sys

from ingestion.build_documents import build_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents
from ingestion.filter_papers import filter_papers
from ingestion.load_arxiv import load_arxiv_papers
from pipeline.retriever import ensure_schema, upsert_documents

logger = logging.getLogger(__name__)


def run_ingestion(
    data_path: str | None = None,
    lookback_months: int | None = None,
    database_url: str | None = None,
) -> dict:
    """Execute the full ingestion pipeline.

    Args:
        data_path: Path to raw arXiv data file. Falls back to ARXIV_DATA_PATH env var.
        lookback_months: Filter window in months. Falls back to LOOKBACK_MONTHS env var.
        database_url: PostgreSQL connection string. Falls back to DATABASE_URL env var.

    Returns:
        Summary dict with counts for each stage.
    """
    data_path = data_path or os.environ.get("ARXIV_DATA_PATH", "data/raw/arxiv-metadata.jsonl")
    lookback_months = lookback_months or int(os.environ.get("LOOKBACK_MONTHS", "9"))
    database_url = database_url or os.environ.get(
        "DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/arxiv_rag"
    )

    logger.info("=== Starting ingestion pipeline ===")

    # Step 1: Load
    logger.info("Step 1/7: Loading papers from %s", data_path)
    papers = load_arxiv_papers(data_path)
    logger.info("Loaded %d raw papers", len(papers))

    # Step 2–3: Filter
    logger.info("Step 2/7: Filtering to last %d months, AI categories", lookback_months)
    papers = filter_papers(papers, lookback_months=lookback_months)
    logger.info("After filtering: %d papers", len(papers))

    if not papers:
        logger.warning("No papers after filtering. Check data and date range.")
        return {"loaded": 0, "filtered": 0, "documents": 0, "chunks": 0, "embedded": 0}

    # Step 4–5: Build documents (includes topic assignment)
    logger.info("Step 3/7: Building documents with topic assignment")
    documents = build_documents(papers)
    logger.info("Built %d documents", len(documents))

    # Step 6: Chunk
    logger.info("Step 4/7: Chunking documents")
    chunks = chunk_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    # Step 7: Embed
    logger.info("Step 5/7: Embedding %d chunks", len(chunks))
    chunks = embed_documents(chunks)
    logger.info("Embedded %d chunks", len(chunks))

    # Step 8: Upsert into pgvector
    logger.info("Step 6/7: Ensuring database schema")
    ensure_schema(database_url)

    logger.info("Step 7/7: Upserting %d documents into pgvector", len(chunks))
    upsert_documents(chunks, database_url)

    summary = {
        "loaded": len(papers),
        "filtered": len(papers),
        "documents": len(documents),
        "chunks": len(chunks),
        "embedded": len(chunks),
    }
    logger.info("=== Ingestion complete: %s ===", summary)
    return summary


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        summary = run_ingestion()
        print(f"\nIngestion complete: {summary}")
    except Exception:
        logger.exception("Ingestion pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
