"""
retriever_faiss.py

FAISS-backed semantic retriever for the AI Research RAG pipeline.

Replaces the earlier pgvector-based retriever with:
- local FAISS index loading
- JSONL metadata loading
- cosine-style similarity search (IndexFlatIP + normalized embeddings)
- lightweight post-filtering on metadata

Designed to match the existing rag_pipeline.py interface:
    search_documents(
        query_embedding=query_embedding,
        top_k=retrieval_k,
        primary_topic=...,
        category=...,
        date_from=...,
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384

DEFAULT_INDEX_PATH = "data/index/arxiv_ai.index"
DEFAULT_METADATA_PATH = "data/index/arxiv_ai_metadata.jsonl"

_index = None
_metadata: list[dict[str, Any]] | None = None


def _get_index_path(index_path: str | None = None) -> str:
    return index_path or os.environ.get("FAISS_INDEX_PATH", DEFAULT_INDEX_PATH)


def _get_metadata_path(metadata_path: str | None = None) -> str:
    return metadata_path or os.environ.get("FAISS_METADATA_PATH", DEFAULT_METADATA_PATH)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text[:19]).date()
    except ValueError:
        return None


def _normalize_query_embedding(query_embedding: list[float]) -> np.ndarray:
    vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

    if vector.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Query embedding dimension mismatch: expected {EMBEDDING_DIM}, got {vector.shape[1]}"
        )

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

    faiss.normalize_L2(vector)
    return vector


def load_index(index_path: str | None = None, force_reload: bool = False):
    """Load FAISS index lazily and cache it in memory."""
    global _index

    if _index is not None and not force_reload:
        return _index

    path = Path(_get_index_path(index_path))
    if not path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {path}")

    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

    logger.info("Loading FAISS index from %s", path)
    _index = faiss.read_index(str(path))
    logger.info("FAISS index loaded: ntotal=%d", _index.ntotal)
    return _index


def load_metadata(metadata_path: str | None = None, force_reload: bool = False) -> list[dict[str, Any]]:
    """Load JSONL metadata lazily and cache it in memory."""
    global _metadata

    if _metadata is not None and not force_reload:
        return _metadata

    path = Path(_get_metadata_path(metadata_path))
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    logger.info("Loading metadata from %s", path)
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed metadata row at line %d: %s", line_number, exc)

    logger.info("Loaded %d metadata rows", len(rows))
    _metadata = rows
    return _metadata


def warmup(index_path: str | None = None, metadata_path: str | None = None) -> None:
    """Preload index + metadata to avoid first-query latency."""
    index = load_index(index_path=index_path)
    metadata = load_metadata(metadata_path=metadata_path)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata size mismatch: ntotal={index.ntotal}, metadata_rows={len(metadata)}"
        )

    logger.info("Retriever warmup complete")


def _matches_filters(
    row: dict[str, Any],
    primary_topic: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
) -> bool:
    if primary_topic:
        row_topic = row.get("primary_topic_en") or row.get("primary_topic") or ""
        if row_topic != primary_topic:
            return False

    if category:
        categories = row.get("categories", [])
        if isinstance(categories, str):
            categories_text = categories
            categories_list = categories.replace(",", " ").split()
        elif isinstance(categories, list):
            categories_list = [str(c) for c in categories]
            categories_text = " ".join(categories_list)
        else:
            categories_list = []
            categories_text = ""

        if category not in categories_list and category not in categories_text:
            return False

    if date_from:
        cutoff = _parse_date(date_from)
        row_date = _parse_date(row.get("published_date"))
        if cutoff and row_date and row_date < cutoff:
            return False
        if cutoff and row_date is None:
            return False

    return True


def _to_result(row: dict[str, Any], similarity: float, rank: int) -> dict[str, Any]:
    """
    Map the curated metadata schema back to the keys expected by the current pipeline.
    """
    return {
        "rank": rank,
        "doc_id": row.get("paper_id", ""),
        "paper_id": row.get("paper_id", ""),
        "title": row.get("title", ""),
        "content": row.get("content", ""),
        "abstract": row.get("abstract", ""),
        "source": row.get("source", "arxiv"),
        "doc_type": row.get("doc_type", "paper_abstract"),
        "published_date": row.get("published_date"),
        "categories": row.get("categories", []),
        "authors": row.get("authors", []),
        "primary_topic": row.get("primary_topic_en", row.get("primary_topic", "")),
        "secondary_topics": row.get(
            "secondary_topics_en",
            row.get("secondary_topics", []),
        ),
        "url": row.get("arxiv_abs_url", row.get("url")),
        "arxiv_abs_url": row.get("arxiv_abs_url"),
        "arxiv_pdf_url": row.get("arxiv_pdf_url"),
        "similarity": float(similarity),
        "metadata": {
            "primary_topic_hu": row.get("primary_topic_hu", ""),
            "secondary_topics_hu": row.get("secondary_topics_hu", []),
            "topic_reason": row.get("topic_reason", ""),
        },
    }


def search_documents(
    query_embedding: list[float],
    top_k: int = 20,
    primary_topic: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    index_path: str | None = None,
    metadata_path: str | None = None,
    search_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the most similar documents from FAISS and apply optional metadata filters.

    Args:
        query_embedding: Query vector (dim=384).
        top_k: Number of filtered results to return.
        primary_topic: Exact match against primary_topic_en.
        category: Category code filter, e.g. 'cs.CL'.
        date_from: Lower date bound, e.g. '2025-01-01'.
        index_path: Optional override for FAISS index file path.
        metadata_path: Optional override for metadata JSONL path.
        search_k: How many initial candidates to pull from FAISS before filtering.
                  Defaults to max(top_k * 5, 50).

    Returns:
        List of document dicts with similarity scores.
    """
    if top_k <= 0:
        return []

    index = load_index(index_path=index_path)
    metadata = load_metadata(metadata_path=metadata_path)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Index/metadata size mismatch: ntotal={index.ntotal}, metadata_rows={len(metadata)}"
        )

    vector = _normalize_query_embedding(query_embedding)

    initial_k = search_k or max(top_k * 5, 50)
    initial_k = min(initial_k, len(metadata))

    scores, indices = index.search(vector, initial_k)

    results: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()

    for rank_idx, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue

        row = metadata[idx]

        if not _matches_filters(
            row,
            primary_topic=primary_topic,
            category=category,
            date_from=date_from,
        ):
            continue

        paper_id = str(row.get("paper_id", ""))
        if paper_id and paper_id in seen_paper_ids:
            continue

        if paper_id:
            seen_paper_ids.add(paper_id)

        results.append(_to_result(row=row, similarity=float(score), rank=rank_idx))

        if len(results) >= top_k:
            break

    logger.info(
        "FAISS retrieval complete: requested_top_k=%d, returned=%d, search_k=%d, filters={topic=%s, category=%s, date_from=%s}",
        top_k,
        len(results),
        initial_k,
        primary_topic,
        category,
        date_from,
    )
    return results