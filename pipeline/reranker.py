"""Cross-encoder reranker – reranks retrieved documents by query relevance.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for precise relevance scoring.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Lazy-loaded model singleton
_reranker = None


def _get_reranker():
    """Lazy-load the CrossEncoder model."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
    return _reranker


def rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    content_key: str = "content",
) -> list[dict]:
    """Rerank documents using cross-encoder relevance scoring.

    Args:
        query: User query string.
        documents: List of retrieved document dicts.
        top_k: Number of top documents to return after reranking.
        content_key: Key in document dict containing the text to compare.

    Returns:
        Reranked list of document dicts with added 'rerank_score' field.
    """
    if not documents:
        return []

    reranker = _get_reranker()

    pairs = [(query, doc.get(content_key, "")) for doc in documents]
    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
    result = ranked[:top_k]

    logger.info(
        "Reranked %d → %d documents (top score=%.4f)",
        len(documents),
        len(result),
        result[0]["rerank_score"] if result else 0.0,
    )
    return result
