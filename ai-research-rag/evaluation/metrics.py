"""Evaluation metrics for the AI Research RAG system.

Provides lightweight metrics for retrieval quality and answer assessment.
Designed for MVP: deterministic checks first, optional semantic hooks.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def retrieval_hit_rate(
    retrieved_ids: list[str], relevant_ids: list[str]
) -> float:
    """Compute retrieval hit rate (recall@K).

    Args:
        retrieved_ids: List of retrieved document IDs.
        relevant_ids: List of ground-truth relevant document IDs.

    Returns:
        Fraction of relevant documents that appear in retrieved set.
    """
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(relevant_ids)


def answer_non_empty(answer: str) -> bool:
    """Check that the generated answer is non-empty and has some substance.

    Args:
        answer: Generated answer text.

    Returns:
        True if the answer contains meaningful content.
    """
    stripped = answer.strip()
    if not stripped:
        return False
    # Minimum threshold: at least 20 characters and 3 words
    return len(stripped) >= 20 and len(stripped.split()) >= 3


def answer_structure_score(answer: str) -> float:
    """Heuristic score for answer quality based on structural features.

    Checks for paragraph structure, mentions of specifics, and length.
    Score range: 0.0 to 1.0.

    Args:
        answer: Generated answer text.

    Returns:
        Heuristic quality score.
    """
    score = 0.0

    if not answer.strip():
        return 0.0

    # Length check (prefer 100-1500 chars)
    length = len(answer.strip())
    if length >= 100:
        score += 0.25
    if length >= 300:
        score += 0.15

    # Paragraph structure
    paragraphs = [p for p in answer.strip().split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        score += 0.2

    # Contains specific language indicators
    specifics = ["research", "model", "method", "paper", "approach", "recent", "trend"]
    matches = sum(1 for s in specifics if s.lower() in answer.lower())
    score += min(0.25, matches * 0.05)

    # Mentions uncertainty appropriately
    uncertainty_markers = ["however", "limitation", "unclear", "further research", "insufficient"]
    if any(m in answer.lower() for m in uncertainty_markers):
        score += 0.15

    return min(1.0, score)


def latency_acceptable(latency_ms: float, threshold_ms: float = 10000.0) -> bool:
    """Check if response latency is within acceptable bounds.

    Args:
        latency_ms: Response latency in milliseconds.
        threshold_ms: Maximum acceptable latency.

    Returns:
        True if latency is within threshold.
    """
    return latency_ms <= threshold_ms


def semantic_similarity_placeholder(answer: str, reference: str) -> float:
    """Placeholder for semantic similarity scoring.

    In a full implementation, this would use BERTScore or similar.
    Currently returns a stub value.

    Args:
        answer: Generated answer.
        reference: Reference answer.

    Returns:
        Similarity score (stub: always 0.0).
    """
    # TODO: Integrate BERTScore or RAGAS semantic similarity when available
    logger.debug("Semantic similarity is a placeholder – returning 0.0")
    return 0.0
