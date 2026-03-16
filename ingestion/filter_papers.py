"""Paper filtering – date range and AI-category filters for arXiv papers.

Filters the raw paper list to retain only recent, AI-relevant papers.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from pipeline.models import PaperRecord

logger = logging.getLogger(__name__)

# arXiv category allowlist for AI-relevant papers
AI_CATEGORIES: set[str] = {
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.RO",
    "stat.ML",
    "cs.HC",
}

# Fallback keyword check when category metadata is missing or too broad
AI_KEYWORDS: list[str] = [
    "deep learning",
    "neural network",
    "transformer",
    "language model",
    "reinforcement learning",
    "computer vision",
    "natural language",
    "machine learning",
    "generative",
    "diffusion",
    "attention mechanism",
]


def _has_ai_category(categories: str) -> bool:
    """Check if any arXiv category matches the AI allowlist."""
    cats = set(categories.replace(",", " ").split())
    return bool(cats & AI_CATEGORIES)


def _has_ai_keyword(title: str, abstract: str) -> bool:
    """Fallback: check if title or abstract contains AI-relevant keywords."""
    text = f"{title} {abstract}".lower()
    return any(kw in text for kw in AI_KEYWORDS)


def filter_papers(
    papers: list[PaperRecord],
    lookback_months: int = 9,
    reference_date: date | None = None,
) -> list[PaperRecord]:
    """Filter papers to recent AI-relevant subset.

    Args:
        papers: Full list of loaded paper records.
        lookback_months: Only keep papers published within this many months.
        reference_date: Date to calculate lookback from (defaults to today).

    Returns:
        Filtered list of PaperRecord objects.
    """
    ref = reference_date or date.today()
    cutoff = ref - timedelta(days=lookback_months * 30)

    filtered: list[PaperRecord] = []
    date_skipped = 0
    topic_skipped = 0

    for paper in papers:
        # Date filter
        if paper.published_date and paper.published_date < cutoff:
            date_skipped += 1
            continue

        # Category + keyword filter
        if not _has_ai_category(paper.categories) and not _has_ai_keyword(
            paper.title, paper.abstract
        ):
            topic_skipped += 1
            continue

        filtered.append(paper)

    logger.info(
        "Filtered %d → %d papers (date_skipped=%d, topic_skipped=%d)",
        len(papers),
        len(filtered),
        date_skipped,
        topic_skipped,
    )
    return filtered
