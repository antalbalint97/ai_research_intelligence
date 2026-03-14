"""Document builder – converts PaperRecords into normalized DocumentRecords.

Produces a structured text block per paper, suitable for embedding and retrieval.
"""

from __future__ import annotations

import hashlib
import logging

from ingestion.topic_mapper import assign_topics
from pipeline.models import DocumentRecord, PaperRecord

logger = logging.getLogger(__name__)


def _arxiv_url(paper_id: str) -> str:
    """Derive an arXiv abstract URL from the paper ID."""
    # Strip version suffix for URL (e.g. "2401.12345v1" -> "2401.12345")
    clean_id = paper_id.split("v")[0] if "v" in paper_id else paper_id
    return f"https://arxiv.org/abs/{clean_id}"


def _make_doc_id(paper_id: str, doc_type: str) -> str:
    """Generate a deterministic document ID."""
    raw = f"{paper_id}:{doc_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_document_text(paper: PaperRecord, topic_primary: str, topic_secondary: list[str]) -> str:
    """Build the structured document text for a paper.

    Format:
        Title: ...
        Published: ...
        Categories: ...
        Primary Topic: ...
        Secondary Topics: ...
        Authors: ...

        Abstract:
        ...
    """
    secondary_str = ", ".join(topic_secondary) if topic_secondary else "None"
    pub_date = paper.published_date.isoformat() if paper.published_date else "Unknown"

    return (
        f"Title: {paper.title}\n"
        f"Published: {pub_date}\n"
        f"Categories: {paper.categories}\n"
        f"Primary Topic: {topic_primary}\n"
        f"Secondary Topics: {secondary_str}\n"
        f"Authors: {paper.authors}\n"
        f"\nAbstract:\n{paper.abstract}"
    )


def build_documents(papers: list[PaperRecord]) -> list[DocumentRecord]:
    """Convert paper records into structured DocumentRecords with topic assignments.

    Args:
        papers: Filtered list of PaperRecord objects.

    Returns:
        List of DocumentRecord objects ready for chunking and embedding.
    """
    documents: list[DocumentRecord] = []

    for paper in papers:
        # Assign topics via keyword heuristics
        assignment = assign_topics(paper.title, paper.abstract)

        content = _build_document_text(
            paper, assignment.primary_topic, assignment.secondary_topics
        )

        doc = DocumentRecord(
            doc_id=_make_doc_id(paper.id, "paper_abstract"),
            paper_id=paper.id,
            title=paper.title,
            content=content,
            source="arxiv",
            doc_type="paper_abstract",
            published_date=paper.published_date,
            categories=paper.categories,
            authors=paper.authors,
            primary_topic=assignment.primary_topic,
            secondary_topics=assignment.secondary_topics,
            language="en",
            url=_arxiv_url(paper.id),
            metadata={
                "topic_reason": assignment.topic_reason,
            },
        )
        documents.append(doc)

    logger.info("Built %d documents from %d papers", len(documents), len(papers))
    return documents
