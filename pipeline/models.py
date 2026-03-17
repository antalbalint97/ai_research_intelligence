"""Domain models for the AI Research RAG pipeline.

Defines Pydantic models used across ingestion, retrieval, and API layers.
All models use strict type annotations for production-grade data validation.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    """Raw paper record parsed from arXiv dataset."""

    id: str = Field(..., description="arXiv paper ID (e.g. '2401.12345')")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(default="", description="Paper abstract text")
    authors: str = Field(default="", description="Authors as a comma-separated string")
    categories: str = Field(default="", description="Space-separated arXiv category codes")
    published_date: Optional[date] = Field(
        default=None, description="Publication or last update date"
    )
    doi: Optional[str] = Field(default=None, description="DOI if available")
    journal_ref: Optional[str] = Field(default=None, description="Journal reference if available")
    full_text_path: Optional[str] = Field(
        default=None, description="Path to full-text file (future use)"
    )


class DocumentRecord(BaseModel):
    """Normalized document record ready for embedding and indexing."""

    doc_id: str = Field(..., description="Unique document identifier")
    paper_id: str = Field(..., description="Source arXiv paper ID")
    title: str = Field(..., description="Paper title")
    content: str = Field(..., description="Full document text for embedding")
    source: str = Field(default="arxiv", description="Data source identifier")
    doc_type: str = Field(
        default="paper_abstract",
        description="Document type: 'paper_abstract' or 'paper_chunk'",
    )
    published_date: Optional[date] = Field(default=None, description="Publication date")
    categories: str = Field(default="", description="arXiv category codes")
    authors: str = Field(default="", description="Authors string")
    primary_topic: str = Field(default="", description="Primary AI topic label")
    secondary_topics: list[str] = Field(
        default_factory=list, description="Secondary AI topic labels"
    )
    language: str = Field(default="en", description="Content language code")
    url: Optional[str] = Field(default=None, description="arXiv paper URL")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[list[float]] = Field(
        default=None, description="Vector embedding (dim=384 for MiniLM)"
    )


class SourceCitation(BaseModel):
    """Citation reference returned to the user with each RAG answer."""

    title: str
    url: Optional[str] = None
    published_date: Optional[str] = None
    primary_topic: str = ""
    relevance_score: float = 0.0


class QueryFilters(BaseModel):
    """Optional filters for the /query endpoint."""

    primary_topic: Optional[str] = None
    category: Optional[str] = None
    date_from: Optional[str] = None


class QueryRequest(BaseModel):
    """Incoming query request from the user."""

    query: str = Field(..., min_length=3, description="Natural-language research question")
    filters: Optional[QueryFilters] = None
    mode: Literal["fast", "full"] = Field(
        default="fast",
        description="Inference mode. 'fast' favors lower latency, 'full' favors answer depth.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    retrieval_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of initial candidates to retrieve before reranking. "
        "If omitted, mode-specific defaults are used.",
    )


class QueryResponse(BaseModel):
    """Full response returned by the RAG pipeline."""

    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    retrieval_count: int = 0
    reranked_count: int = 0
    mode: Literal["fast", "full"] = "fast"
    timings: dict[str, float] = Field(default_factory=dict)
    prompt_chars: int = 0
    answer_chars: int = 0
