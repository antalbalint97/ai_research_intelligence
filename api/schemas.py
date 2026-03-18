"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel

from pipeline.models import (
    QueryFilters,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)


class HealthResponse(BaseModel):
    status: str


class MetricsResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int = 0


__all__ = [
    "HealthResponse",
    "MetricsResponse",
    "QueryFilters",
    "QueryRequest",
    "QueryResponse",
    "SourceCitation",
]