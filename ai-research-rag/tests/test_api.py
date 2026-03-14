"""Tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_returns_ok(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime_s" in data


class TestTopicsEndpoint:
    """Test the /topics endpoint."""

    def test_topics_returns_list(self) -> None:
        response = client.get("/topics")
        assert response.status_code == 200
        topics = response.json()
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_topics_structure(self) -> None:
        response = client.get("/topics")
        topics = response.json()
        for t in topics:
            assert "topic" in t
            assert "translation_hu" in t


class TestMetricsEndpoint:
    """Test the /metrics endpoint."""

    def test_metrics_returns_data(self) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "avg_latency_ms" in data
        assert "uptime_s" in data


class TestQueryEndpointValidation:
    """Test /query request validation (does not require database)."""

    def test_empty_query_rejected(self) -> None:
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_rejected(self) -> None:
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_short_query_rejected(self) -> None:
        response = client.post("/query", json={"query": "ab"})
        assert response.status_code == 422

    def test_invalid_top_k_rejected(self) -> None:
        response = client.post("/query", json={"query": "test query", "top_k": 0})
        assert response.status_code == 422

    def test_valid_query_schema_accepted(self) -> None:
        """Valid schema should be accepted (may fail on DB, but not on validation)."""
        response = client.post(
            "/query",
            json={
                "query": "What are the recent trends in AI?",
                "top_k": 5,
                "filters": {"primary_topic": "Large Language Models"},
            },
        )
        # May return 500 if DB is not available, but should NOT return 422
        assert response.status_code != 422
