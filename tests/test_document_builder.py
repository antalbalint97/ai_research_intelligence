"""Tests for the document builder module."""

from __future__ import annotations

from datetime import date

import pytest

from ingestion.build_documents import build_documents, _arxiv_url, _make_doc_id
from pipeline.models import PaperRecord


def _make_paper(**kwargs) -> PaperRecord:
    """Helper to create a PaperRecord with sensible defaults."""
    defaults = {
        "id": "2401.12345",
        "title": "Test Paper on Large Language Models",
        "abstract": "We study LLMs and their applications.",
        "authors": "Alice Smith, Bob Jones",
        "categories": "cs.CL cs.AI",
        "published_date": date(2025, 9, 15),
    }
    defaults.update(kwargs)
    return PaperRecord(**defaults)


class TestArxivUrl:
    """Test arXiv URL derivation."""

    def test_simple_id(self) -> None:
        assert _arxiv_url("2401.12345") == "https://arxiv.org/abs/2401.12345"

    def test_versioned_id(self) -> None:
        assert _arxiv_url("2401.12345v2") == "https://arxiv.org/abs/2401.12345"


class TestMakeDocId:
    """Test deterministic document ID generation."""

    def test_deterministic(self) -> None:
        id1 = _make_doc_id("2401.12345", "paper_abstract")
        id2 = _make_doc_id("2401.12345", "paper_abstract")
        assert id1 == id2

    def test_different_types(self) -> None:
        id1 = _make_doc_id("2401.12345", "paper_abstract")
        id2 = _make_doc_id("2401.12345", "paper_chunk")
        assert id1 != id2


class TestBuildDocuments:
    """Test the document building pipeline."""

    def test_builds_documents_from_papers(self) -> None:
        papers = [_make_paper()]
        docs = build_documents(papers)
        assert len(docs) == 1

    def test_document_fields(self) -> None:
        papers = [_make_paper()]
        doc = build_documents(papers)[0]
        assert doc.paper_id == "2401.12345"
        assert doc.source == "arxiv"
        assert doc.doc_type == "paper_abstract"
        assert doc.language == "en"
        assert "https://arxiv.org/abs/2401.12345" == doc.url

    def test_content_format(self) -> None:
        papers = [_make_paper()]
        doc = build_documents(papers)[0]
        assert "Title: Test Paper on Large Language Models" in doc.content
        assert "Abstract:" in doc.content
        assert "LLMs" in doc.content

    def test_topic_assignment_in_document(self) -> None:
        papers = [_make_paper()]
        doc = build_documents(papers)[0]
        assert doc.primary_topic == "Large Language Models"

    def test_empty_papers(self) -> None:
        docs = build_documents([])
        assert docs == []

    def test_multiple_papers(self) -> None:
        papers = [
            _make_paper(id="001", title="LLM Research", abstract="Language model study"),
            _make_paper(id="002", title="Robot Navigation", abstract="Robotics research"),
        ]
        docs = build_documents(papers)
        assert len(docs) == 2
        assert docs[0].paper_id != docs[1].paper_id
