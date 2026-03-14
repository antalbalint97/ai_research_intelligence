"""Tests for the topic mapper module."""

from __future__ import annotations

import pytest

from ingestion.topic_mapper import (
    TOPIC_TAXONOMY,
    TopicAssignment,
    assign_topics,
    get_all_topics,
)


class TestAssignTopics:
    """Test topic assignment via keyword heuristics."""

    def test_llm_topic_from_title(self) -> None:
        result = assign_topics("A Survey of Large Language Models", "")
        assert result.primary_topic == "Large Language Models"

    def test_llm_topic_from_abstract(self) -> None:
        result = assign_topics("Scaling Laws", "We study LLMs and instruction tuning.")
        assert result.primary_topic == "Large Language Models"

    def test_multimodal_topic(self) -> None:
        result = assign_topics("Vision-Language Pretraining", "A multimodal approach.")
        assert result.primary_topic == "Multimodal AI"

    def test_rag_topic(self) -> None:
        result = assign_topics("Improving RAG Systems", "Retrieval-augmented generation.")
        assert result.primary_topic == "Retrieval-Augmented Generation"

    def test_healthcare_topic(self) -> None:
        result = assign_topics("AI for Clinical Diagnosis", "A medical imaging approach.")
        assert result.primary_topic == "AI for Healthcare"

    def test_robotics_topic(self) -> None:
        result = assign_topics("Robotic Manipulation", "Sim-to-real transfer for robot arms.")
        assert result.primary_topic == "AI for Robotics"

    def test_safety_topic(self) -> None:
        result = assign_topics("Red-Teaming LLMs", "Jailbreak attacks and defenses.")
        assert result.primary_topic == "AI Safety / Alignment"

    def test_agents_topic(self) -> None:
        result = assign_topics("Autonomous Agents for Tool Use", "Multi-agent systems.")
        assert result.primary_topic == "AI Agents"

    def test_no_match_returns_other(self) -> None:
        result = assign_topics("Cooking Recipes", "How to bake a cake")
        assert result.primary_topic == "Other"
        assert result.secondary_topics == []

    def test_secondary_topics(self) -> None:
        result = assign_topics(
            "LLM Agents for Healthcare",
            "We build autonomous agents powered by large language models for clinical use.",
        )
        assert result.primary_topic in ("Large Language Models", "AI Agents", "AI for Healthcare")
        assert len(result.secondary_topics) >= 1

    def test_topic_reason_populated(self) -> None:
        result = assign_topics("Graph Neural Networks for Molecules", "GNN-based approach.")
        assert result.primary_topic == "Graph Neural Networks"
        assert result.topic_reason  # non-empty


class TestGetAllTopics:
    """Test topic taxonomy retrieval."""

    def test_returns_all_topics(self) -> None:
        topics = get_all_topics()
        assert len(topics) == len(TOPIC_TAXONOMY)

    def test_topic_structure(self) -> None:
        topics = get_all_topics()
        for t in topics:
            assert "topic" in t
            assert "translation_hu" in t
            assert t["topic"]
            assert t["translation_hu"]
