"""Manually-authored evaluation test set for the AI Research RAG system.

Contains 25 business-style questions across AI topics for retrieval and
answer-quality evaluation. Each entry includes the query, expected topic,
and optional relevance hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestCase:
    """Single evaluation test case."""

    query: str
    expected_topic: str
    description: str = ""
    relevant_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation test set: 25 business-oriented AI research questions
# ---------------------------------------------------------------------------
EVAL_TEST_SET: list[TestCase] = [
    # Large Language Models
    TestCase(
        query="What are the latest advances in large language model reasoning capabilities?",
        expected_topic="Large Language Models",
        description="LLM reasoning – chain-of-thought, planning",
        relevant_keywords=["reasoning", "chain-of-thought", "LLM"],
    ),
    TestCase(
        query="How are companies reducing the cost of running large language models in production?",
        expected_topic="Large Language Models",
        description="LLM inference efficiency",
        relevant_keywords=["inference", "cost", "optimization"],
    ),
    TestCase(
        query="What instruction tuning methods are gaining traction in recent LLM research?",
        expected_topic="Large Language Models",
        description="Instruction tuning trends",
        relevant_keywords=["instruction", "tuning", "RLHF"],
    ),
    # Multimodal AI
    TestCase(
        query="What are the recent trends in multimodal AI research?",
        expected_topic="Multimodal AI",
        description="General multimodal trends",
        relevant_keywords=["multimodal", "vision-language", "image-text"],
    ),
    TestCase(
        query="How is vision-language model research evolving for commercial applications?",
        expected_topic="Multimodal AI",
        description="VLM commercial applications",
        relevant_keywords=["vision", "language", "VLM"],
    ),
    TestCase(
        query="What text-to-image generation methods are showing the most promise?",
        expected_topic="Multimodal AI",
        description="Text-to-image generation",
        relevant_keywords=["text-to-image", "diffusion", "generation"],
    ),
    # AI Agents
    TestCase(
        query="What methods are becoming popular in AI agents research?",
        expected_topic="AI Agents",
        description="Agent architectures and methods",
        relevant_keywords=["agent", "tool-use", "autonomous"],
    ),
    TestCase(
        query="How are multi-agent systems being applied in recent AI research?",
        expected_topic="AI Agents",
        description="Multi-agent systems",
        relevant_keywords=["multi-agent", "collaboration", "coordination"],
    ),
    TestCase(
        query="What are the commercially promising directions for AI agent platforms?",
        expected_topic="AI Agents",
        description="Agent commercialization",
        relevant_keywords=["agent", "platform", "tool"],
    ),
    # RAG
    TestCase(
        query="What are the limitations of current RAG research?",
        expected_topic="Retrieval-Augmented Generation",
        description="RAG limitations and challenges",
        relevant_keywords=["RAG", "retrieval", "limitation"],
    ),
    TestCase(
        query="What new retrieval strategies are being explored for RAG systems?",
        expected_topic="Retrieval-Augmented Generation",
        description="Novel retrieval for RAG",
        relevant_keywords=["retrieval", "RAG", "dense"],
    ),
    # AI for Healthcare
    TestCase(
        query="How is AI being used in healthcare research recently?",
        expected_topic="AI for Healthcare",
        description="AI healthcare applications",
        relevant_keywords=["healthcare", "medical", "clinical"],
    ),
    TestCase(
        query="What machine learning approaches are emerging for drug discovery?",
        expected_topic="AI for Healthcare",
        description="ML drug discovery",
        relevant_keywords=["drug", "discovery", "molecular"],
    ),
    TestCase(
        query="How are foundation models being adapted for medical imaging?",
        expected_topic="AI for Healthcare",
        description="Medical imaging AI",
        relevant_keywords=["medical", "imaging", "foundation"],
    ),
    # AI for Robotics
    TestCase(
        query="What commercially promising directions are emerging in robotics AI?",
        expected_topic="AI for Robotics",
        description="Robotics commercial AI",
        relevant_keywords=["robot", "manipulation", "embodied"],
    ),
    TestCase(
        query="How are sim-to-real transfer methods improving in robotics research?",
        expected_topic="AI for Robotics",
        description="Sim-to-real robotics",
        relevant_keywords=["sim-to-real", "robot", "transfer"],
    ),
    # AI Safety / Alignment
    TestCase(
        query="What are the current research priorities in AI safety and alignment?",
        expected_topic="AI Safety / Alignment",
        description="AI safety priorities",
        relevant_keywords=["safety", "alignment", "interpretability"],
    ),
    TestCase(
        query="How effective are current jailbreak defense methods for LLMs?",
        expected_topic="AI Safety / Alignment",
        description="Jailbreak defenses",
        relevant_keywords=["jailbreak", "defense", "safety"],
    ),
    TestCase(
        query="What red-teaming approaches are being used in AI safety research?",
        expected_topic="AI Safety / Alignment",
        description="Red-teaming AI",
        relevant_keywords=["red-team", "adversarial", "safety"],
    ),
    # Reinforcement Learning
    TestCase(
        query="What recent breakthroughs have occurred in reinforcement learning?",
        expected_topic="Reinforcement Learning",
        description="RL breakthroughs",
        relevant_keywords=["reinforcement", "reward", "policy"],
    ),
    TestCase(
        query="How is RL being combined with large language models?",
        expected_topic="Reinforcement Learning",
        description="RL + LLM integration",
        relevant_keywords=["reinforcement", "RLHF", "reward"],
    ),
    # Efficient AI / Model Compression
    TestCase(
        query="What quantization techniques are most effective for deploying LLMs on edge devices?",
        expected_topic="Efficient AI / Model Compression",
        description="Quantization for edge",
        relevant_keywords=["quantization", "edge", "compression"],
    ),
    TestCase(
        query="How are LoRA and parameter-efficient fine-tuning methods evolving?",
        expected_topic="Efficient AI / Model Compression",
        description="PEFT / LoRA trends",
        relevant_keywords=["LoRA", "parameter-efficient", "fine-tuning"],
    ),
    # Graph Neural Networks
    TestCase(
        query="What are the latest trends in graph neural network research?",
        expected_topic="Graph Neural Networks",
        description="GNN trends",
        relevant_keywords=["graph", "GNN", "knowledge graph"],
    ),
    # Synthetic Data
    TestCase(
        query="How is synthetic data being used to improve AI model training?",
        expected_topic="Synthetic Data",
        description="Synthetic data for training",
        relevant_keywords=["synthetic", "data generation", "augmentation"],
    ),
]


def get_eval_test_set() -> list[TestCase]:
    """Return the full evaluation test set."""
    return EVAL_TEST_SET
