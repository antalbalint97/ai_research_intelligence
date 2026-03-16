"""Topic taxonomy and keyword-based topic mapper for AI research papers.

Maps arXiv papers to a predefined topic taxonomy using heuristic keyword
matching over title + abstract. Each paper receives a primary topic,
optional secondary topics, and a brief reason for the assignment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic taxonomy: English label -> Hungarian translation
# ---------------------------------------------------------------------------
TOPIC_TAXONOMY: dict[str, str] = {
    "Large Language Models": "Nagy nyelvi modellek",
    "Multimodal AI": "Multimodális mesterséges intelligencia",
    "AI Agents": "AI ügynökök / autonóm ügynökök",
    "Retrieval-Augmented Generation": "Visszakereséssel bővített generálás",
    "Reinforcement Learning": "Megerősítéses tanulás",
    "Graph Neural Networks": "Gráf neurális hálók",
    "AI for Healthcare": "AI az egészségügyben",
    "AI for Robotics": "AI a robotikában",
    "AI Safety / Alignment": "AI biztonság és igazítás",
    "Efficient AI / Model Compression": "Hatékony AI / modellkompresszió",
    "Synthetic Data": "Szintetikus adatgenerálás",
    "Foundation Models": "Alapmodellek",
}

# ---------------------------------------------------------------------------
# Keyword rules: topic -> list of keyword patterns
# Order matters – first strong match wins as primary topic.
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Large Language Models": [
        r"\bllm\b",
        r"\bllms\b",
        r"large language model",
        r"language model",
        r"\bgpt\b",
        r"\bchatgpt\b",
        r"instruction[- ]?tuning",
        r"in[- ]?context learning",
        r"\brlhf\b",
        r"chain[- ]?of[- ]?thought",
    ],
    "Multimodal AI": [
        r"multimodal",
        r"vision[- ]?language",
        r"image[- ]?text",
        r"text[- ]?to[- ]?image",
        r"visual question answering",
        r"\bvlm\b",
        r"cross[- ]?modal",
    ],
    "AI Agents": [
        r"\bagent\b",
        r"\bagents\b",
        r"autonomous agent",
        r"tool[- ]?use",
        r"tool[- ]?augmented",
        r"agentic",
        r"multi[- ]?agent",
    ],
    "Retrieval-Augmented Generation": [
        r"\brag\b",
        r"retrieval[- ]?augmented",
        r"retrieval[- ]?enhanced",
        r"grounded generation",
    ],
    "Reinforcement Learning": [
        r"reinforcement learning",
        r"\brl\b",
        r"reward model",
        r"policy gradient",
        r"q[- ]?learning",
        r"multi[- ]?arm",
        r"bandit",
    ],
    "Graph Neural Networks": [
        r"graph neural",
        r"\bgnn\b",
        r"\bgcn\b",
        r"graph transformer",
        r"knowledge graph",
        r"graph representation",
    ],
    "AI for Healthcare": [
        r"healthcare",
        r"medical",
        r"clinical",
        r"biomedical",
        r"drug discovery",
        r"electronic health",
        r"pathology",
        r"radiology",
        r"diagnosis",
    ],
    "AI for Robotics": [
        r"robot",
        r"robotics",
        r"manipulation",
        r"locomotion",
        r"sim[- ]?to[- ]?real",
        r"embodied",
        r"navigation",
    ],
    "AI Safety / Alignment": [
        r"alignment",
        r"safety",
        r"jailbreak",
        r"red[- ]?team",
        r"harmfulness",
        r"toxicity",
        r"guardrail",
        r"interpretability",
        r"explainability",
    ],
    "Efficient AI / Model Compression": [
        r"quantization",
        r"pruning",
        r"distillation",
        r"model compression",
        r"efficient",
        r"low[- ]?rank",
        r"\blora\b",
        r"\bqlora\b",
        r"sparse",
    ],
    "Synthetic Data": [
        r"synthetic data",
        r"data augmentation",
        r"data generation",
        r"self[- ]?instruct",
    ],
    "Foundation Models": [
        r"foundation model",
        r"pretrain",
        r"pre[- ]?train",
        r"self[- ]?supervised",
        r"contrastive learning",
        r"masked language",
        r"bert",
        r"transformer architecture",
    ],
}


@dataclass
class TopicAssignment:
    """Result of topic mapping for a single paper."""

    primary_topic: str = ""
    secondary_topics: list[str] = field(default_factory=list)
    topic_reason: str = ""


def assign_topics(title: str, abstract: str) -> TopicAssignment:
    """Assign AI topics to a paper based on keyword matching over title + abstract.

    Args:
        title: Paper title.
        abstract: Paper abstract.

    Returns:
        TopicAssignment with primary_topic, secondary_topics, and topic_reason.
    """
    text = f"{title} {abstract}".lower()
    matched: list[tuple[str, str]] = []  # (topic, matched_keyword)

    for topic, patterns in TOPIC_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.append((topic, pattern))
                break  # one match per topic is enough

    if not matched:
        return TopicAssignment(
            primary_topic="Other",
            secondary_topics=[],
            topic_reason="No keyword match found in title or abstract.",
        )

    primary = matched[0][0]
    secondary = [t for t, _ in matched[1:] if t != primary]

    reasons = [f"'{kw}' matched for {t}" for t, kw in matched]
    reason_str = "; ".join(reasons)

    return TopicAssignment(
        primary_topic=primary,
        secondary_topics=secondary,
        topic_reason=reason_str,
    )


def get_all_topics() -> list[dict[str, str]]:
    """Return the full topic taxonomy as a list of dicts.

    Returns:
        List of {\"topic\": ..., \"translation_hu\": ...} dicts.
    """
    return [
        {"topic": topic, "translation_hu": hu} for topic, hu in TOPIC_TAXONOMY.items()
    ]
