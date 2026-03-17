#!/usr/bin/env python3
"""
build_ai_dataset.py

Build a recent AI-focused arXiv abstract dataset for RAG experiments.

Input:
- Raw Kaggle arXiv metadata JSONL
  OR
- A pre-filtered snapshot JSONL created by build_recent_ai_snapshot.py

Output:
- JSONL file with normalized, topic-labeled recent AI paper records

Example:
    python build_ai_dataset.py \
        --input ../data/processed/arxiv_ai_recent_sorted.jsonl \
        --output ../data/processed/arxiv_ai_curated.jsonl \
        --months 12

Notes:
- Uses title + abstract as the MVP corpus.
- Prefers published_date for trend filtering.
- Topic assignment is heuristic but more conservative than the previous version.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("build_ai_dataset")

AI_CATEGORIES = {
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "cs.RO",
    "cs.HC",
    "stat.ML",
}

DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %Z",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
]

TOPIC_TAXONOMY: dict[str, dict[str, Any]] = {
    "Large Language Models": {
        "hu": "Nagy nyelvi modellek",
        "keywords": [
            "large language model",
            "large language models",
            "llm",
            "instruction tuning",
            "instruction-tuned",
            "in-context learning",
            "language model reasoning",
            "reasoning language model",
        ],
        "weight": 4,
    },
    "Multimodal AI": {
        "hu": "Multimodális mesterséges intelligencia",
        "keywords": [
            "multimodal",
            "multi-modal",
            "vision-language",
            "vision language",
            "text-image",
            "image-text",
            "video-language",
            "cross-modal",
            "audio-visual",
        ],
        "weight": 4,
    },
    "AI Agents": {
        "hu": "AI ügynökök / autonóm ügynökök",
        "keywords": [
            "autonomous agent",
            "autonomous agents",
            "multi-agent",
            "multi agent",
            "agentic",
            "tool use",
            "tool-using",
            "planning with llm",
            "llm agent",
            "llm agents",
        ],
        "weight": 4,
    },
    "Retrieval-Augmented Generation": {
        "hu": "Visszakereséssel bővített generálás",
        "keywords": [
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "rag system",
            "rag framework",
            "grounded generation",
            "knowledge-grounded generation",
            "retrieval augmented",
            "retrieval-augmented",
            "grounded generation",
            "knowledge grounding",
            "external knowledge",
            "retrieval enhanced",
            "retrieval-based"
        ],
        "weight": 5,
    },
    "Reinforcement Learning": {
        "hu": "Megerősítéses tanulás",
        "keywords": [
            "reinforcement learning",
            "policy optimization",
            "policy gradient",
            "q-learning",
            "offline rl",
            "online rl",
            "reward model",
            "rlhf",
        ],
        "weight": 4,
    },
    "Graph Neural Networks": {
        "hu": "Gráf neurális hálók",
        "keywords": [
            "graph neural network",
            "graph neural networks",
            "graph transformer",
            "graph learning",
            "node classification",
            "link prediction",
        ],
        "weight": 4,
    },
    "AI for Healthcare": {
        "hu": "AI az egészségügyben",
        "keywords": [
            "healthcare",
            "medical ai",
            "clinical",
            "diagnosis",
            "biomedical",
            "medical imaging",
            "electronic health record",
            "ehr",
            "patient outcome",
        ],
        "weight": 4,
    },
    "AI for Robotics": {
        "hu": "AI a robotikában",
        "keywords": [
            "robotics",
            "robot learning",
            "embodied ai",
            "embodied agent",
            "robot manipulation",
            "locomotion",
            "robot control",
        ],
        "weight": 4,
    },
    "AI Safety / Alignment": {
        "hu": "AI biztonság és igazítás",
        "keywords": [
            "alignment",
            "ai safety",
            "safe ai",
            "controllability",
            "oversight",
            "jailbreak",
            "red teaming",
            "alignment",
            "ai safety",
            "safe ai",
            "jailbreak",
            "red teaming",
            "alignment training"
        ],
        "weight": 4,
    },
    "Efficient AI / Model Compression": {
        "hu": "Hatékony AI / modellkompresszió",
        "keywords": [
            "model compression",
            "quantization",
            "distillation",
            "pruning",
            "efficient inference",
            "parameter efficient",
            "parameter-efficient",
            "low-rank adaptation",
            "lora",
        ],
        "weight": 4,
    },
    "Synthetic Data": {
        "hu": "Szintetikus adatgenerálás",
        "keywords": [
            "synthetic data",
            "data synthesis",
            "generated data",
            "simulated data",
            "synthetic dataset",
        ],
        "weight": 3,
    },
    "Foundation Models": {
        "hu": "Alapmodellek",
        "keywords": [
            "foundation model",
            "foundation models",
            "pretrained model",
            "pre-trained model",
            "general-purpose model",
            "scaling law",
            "model scaling",
        ],
        "weight": 2,
    },
}

CATEGORY_TOPIC_HINTS: dict[str, tuple[str, int]] = {
    "cs.CL": ("Large Language Models", 1),
    "cs.CV": ("Multimodal AI", 1),
    "cs.RO": ("AI for Robotics", 2),
    "cs.HC": ("AI for Healthcare", 2),
    "cs.AI": ("Foundation Models", 1),
    "cs.LG": ("Foundation Models", 1),
    "stat.ML": ("Foundation Models", 1),
}


@dataclass
class ProcessedPaper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_topic_en: str
    primary_topic_hu: str
    secondary_topics_en: list[str]
    secondary_topics_hu: list[str]
    topic_reason: str
    published_date: str
    updated_date: str
    arxiv_abs_url: str
    arxiv_pdf_url: str
    source: str
    doc_type: str
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recent AI-focused arXiv dataset")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Lookback window in months (default: 12)",
    )
    parser.add_argument(
        "--min-abstract-length",
        type=int,
        default=200,
        help="Minimum abstract character length (default: 200)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional hard limit on written records (0 means no limit)",
    )
    parser.add_argument(
        "--skip-date-filter",
        action="store_true",
        help="Skip the date filter if the input is already a recent curated snapshot",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def month_lookback_cutoff(months: int) -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(microsecond=0) - timedelta(days=months * 30)


def parse_arxiv_date(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    return None


def extract_dates(record: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    # Snapshot mode support
    snapshot_published = parse_arxiv_date(record.get("published_date"))
    snapshot_updated = parse_arxiv_date(record.get("updated_date"))
    if snapshot_published or snapshot_updated:
        return snapshot_published, snapshot_updated or snapshot_published

    versions = record.get("versions") or []

    published_dt = None
    updated_dt = None

    if isinstance(versions, list) and versions:
        first = versions[0]
        last = versions[-1]
        if isinstance(first, dict):
            published_dt = parse_arxiv_date(first.get("created"))
        if isinstance(last, dict):
            updated_dt = parse_arxiv_date(last.get("created"))

    if not published_dt:
        published_dt = parse_arxiv_date(record.get("published")) or parse_arxiv_date(
            record.get("update_date")
        )

    if not updated_dt:
        updated_dt = parse_arxiv_date(record.get("update_date")) or published_dt

    return published_dt, updated_dt


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_title(title: str) -> str:
    title = normalize_whitespace(title)
    return re.sub(r"\s+", " ", title).strip()


def normalize_abstract(text: str) -> str:
    text = normalize_whitespace(text)
    # Lightweight cleanup for common junk
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    return text.strip()


def parse_authors(raw_authors: Any) -> list[str]:
    if raw_authors is None:
        return []

    if isinstance(raw_authors, list):
        names = []
        for item in raw_authors:
            if isinstance(item, str):
                names.append(normalize_whitespace(item))
            elif isinstance(item, dict):
                name = item.get("name") or item.get("author")
                if isinstance(name, str):
                    names.append(normalize_whitespace(name))
        return [x for x in names if x]

    if isinstance(raw_authors, str):
        raw_authors = raw_authors.strip()
        if not raw_authors:
            return []

        parts = re.split(r",\s*| and ", raw_authors)
        parts = [normalize_whitespace(p) for p in parts]
        return [p for p in parts if p]

    return []


def parse_categories(raw_categories: Any) -> list[str]:
    if raw_categories is None:
        return []

    if isinstance(raw_categories, list):
        return [normalize_whitespace(str(x)) for x in raw_categories if str(x).strip()]

    if isinstance(raw_categories, str):
        return [c.strip() for c in raw_categories.split() if c.strip()]

    return []


def is_ai_relevant(categories: list[str], title: str, abstract: str) -> bool:
    category_match = bool(set(categories) & AI_CATEGORIES)
    if category_match:
        return True

    text = f"{title} {abstract}".lower()
    fallback_keywords = [
        "large language model",
        "multimodal",
        "computer vision",
        "machine learning",
        "deep learning",
        "reinforcement learning",
        "robotics",
        "graph neural network",
        "alignment",
        "foundation model",
        "retrieval-augmented generation",
    ]
    return any(keyword in text for keyword in fallback_keywords)


def add_keyword_scores(
    text: str,
    scores: Counter[str],
    reasons: list[str],
) -> None:
    for topic_en, cfg in TOPIC_TAXONOMY.items():
        weight = int(cfg.get("weight", 1))
        for kw in cfg["keywords"]:
            if kw in text:
                scores[topic_en] += weight
                reasons.append(f"{topic_en}: keyword match '{kw}' (+{weight})")


def add_category_scores(
    categories: list[str],
    scores: Counter[str],
    reasons: list[str],
) -> None:
    category_set = set(categories)
    for cat, (topic, value) in CATEGORY_TOPIC_HINTS.items():
        if cat in category_set:
            scores[topic] += value
            reasons.append(f"{topic}: category hint {cat} (+{value})")


def select_topics(scores: Counter[str]) -> tuple[str, list[str]]:
    if not scores:
        return "Foundation Models", []

    ranked = scores.most_common()

    # Strong anti-overuse rule:
    # do not let Foundation Models win if another topic has close or equal evidence.
    foundation_score = scores.get("Foundation Models", 0)
    stronger_non_foundation = [
        topic for topic, score in ranked
        if topic != "Foundation Models" and score >= max(2, foundation_score)
    ]
    if stronger_non_foundation:
        primary = stronger_non_foundation[0]
    else:
        primary = ranked[0][0]

    secondary = [topic for topic, _ in ranked if topic != primary][:2]
    return primary, secondary


def match_topics(title: str, abstract: str, categories: list[str]) -> tuple[str, list[str], str]:
    text = f"{title}\n{abstract}".lower()
    scores: Counter[str] = Counter()
    reasons: list[str] = []

    add_keyword_scores(text=text, scores=scores, reasons=reasons)
    add_category_scores(categories=categories, scores=scores, reasons=reasons)

    # Conservative fallback only
    if not scores:
        return "Foundation Models", [], "Fallback default topic for AI paper"

    primary, secondary = select_topics(scores)
    topic_reason = "; ".join(reasons[:10]) if reasons else "keyword/category heuristic"
    return primary, secondary, topic_reason


def derive_arxiv_urls(record: dict[str, Any], arxiv_id: str) -> tuple[str, str]:
    abs_url = normalize_whitespace(record.get("arxiv_abs_url", ""))
    pdf_url = normalize_whitespace(record.get("arxiv_pdf_url", ""))

    if abs_url and pdf_url:
        return abs_url, pdf_url

    safe_id = arxiv_id.strip()
    return f"https://arxiv.org/abs/{safe_id}", f"https://arxiv.org/pdf/{safe_id}.pdf"


def build_content(
    title: str,
    abstract: str,
    authors: list[str],
    categories: list[str],
    published_date: str,
    primary_topic_en: str,
    secondary_topics_en: list[str],
) -> str:
    secondary = ", ".join(secondary_topics_en) if secondary_topics_en else "None"
    author_text = ", ".join(authors[:12]) if authors else "Unknown"
    category_text = ", ".join(categories) if categories else "Unknown"

    return (
        f"Title: {title}\n\n"
        f"Published: {published_date}\n"
        f"Categories: {category_text}\n"
        f"Primary Topic: {primary_topic_en}\n"
        f"Secondary Topics: {secondary}\n"
        f"Authors: {author_text}\n\n"
        f"Abstract:\n{abstract}"
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON on line %d: %s", line_number, exc)


def process_record(
    record: dict[str, Any],
    cutoff: datetime,
    min_abstract_length: int,
    skip_date_filter: bool,
) -> ProcessedPaper | None:
    paper_id = normalize_whitespace(str(record.get("paper_id") or record.get("id") or ""))
    title = normalize_title(record.get("title", ""))
    abstract = normalize_abstract(record.get("abstract", ""))
    authors = parse_authors(record.get("authors"))
    categories = parse_categories(record.get("categories"))

    if not paper_id or not title or not abstract:
        return None

    if len(abstract) < min_abstract_length:
        return None

    published_dt, updated_dt = extract_dates(record)
    if published_dt is None:
        return None

    # Prefer published date for trend windows
    if not skip_date_filter and published_dt < cutoff:
        return None

    if not is_ai_relevant(categories, title, abstract):
        return None

    primary_topic_en, secondary_topics_en, topic_reason = match_topics(title, abstract, categories)
    primary_topic_hu = TOPIC_TAXONOMY[primary_topic_en]["hu"]
    secondary_topics_hu = [TOPIC_TAXONOMY[t]["hu"] for t in secondary_topics_en]

    abs_url, pdf_url = derive_arxiv_urls(record, paper_id)

    published_date = published_dt.date().isoformat()
    updated_date = updated_dt.date().isoformat() if updated_dt else published_date

    content = build_content(
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        published_date=published_date,
        primary_topic_en=primary_topic_en,
        secondary_topics_en=secondary_topics_en,
    )

    return ProcessedPaper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_topic_en=primary_topic_en,
        primary_topic_hu=primary_topic_hu,
        secondary_topics_en=secondary_topics_en,
        secondary_topics_hu=secondary_topics_hu,
        topic_reason=topic_reason,
        published_date=published_date,
        updated_date=updated_date,
        arxiv_abs_url=abs_url,
        arxiv_pdf_url=pdf_url,
        source="arxiv",
        doc_type="paper_abstract",
        content=content,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cutoff = month_lookback_cutoff(args.months)
    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Output: %s", output_path)
    LOGGER.info("Lookback cutoff: %s", cutoff.isoformat())
    LOGGER.info("AI categories: %s", sorted(AI_CATEGORIES))
    LOGGER.info("Skip date filter: %s", args.skip_date_filter)

    total = 0
    kept = 0
    topic_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in iter_jsonl(input_path):
            total += 1

            processed = process_record(
                record=record,
                cutoff=cutoff,
                min_abstract_length=args.min_abstract_length,
                skip_date_filter=args.skip_date_filter,
            )
            if processed is None:
                continue

            out_f.write(json.dumps(asdict(processed), ensure_ascii=False) + "\n")
            kept += 1

            topic_counter[processed.primary_topic_en] += 1
            for c in processed.categories:
                if c in AI_CATEGORIES:
                    category_counter[c] += 1

            if args.limit and kept >= args.limit:
                LOGGER.info("Reached output limit: %d", args.limit)
                break

            if total % 100000 == 0:
                LOGGER.info("Processed %d records, kept %d", total, kept)

    LOGGER.info("Done. Processed=%d, kept=%d", total, kept)
    LOGGER.info("Top topics: %s", topic_counter.most_common(12))
    LOGGER.info("Top AI categories: %s", category_counter.most_common(10))


if __name__ == "__main__":
    main()