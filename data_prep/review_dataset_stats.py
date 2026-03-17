#!/usr/bin/env python3
"""
review_dataset_stats.py

Audit and inspect a curated AI arXiv dataset for topic quality.

What it does:
- reads curated JSONL records
- prints topic distribution
- prints category distribution
- shows sample papers per topic
- optionally exports a CSV for manual review

Example:
    python review_dataset_stats.py \
        --input ../data/processed/arxiv_ai_curated.jsonl \
        --samples-per-topic 5

CSV export example:
    python review_dataset_stats.py \
        --input ../data/processed/arxiv_ai_curated.jsonl \
        --samples-per-topic 5 \
        --export-csv ../data/processed/arxiv_ai_curated_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("review_dataset_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review curated AI dataset statistics")
    parser.add_argument("--input", required=True, help="Path to curated JSONL file")
    parser.add_argument(
        "--samples-per-topic",
        type=int,
        default=5,
        help="Number of sample rows to display per topic (default: 5)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=20,
        help="Maximum number of topics to print in summary (default: 20)",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Maximum number of categories to print in summary (default: 20)",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="Optional path to export audit CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON on line %d: %s", line_number, exc)


def truncate(text: str, max_len: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def export_audit_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "paper_id",
        "published_date",
        "primary_topic_en",
        "primary_topic_hu",
        "secondary_topics_en",
        "categories",
        "title",
        "topic_reason",
        "abstract_preview",
        "arxiv_abs_url",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "paper_id": row.get("paper_id", ""),
                    "published_date": row.get("published_date", ""),
                    "primary_topic_en": row.get("primary_topic_en", ""),
                    "primary_topic_hu": row.get("primary_topic_hu", ""),
                    "secondary_topics_en": ", ".join(row.get("secondary_topics_en", [])),
                    "categories": ", ".join(row.get("categories", [])),
                    "title": row.get("title", ""),
                    "topic_reason": row.get("topic_reason", ""),
                    "abstract_preview": truncate(row.get("abstract", ""), 500),
                    "arxiv_abs_url": row.get("arxiv_abs_url", ""),
                }
            )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: list[dict[str, Any]] = list(iter_jsonl(input_path))
    total = len(records)

    if total == 0:
        LOGGER.warning("No records found in input file.")
        return

    LOGGER.info("Loaded %d records from %s", total, input_path)

    topic_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for rec in records:
        topic = rec.get("primary_topic_en", "UNKNOWN")
        topic_counter[topic] += 1
        topic_groups[topic].append(rec)

        for cat in rec.get("categories", []):
            category_counter[cat] += 1

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total records: {total}")
    print()

    print("Top topics:")
    for topic, count in topic_counter.most_common(args.max_topics):
        pct = (count / total) * 100
        print(f"  - {topic}: {count} ({pct:.2f}%)")

    print()
    print("Top categories:")
    for cat, count in category_counter.most_common(args.max_categories):
        pct = (count / total) * 100
        print(f"  - {cat}: {count} ({pct:.2f}%)")

    print("\n" + "=" * 80)
    print("TOPIC SAMPLES")
    print("=" * 80)

    audit_rows: list[dict[str, Any]] = []

    for topic, count in topic_counter.most_common():
        print(f"\n### {topic} ({count})")
        group = topic_groups[topic]

        if len(group) <= args.samples_per_topic:
            sample_rows = group
        else:
            sample_rows = random.sample(group, args.samples_per_topic)

        sample_rows = sorted(
            sample_rows,
            key=lambda x: x.get("published_date", ""),
            reverse=True,
        )

        for idx, row in enumerate(sample_rows, start=1):
            title = row.get("title", "")
            published_date = row.get("published_date", "")
            categories = ", ".join(row.get("categories", []))
            secondary = ", ".join(row.get("secondary_topics_en", []))
            topic_reason = row.get("topic_reason", "")
            abstract_preview = truncate(row.get("abstract", ""), 250)

            print(f"\n  [{idx}] {title}")
            print(f"      published: {published_date}")
            print(f"      categories: {categories}")
            print(f"      secondary topics: {secondary if secondary else 'None'}")
            print(f"      reason: {truncate(topic_reason, 180)}")
            print(f"      abstract: {abstract_preview}")
            print(f"      url: {row.get('arxiv_abs_url', '')}")

            audit_rows.append(row)

    if args.export_csv:
        export_path = Path(args.export_csv)
        export_audit_csv(audit_rows, export_path)
        LOGGER.info("Audit CSV written to %s", export_path)


if __name__ == "__main__":
    main()