#!/usr/bin/env python3
"""
build_recent_ai_snapshot.py

Create a smaller, curated arXiv snapshot for fast downstream experimentation.

What it does:
- reads the large arXiv metadata JSONL file in streaming mode
- keeps only recent papers (default: last 12 months)
- keeps only AI-relevant categories
- extracts a minimal normalized schema
- sorts kept records by published_date descending
- writes a compact JSONL snapshot

Example:
    python build_recent_ai_snapshot.py \
        --input ../data/raw/arxiv-metadata-oai-snapshot.json \
        --output ../data/processed/arxiv_ai_recent_sorted.jsonl \
        --months 12
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("build_recent_ai_snapshot")

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
    "%a, %d %b %Y %H:%M:%S %Z",  # Mon, 15 Jan 2024 12:00:00 GMT
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
]


@dataclass
class SnapshotRecord:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_date: str
    updated_date: str
    arxiv_abs_url: str
    arxiv_pdf_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recent AI arXiv snapshot")
    parser.add_argument("--input", required=True, help="Path to raw arXiv metadata JSONL file")
    parser.add_argument("--output", required=True, help="Path to output snapshot JSONL file")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Lookback window in months, approximated as 30 days/month (default: 12)",
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
        help="Optional max number of output records after filtering/sorting (default: unlimited)",
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


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def parse_authors(raw_authors: Any) -> list[str]:
    if raw_authors is None:
        return []

    if isinstance(raw_authors, list):
        result: list[str] = []
        for item in raw_authors:
            if isinstance(item, str):
                name = normalize_whitespace(item)
                if name:
                    result.append(name)
            elif isinstance(item, dict):
                name = normalize_whitespace(str(item.get("name") or item.get("author") or ""))
                if name:
                    result.append(name)
        return result

    if isinstance(raw_authors, str):
        raw = normalize_whitespace(raw_authors)
        if not raw:
            return []
        # Kaggle arXiv metadata commonly stores authors as a single string.
        parts = raw.replace(" and ", ", ").split(",")
        return [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]

    return []


def parse_categories(raw_categories: Any) -> list[str]:
    if raw_categories is None:
        return []

    if isinstance(raw_categories, list):
        return [normalize_whitespace(str(x)) for x in raw_categories if normalize_whitespace(str(x))]

    if isinstance(raw_categories, str):
        return [c.strip() for c in raw_categories.split() if c.strip()]

    return []


def parse_date(value: str | None) -> datetime | None:
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
    """
    Prefer the first version's created date as published_date,
    and the last version's created date as updated_date.
    """
    versions = record.get("versions") or []
    published_dt = None
    updated_dt = None

    if isinstance(versions, list) and versions:
        first = versions[0]
        last = versions[-1]

        if isinstance(first, dict):
            published_dt = parse_date(first.get("created"))
        if isinstance(last, dict):
            updated_dt = parse_date(last.get("created"))

    if not published_dt:
        published_dt = parse_date(record.get("published")) or parse_date(record.get("update_date"))

    if not updated_dt:
        updated_dt = parse_date(record.get("update_date")) or published_dt

    return published_dt, updated_dt


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


def build_urls(arxiv_id: str) -> tuple[str, str]:
    arxiv_id = arxiv_id.strip()
    return (
        f"https://arxiv.org/abs/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def keep_record(
    record: dict[str, Any],
    cutoff: datetime,
    min_abstract_length: int,
) -> SnapshotRecord | None:
    paper_id = normalize_whitespace(str(record.get("id", "")))
    title = normalize_whitespace(record.get("title", ""))
    abstract = normalize_whitespace(record.get("abstract", ""))
    authors = parse_authors(record.get("authors"))
    categories = parse_categories(record.get("categories"))

    if not paper_id or not title or not abstract:
        return None

    if len(abstract) < min_abstract_length:
        return None

    if not (set(categories) & AI_CATEGORIES):
        return None

    published_dt, updated_dt = extract_dates(record)
    if published_dt is None:
        return None

    # Trend analysis should use first publication date, not last update date.
    if published_dt < cutoff:
        return None

    abs_url, pdf_url = build_urls(paper_id)

    return SnapshotRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        published_date=published_dt.date().isoformat(),
        updated_date=updated_dt.date().isoformat() if updated_dt else published_dt.date().isoformat(),
        arxiv_abs_url=abs_url,
        arxiv_pdf_url=pdf_url,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.months * 30)

    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Output: %s", output_path)
    LOGGER.info("Cutoff date: %s", cutoff.date().isoformat())
    LOGGER.info("AI categories: %s", sorted(AI_CATEGORIES))

    total = 0
    kept_records: list[SnapshotRecord] = []

    for record in iter_jsonl(input_path):
        total += 1

        kept = keep_record(
            record=record,
            cutoff=cutoff,
            min_abstract_length=args.min_abstract_length,
        )
        if kept is not None:
            kept_records.append(kept)

        if total % 100000 == 0:
            LOGGER.info("Scanned %d records, kept %d", total, len(kept_records))

    LOGGER.info("Finished scanning raw file. Total=%d, kept=%d", total, len(kept_records))

    kept_records.sort(
        key=lambda x: (x.published_date, x.updated_date, x.paper_id),
        reverse=True,
    )

    if args.limit and args.limit > 0:
        kept_records = kept_records[: args.limit]
        LOGGER.info("Applied final output limit: %d", args.limit)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in kept_records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    LOGGER.info("Snapshot written: %s (%d records)", output_path, len(kept_records))


if __name__ == "__main__":
    main()