"""arXiv data loader – reads paper metadata from JSONL, JSON, NDJSON, or CSV.

Designed to be resilient: handles multiple field-name variants for dates,
authors, and categories, and skips malformed records gracefully.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Generator

from pipeline.models import PaperRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field-name aliases for common arXiv dataset variants
# ---------------------------------------------------------------------------
_DATE_FIELDS = ["update_date", "updated", "published", "versions"]
_ID_FIELDS = ["id", "arxiv_id", "paper_id"]
_TITLE_FIELDS = ["title"]
_ABSTRACT_FIELDS = ["abstract", "summary"]
_AUTHORS_FIELDS = ["authors", "authors_parsed"]
_CATEGORIES_FIELDS = ["categories", "category"]
_DOI_FIELDS = ["doi"]
_JOURNAL_FIELDS = ["journal-ref", "journal_ref"]


def _pick(row: dict, candidates: list[str], default: str = "") -> str:
    """Return the first non-empty value among candidate field names."""
    for key in candidates:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def _parse_date(raw: str) -> date | None:
    """Best-effort date parsing from various arXiv date formats."""
    if not raw:
        return None
    # Handle versions list: [{"created": "Mon, 1 Jan 2024 ..."}]
    if raw.startswith("["):
        try:
            versions = json.loads(raw.replace("'", '"'))
            if versions and isinstance(versions, list):
                raw = versions[-1].get("created", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt).date()
        except ValueError:
            continue
    # Try dateutil-style fallback
    try:
        return datetime.fromisoformat(raw[:19]).date()
    except ValueError:
        pass
    return None


def _parse_authors(raw: str) -> str:
    """Normalize authors field (handles stringified lists)."""
    if raw.startswith("["):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], list):
                    # authors_parsed format: [["Last", "First", ""], ...]
                    return ", ".join(
                        " ".join(part for part in parts if part).strip()
                        for parts in parsed
                    )
                return ", ".join(str(a) for a in parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    return raw


def _row_to_paper(row: dict) -> PaperRecord | None:
    """Convert a raw dict row into a PaperRecord, or None if unusable."""
    paper_id = _pick(row, _ID_FIELDS)
    title = _pick(row, _TITLE_FIELDS)
    if not paper_id or not title:
        return None

    date_raw = _pick(row, _DATE_FIELDS)
    published = _parse_date(date_raw)
    authors_raw = _pick(row, _AUTHORS_FIELDS)

    return PaperRecord(
        id=paper_id,
        title=title,
        abstract=_pick(row, _ABSTRACT_FIELDS),
        authors=_parse_authors(authors_raw),
        categories=_pick(row, _CATEGORIES_FIELDS),
        published_date=published,
        doi=_pick(row, _DOI_FIELDS) or None,
        journal_ref=_pick(row, _JOURNAL_FIELDS) or None,
        full_text_path=row.get("full_text_path") or None,
    )


def _iter_jsonl(path: Path) -> Generator[dict, None, None]:
    """Stream records from a JSONL / NDJSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", lineno, path)


def _iter_json(path: Path) -> Generator[dict, None, None]:
    """Load a full JSON array file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        yield from data
    else:
        yield data


def _iter_csv(path: Path) -> Generator[dict, None, None]:
    """Stream records from a CSV file."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        yield from reader


def load_arxiv_papers(path: str | Path) -> list[PaperRecord]:
    """Load arXiv papers from a local file.

    Supports JSONL, NDJSON, JSON, and CSV based on file extension.

    Args:
        path: Path to the data file.

    Returns:
        List of PaperRecord objects.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        iterator = _iter_jsonl(path)
    elif suffix == ".json":
        iterator = _iter_json(path)
    elif suffix == ".csv":
        iterator = _iter_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .jsonl, .ndjson, .json, or .csv")

    papers: list[PaperRecord] = []
    skipped = 0
    for row in iterator:
        paper = _row_to_paper(row)
        if paper:
            papers.append(paper)
        else:
            skipped += 1

    logger.info("Loaded %d papers from %s (skipped %d malformed rows)", len(papers), path, skipped)
    return papers
