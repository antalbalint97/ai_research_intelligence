#!/usr/bin/env python3
"""
build_faiss_index.py

Build a FAISS vector index from the curated AI arXiv dataset.

Designed for abstract-level RAG:
- each record in arxiv_ai_curated.jsonl is one retrieval unit
- embeds the `content` field by default
- stores a FAISS index + aligned metadata JSONL

Example:
    python build_faiss_index.py \
        --input ../data/processed/arxiv_ai_curated.jsonl \
        --index-output ../data/index/arxiv_ai.index \
        --metadata-output ../data/index/arxiv_ai_metadata.jsonl

Why this design:
- build_ai_dataset.py already creates a normalized RAG-ready content field
- abstracts are short enough that chunking is not necessary for this project
- IndexFlatIP + normalized vectors gives cosine similarity behavior
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger("build_faiss_index")

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_TEXT_FIELD = "content"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from curated AI dataset")
    parser.add_argument("--input", required=True, help="Path to curated JSONL dataset")
    parser.add_argument("--index-output", required=True, help="Path to output FAISS index file")
    parser.add_argument(
        "--metadata-output",
        required=True,
        help="Path to output metadata JSONL aligned with FAISS row ids",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--text-field",
        default=DEFAULT_TEXT_FIELD,
        help=f"Field to embed from each JSONL record (default: {DEFAULT_TEXT_FIELD})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional hard limit on number of indexed records (default: unlimited)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize embeddings before indexing (default: True)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalization",
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


def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc

    LOGGER.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def make_metadata_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only the fields needed for retrieval/result rendering.
    Row order in this file must match FAISS row ids.
    """
    return {
        "paper_id": record.get("paper_id", ""),
        "title": record.get("title", ""),
        "abstract": record.get("abstract", ""),
        "authors": record.get("authors", []),
        "categories": record.get("categories", []),
        "primary_topic_en": record.get("primary_topic_en", ""),
        "primary_topic_hu": record.get("primary_topic_hu", ""),
        "secondary_topics_en": record.get("secondary_topics_en", []),
        "secondary_topics_hu": record.get("secondary_topics_hu", []),
        "topic_reason": record.get("topic_reason", ""),
        "published_date": record.get("published_date", ""),
        "updated_date": record.get("updated_date", ""),
        "arxiv_abs_url": record.get("arxiv_abs_url", ""),
        "arxiv_pdf_url": record.get("arxiv_pdf_url", ""),
        "source": record.get("source", "arxiv"),
        "doc_type": record.get("doc_type", "paper_abstract"),
        "content": record.get("content", ""),
    }


def batched(iterable: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(
    input_path: Path,
    index_output_path: Path,
    metadata_output_path: Path,
    model_name: str,
    text_field: str,
    batch_size: int,
    limit: int,
    normalize: bool,
) -> dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss is required. Install with: pip install faiss-cpu") from exc

    model = load_model(model_name)

    index = None
    dim = None
    total_indexed = 0
    total_seen = 0
    metadata_rows: list[dict[str, Any]] = []

    LOGGER.info("Input: %s", input_path)
    LOGGER.info("Index output: %s", index_output_path)
    LOGGER.info("Metadata output: %s", metadata_output_path)
    LOGGER.info("Text field: %s", text_field)
    LOGGER.info("Batch size: %d", batch_size)
    LOGGER.info("Normalize embeddings: %s", normalize)

    records_iter = iter_jsonl(input_path)

    if limit and limit > 0:
        def limited_records():
            nonlocal total_seen
            for record in records_iter:
                if total_seen >= limit:
                    break
                total_seen += 1
                yield record
        stream = limited_records()
    else:
        def unlimited_records():
            nonlocal total_seen
            for record in records_iter:
                total_seen += 1
                yield record
        stream = unlimited_records()

    for batch_number, batch_records in enumerate(batched(stream, batch_size), start=1):
        texts: list[str] = []
        clean_records: list[dict[str, Any]] = []

        for record in batch_records:
            text = normalize_text(record.get(text_field, ""))
            if not text:
                continue
            texts.append(text)
            clean_records.append(record)

        if not texts:
            continue

        batch_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)

        if batch_embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding array, got shape {batch_embeddings.shape}")

        if normalize:
            faiss.normalize_L2(batch_embeddings)

        if index is None:
            dim = int(batch_embeddings.shape[1])
            index = faiss.IndexFlatIP(dim)
            LOGGER.info("Initialized FAISS IndexFlatIP with dim=%d", dim)

        index.add(batch_embeddings)
        total_indexed += batch_embeddings.shape[0]

        for record in clean_records:
            metadata_rows.append(make_metadata_record(record))

        if batch_number % 20 == 0 or total_indexed % 10000 < batch_embeddings.shape[0]:
            LOGGER.info(
                "Processed batches=%d, indexed=%d documents",
                batch_number,
                total_indexed,
            )

    if index is None or dim is None:
        raise ValueError("No valid records were indexed. Check input data and text field.")

    if len(metadata_rows) != total_indexed:
        raise RuntimeError(
            f"Metadata/index size mismatch: metadata_rows={len(metadata_rows)} indexed={total_indexed}"
        )

    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_output_path))
    LOGGER.info("FAISS index written to %s", index_output_path)

    with metadata_output_path.open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("Metadata written to %s", metadata_output_path)

    return {
        "indexed_documents": total_indexed,
        "embedding_dim": dim,
        "index_type": "IndexFlatIP",
        "normalized": normalize,
        "model_name": model_name,
        "text_field": text_field,
        "index_output": str(index_output_path),
        "metadata_output": str(metadata_output_path),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    input_path = Path(args.input)
    index_output_path = Path(args.index_output)
    metadata_output_path = Path(args.metadata_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    summary = build_index(
        input_path=input_path,
        index_output_path=index_output_path,
        metadata_output_path=metadata_output_path,
        model_name=args.model_name,
        text_field=args.text_field,
        batch_size=args.batch_size,
        limit=args.limit,
        normalize=args.normalize,
    )

    LOGGER.info("Done: %s", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()