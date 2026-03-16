"""Text chunker – splits long documents into smaller overlapping chunks.

Uses a simple sentence-aware sliding-window approach.
Each chunk retains a reference back to the parent document.
"""

from __future__ import annotations

import hashlib
import logging
import re

from pipeline.models import DocumentRecord

logger = logging.getLogger(__name__)

# Default chunk parameters
DEFAULT_CHUNK_SIZE = 512  # tokens ≈ chars / 4 → we use ~1500 chars
DEFAULT_CHUNK_OVERLAP = 128  # character overlap between consecutive chunks
CHAR_CHUNK_SIZE = 1500
CHAR_CHUNK_OVERLAP = 300


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex heuristics."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def chunk_document(
    doc: DocumentRecord,
    chunk_size: int = CHAR_CHUNK_SIZE,
    chunk_overlap: int = CHAR_CHUNK_OVERLAP,
) -> list[DocumentRecord]:
    """Split a document into smaller chunk records if it exceeds chunk_size.

    Short documents (below chunk_size) are returned as-is.

    Args:
        doc: Source DocumentRecord.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of DocumentRecord objects (original or chunked).
    """
    if len(doc.content) <= chunk_size:
        return [doc]

    sentences = _split_into_sentences(doc.content)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep last sentences that fit within overlap
            overlap_chunk: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_len += len(s)
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Build chunk records
    chunk_docs: list[DocumentRecord] = []
    for idx, chunk_text in enumerate(chunks):
        chunk_id = hashlib.sha256(f"{doc.doc_id}:chunk:{idx}".encode()).hexdigest()[:16]
        chunk_doc = doc.model_copy(
            update={
                "doc_id": chunk_id,
                "content": chunk_text,
                "doc_type": "paper_chunk",
                "metadata": {**doc.metadata, "chunk_index": idx, "parent_doc_id": doc.doc_id},
            }
        )
        chunk_docs.append(chunk_doc)

    logger.debug("Chunked document %s into %d chunks", doc.doc_id, len(chunk_docs))
    return chunk_docs


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int = CHAR_CHUNK_SIZE,
    chunk_overlap: int = CHAR_CHUNK_OVERLAP,
) -> list[DocumentRecord]:
    """Chunk a list of documents.

    Args:
        documents: List of DocumentRecord objects.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        Flat list of all document chunks.
    """
    all_chunks: list[DocumentRecord] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    logger.info(
        "Chunked %d documents into %d total chunks", len(documents), len(all_chunks)
    )
    return all_chunks
