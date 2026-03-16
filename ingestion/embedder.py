"""Embedding module – encodes document texts into dense vectors.

Uses sentence-transformers/all-MiniLM-L6-v2 (dimension 384).
Batches encoding for efficiency and stores embeddings on DocumentRecords.
"""

from __future__ import annotations

import logging
from typing import Optional

from pipeline.models import DocumentRecord

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
    return _model


def embed_texts(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts using the MiniLM model.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to encode per batch.

    Returns:
        List of embedding vectors (each a list of 384 floats).
    """
    model = _get_model()
    logger.info("Embedding %d texts in batches of %d", len(texts), batch_size)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query string.

    Args:
        query: The search query text.

    Returns:
        Embedding vector (list of 384 floats).
    """
    model = _get_model()
    embedding = model.encode([query])[0]
    return embedding.tolist()


def embed_documents(
    documents: list[DocumentRecord], batch_size: int = DEFAULT_BATCH_SIZE
) -> list[DocumentRecord]:
    """Compute and attach embeddings to a list of DocumentRecords.

    Args:
        documents: List of DocumentRecord objects.
        batch_size: Encoding batch size.

    Returns:
        The same documents with embedding field populated.
    """
    texts = [doc.content for doc in documents]
    embeddings = embed_texts(texts, batch_size=batch_size)

    for doc, emb in zip(documents, embeddings):
        doc.embedding = emb

    logger.info("Embedded %d documents (dim=%d)", len(documents), EMBEDDING_DIM)
    return documents
