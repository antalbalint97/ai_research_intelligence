"""Embedding module – encodes document texts into dense vectors.

Optimized for low-latency online query embedding:
- lazy singleton SentenceTransformer model
- optional startup warmup to avoid first-query penalty
- no progress bars in online code paths
- query embedding returns normalized float32 vectors
"""

from __future__ import annotations

import logging
import os

import numpy as np

from pipeline.models import DocumentRecord

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
EMBED_NORMALIZE = os.environ.get("EMBED_NORMALIZE", "true").lower() in ("1", "true", "yes")

_model = None
_model_warmed = False


def _get_model():
    """Lazy-load the SentenceTransformer model once per process."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model: %s | device=%s",
                EMBEDDING_MODEL_NAME,
                EMBED_DEVICE,
            )
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBED_DEVICE)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
    return _model


def warmup_embedder() -> None:
    """Force model load + one tiny encode to reduce first-query latency."""
    global _model_warmed
    model = _get_model()

    if _model_warmed:
        return

    logger.info("Warming up embedding model")
    _ = model.encode(
        ["warmup query"],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=EMBED_NORMALIZE,
    )
    _model_warmed = True
    logger.info("Embedding model warmup complete")


def embed_texts(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts using the MiniLM model."""
    if not texts:
        return []

    model = _get_model()
    logger.info("Embedding %d texts in batches of %d", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=EMBED_NORMALIZE,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = _get_model()
    embedding = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=EMBED_NORMALIZE,
    )[0]
    return np.asarray(embedding, dtype=np.float32).tolist()


def embed_documents(
    documents: list[DocumentRecord], batch_size: int = DEFAULT_BATCH_SIZE
) -> list[DocumentRecord]:
    """Compute and attach embeddings to a list of DocumentRecords."""
    if not documents:
        return documents

    texts = [doc.content for doc in documents]
    embeddings = embed_texts(texts, batch_size=batch_size)

    for doc, emb in zip(documents, embeddings):
        doc.embedding = emb

    logger.info("Embedded %d documents (dim=%d)", len(documents), EMBEDDING_DIM)
    return documents
