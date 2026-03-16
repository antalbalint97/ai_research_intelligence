"""Vector retriever – pgvector-backed semantic search with metadata filtering.

Uses raw psycopg2 for maximum control and minimal abstraction.
Creates the required schema (table + HNSW index) on first use.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras

from pipeline.models import DocumentRecord, SourceCitation

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------
CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    source          TEXT DEFAULT 'arxiv',
    doc_type        TEXT DEFAULT 'paper_abstract',
    published_date  DATE,
    categories      TEXT DEFAULT '',
    authors         TEXT DEFAULT '',
    primary_topic   TEXT DEFAULT '',
    secondary_topics TEXT[] DEFAULT ARRAY[]::TEXT[],
    language        TEXT DEFAULT 'en',
    url             TEXT,
    metadata        JSONB DEFAULT '{{}}'::JSONB,
    embedding       vector({EMBEDDING_DIM})
);
"""

CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_documents_embedding
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
"""

UPSERT_SQL = f"""
INSERT INTO documents (
    doc_id, paper_id, title, content, source, doc_type,
    published_date, categories, authors, primary_topic,
    secondary_topics, language, url, metadata, embedding
) VALUES (
    %(doc_id)s, %(paper_id)s, %(title)s, %(content)s, %(source)s, %(doc_type)s,
    %(published_date)s, %(categories)s, %(authors)s, %(primary_topic)s,
    %(secondary_topics)s, %(language)s, %(url)s, %(metadata)s, %(embedding)s
)
ON CONFLICT (doc_id) DO UPDATE SET
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding,
    metadata = EXCLUDED.metadata;
"""

SEARCH_SQL = """
SELECT
    doc_id, paper_id, title, content, source, doc_type,
    published_date, categories, authors, primary_topic,
    secondary_topics, url,
    1 - (embedding <=> %(query_embedding)s::vector) AS similarity
FROM documents
WHERE 1=1
"""


def _get_db_url() -> str:
    """Resolve database URL from environment."""
    return os.environ.get(
        "DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/arxiv_rag"
    )


@contextmanager
def _connect(database_url: str | None = None) -> Generator:
    """Create a database connection context manager."""
    url = database_url or _get_db_url()
    conn = psycopg2.connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema(database_url: str | None = None) -> None:
    """Create the pgvector extension, table, and HNSW index if they don't exist.

    Args:
        database_url: PostgreSQL connection string. Falls back to DATABASE_URL env var.
    """
    with _connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_EXTENSION_SQL)
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_INDEX_SQL)
    logger.info("Database schema ensured (table + HNSW index)")


def upsert_documents(documents: list[DocumentRecord], database_url: str | None = None) -> int:
    """Upsert document records into pgvector.

    Args:
        documents: List of DocumentRecord objects with embeddings.
        database_url: PostgreSQL connection string.

    Returns:
        Number of documents upserted.
    """
    count = 0
    with _connect(database_url) as conn:
        with conn.cursor() as cur:
            for doc in documents:
                if doc.embedding is None:
                    logger.warning("Skipping document %s: no embedding", doc.doc_id)
                    continue
                params = {
                    "doc_id": doc.doc_id,
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "published_date": doc.published_date,
                    "categories": doc.categories,
                    "authors": doc.authors,
                    "primary_topic": doc.primary_topic,
                    "secondary_topics": doc.secondary_topics,
                    "language": doc.language,
                    "url": doc.url,
                    "metadata": psycopg2.extras.Json(doc.metadata),
                    "embedding": doc.embedding,
                }
                cur.execute(UPSERT_SQL, params)
                count += 1
    logger.info("Upserted %d documents", count)
    return count


def search_documents(
    query_embedding: list[float],
    top_k: int = 20,
    primary_topic: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    database_url: str | None = None,
) -> list[dict]:
    """Retrieve the most similar documents from pgvector.

    Args:
        query_embedding: Query vector (dim=384).
        top_k: Maximum number of results.
        primary_topic: Optional filter by primary_topic.
        category: Optional filter by arXiv category.
        date_from: Optional date lower bound (ISO format string).
        database_url: PostgreSQL connection string.

    Returns:
        List of result dicts with document fields and similarity score.
    """
    sql = SEARCH_SQL
    params: dict = {"query_embedding": query_embedding}

    if primary_topic:
        sql += " AND primary_topic = %(primary_topic)s"
        params["primary_topic"] = primary_topic

    if category:
        sql += " AND categories LIKE %(category_pattern)s"
        params["category_pattern"] = f"%{category}%"

    if date_from:
        sql += " AND published_date >= %(date_from)s"
        params["date_from"] = date_from

    sql += " ORDER BY embedding <=> %(query_embedding)s::vector ASC LIMIT %(top_k)s"
    params["top_k"] = top_k

    results: list[dict] = []
    with _connect(database_url) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            for row in rows:
                results.append(dict(row))

    logger.info("Retrieved %d documents (top_k=%d)", len(results), top_k)
    return results
