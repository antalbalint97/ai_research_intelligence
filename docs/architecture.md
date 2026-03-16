# Architecture – AI Research Intelligence RAG

## Executive Summary

This document describes the technical architecture of the AI Research Intelligence RAG system — a containerized Retrieval-Augmented Generation (RAG) Q&A system designed to transform arXiv AI research papers into strategic business intelligence.

The system ingests arXiv metadata, maps papers to a topic taxonomy, chunks and embeds content into a pgvector store, and answers natural-language questions via a two-stage retrieval pipeline backed by a generative LLM.

---

## Design Goals

1. **Business accessibility** – Answers are written in clear, investor-friendly language with commercial relevance commentary
2. **Production readiness** – Docker-based deployment, environment-based configuration, robust error handling
3. **Explainability** – Every architectural decision is documented and defensible
4. **Extensibility** – Architecture supports future addition of full-text parsing, dashboards, and advanced analytics
5. **Simplicity** – Minimal framework usage; explicit, readable code

---

## Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Ecosystem maturity, ML library support |
| Vector Store | PostgreSQL + pgvector | SQL filtering + vector search in one service |
| Embeddings | `all-MiniLM-L6-v2` (384d) | Fast, CPU-friendly, well-benchmarked |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder precision without GPU requirement |
| Generator (primary) | Mistral-7B-Instruct (HF API) | High-quality instruction following |
| Generator (fallback) | flan-t5-base (local) | Guarantees availability without API dependency |
| API Framework | FastAPI | Async, OpenAPI docs, Pydantic validation |
| Frontend | Vanilla HTML/JS/CSS | Zero build tooling, served from FastAPI |
| Container | Docker Compose | Reproducible multi-service deployment |
| Testing | pytest | Standard Python testing framework |

---

## Request Lifecycle

### Online Query Flow

```
User Query
    │
    ▼
[1] Query Embedding (MiniLM-L6-v2)
    │
    ▼
[2] pgvector Cosine Similarity Search (top-20)
    │   + optional metadata filters (topic, category, date)
    │
    ▼
[3] Cross-Encoder Reranking (top-5)
    │
    ▼
[4] Prompt Assembly (context + system prompt + query)
    │
    ▼
[5] LLM Generation (Mistral-7B → flan-t5 fallback)
    │
    ▼
[6] Response Formatting (answer + sources + metadata)
    │
    ▼
JSON Response → Frontend Display
```

### Offline Ingestion Flow

```
Raw arXiv Data (JSONL/CSV/JSON)
    │
    ▼
[1] Load & Parse (field-name normalization)
    │
    ▼
[2] Date Filter (last 9 months)
    │
    ▼
[3] Category + Keyword Filter (AI-relevant only)
    │
    ▼
[4] Topic Assignment (keyword heuristics → 12 topics)
    │
    ▼
[5] Document Building (structured text format)
    │
    ▼
[6] Chunking (sentence-aware sliding window)
    │
    ▼
[7] Embedding (MiniLM-L6-v2, batch encoding)
    │
    ▼
[8] pgvector Upsert (ON CONFLICT update)
```

---

## Component Decision Matrix

| Decision | Options Considered | Chosen | Why |
|---|---|---|---|
| Vector DB | FAISS, Pinecone, Weaviate, pgvector | pgvector | SQL metadata filtering, single Docker service, ACID guarantees |
| Embedding model | OpenAI ada-002, MiniLM, BGE | MiniLM-L6-v2 | Free, fast on CPU, 384d is sufficient for abstract-level search |
| Reranking | No reranker, Cohere, cross-encoder | cross-encoder | Free, local, dramatic precision improvement |
| Generator | GPT-4, Claude, Mistral, Llama | Mistral-7B via HF API | Free tier available, high quality, instruction-tuned |
| Chunking | Fixed-length, sentence-aware, recursive | Sentence-aware sliding window | Respects sentence boundaries, configurable overlap |
| Frontend | Streamlit, React, Vanilla JS | Vanilla JS | Zero build tooling, served from same container |
| Topic mapping | LLM-based, embedding clustering, keywords | Keyword heuristics | Deterministic, fast, no API calls, extensible |

---

## Data Schema

### PaperRecord (raw input)

| Field | Type | Description |
|---|---|---|
| id | str | arXiv paper ID |
| title | str | Paper title |
| abstract | str | Paper abstract |
| authors | str | Comma-separated author names |
| categories | str | Space-separated arXiv categories |
| published_date | date | Publication date |
| doi | str? | DOI if available |
| journal_ref | str? | Journal reference |

### DocumentRecord (indexed)

| Field | Type | Description |
|---|---|---|
| doc_id | str | SHA-256-derived unique ID |
| paper_id | str | Source arXiv ID |
| title | str | Paper title |
| content | str | Structured document text |
| source | str | Always "arxiv" |
| doc_type | str | "paper_abstract" or "paper_chunk" |
| published_date | date | Publication date |
| categories | str | arXiv categories |
| authors | str | Author names |
| primary_topic | str | Assigned primary AI topic |
| secondary_topics | list[str] | Additional matched topics |
| language | str | Always "en" |
| url | str? | arXiv abstract URL |
| metadata | dict | Additional metadata (topic_reason, chunk_index) |
| embedding | list[float] | 384-dimensional vector |

---

## Repository Layout

```
ai-research-rag/
├── api/                  # FastAPI application layer
│   ├── main.py           # App setup, endpoints, startup
│   ├── schemas.py        # Request/response schema re-exports
│   └── middleware.py      # Logging and timing middleware
├── ingestion/            # Offline data processing
│   ├── load_arxiv.py     # Multi-format data loader
│   ├── filter_papers.py  # Date and category filtering
│   ├── topic_mapper.py   # Keyword-based topic assignment
│   ├── build_documents.py # Structured document builder
│   ├── chunker.py        # Sentence-aware text chunking
│   ├── embedder.py       # MiniLM embedding module
│   └── run.py            # Ingestion orchestrator
├── pipeline/             # Online query processing
│   ├── models.py         # Pydantic data models
│   ├── retriever.py      # pgvector search + schema management
│   ├── reranker.py       # Cross-encoder reranking
│   ├── generator.py      # LLM generation with fallback
│   ├── prompt.py         # Prompt templates
│   └── rag_pipeline.py   # Query orchestrator
├── evaluation/           # Quality assessment
│   ├── metrics.py        # Hit rate, structure score, latency
│   ├── testset.py        # 25 business-style test questions
│   └── simple_eval.py    # Evaluation runner
├── frontend/             # Web UI
│   ├── index.html        # Chat interface
│   ├── app.js            # API interaction logic
│   └── style.css         # Clean, professional styling
├── tests/                # Unit tests
├── scripts/              # Shell automation
├── data/                 # Data directories
└── docs/                 # Documentation
```

---

## API Specification

### `GET /health`
Returns system health status and uptime.

### `POST /query`
Execute a RAG query.

**Request:**
```json
{
  "query": "What are the recent trends in multimodal AI?",
  "filters": {
    "primary_topic": "Multimodal AI",
    "category": "cs.CV",
    "date_from": "2025-07-01"
  },
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [
    {
      "title": "...",
      "url": "https://arxiv.org/abs/...",
      "published_date": "2025-09-15",
      "primary_topic": "Multimodal AI",
      "relevance_score": 0.92
    }
  ],
  "latency_ms": 1320,
  "model": "mistral-7b-instruct-v0.2",
  "retrieval_count": 20,
  "reranked_count": 5
}
```

### `GET /topics`
Returns the full AI topic taxonomy with English and Hungarian labels.

### `GET /metrics`
Returns operational metrics (query count, average latency, uptime).

---

## Evaluation

The evaluation pipeline uses three tiers:

1. **Deterministic checks** – answer non-emptiness, structural quality scoring, latency bounds
2. **Test set coverage** – 25 manually-authored business-style questions across all topics
3. **Placeholder hooks** – semantic similarity and BERTScore integration points for future use

The evaluation can run in two modes:
- **Live mode** – queries the full pipeline with database and models
- **Dry-run mode** – validates test set structure without external dependencies

---

## Reproducibility

The system is fully reproducible via Docker:

```bash
# Start all services
docker compose up -d

# Run ingestion
docker compose exec api python -m ingestion.run

# Run evaluation
docker compose exec api python -m evaluation.simple_eval
```

All configuration is environment-variable driven. No secrets are hardcoded.

---

## End-User Accessibility Suggestions

For maximum adoption, consider:

1. **Quick-start topic chips** – Pre-defined question templates for common topics (implemented in frontend)
2. **Source transparency** – Every answer includes clickable arXiv source links
3. **Uncertainty flagging** – The system explicitly notes when context is insufficient
4. **Jargon translation** – Prompts instruct the LLM to explain technical terms simply
5. **Hungarian translations** – Topic taxonomy includes Magyar translations for bilingual teams
