# AI Research Intelligence RAG

Local Retrieval-Augmented Generation system for answering business-oriented questions about recent AI research, built on arXiv paper abstracts. No paid APIs, no GPU required, fully reproducible on CPU.

---

## Project Overview

This system ingests arXiv metadata, maps papers to a 12-topic AI taxonomy, embeds and indexes them into a FAISS vector store, and serves a two-stage retrieval pipeline (bi-encoder recall + cross-encoder reranking) backed by a local quantized LLM (Qwen 2.5-3B via llama.cpp).

Answers are synthesized across multiple papers and framed for a business audience (investors, analysts, strategy teams) with source citations and per-stage latency instrumentation.

---

## Architecture Summary

```
Offline                                    Online
───────                                    ──────
arXiv JSONL/CSV/JSON                       User Query (FastAPI / Streamlit)
  │                                            │
  ▼                                            ▼
load_arxiv.py ─► filter_papers.py          embed_query (MiniLM-L6-v2, 384d)
  │                                            │
  ▼                                            ▼
topic_mapper.py ─► build_documents.py      FAISS IndexFlatIP search (top-K)
  │                                          + metadata post-filters
  ▼                                            │
embedder.py (MiniLM-L6-v2, batch)             ▼
  │                                        cross-encoder rerank (ms-marco-MiniLM)
  ▼                                            │
build_faiss_index.py                           ▼
  ├── data/index/arxiv_ai.index            prompt assembly (full / fast templates)
  └── data/index/arxiv_ai_metadata.jsonl       │
                                               ▼
                                           llama.cpp generation (Qwen 2.5-3B GGUF)
                                               │
                                               ▼
                                           QueryResponse (answer + sources + timings)
```

| Component | Technology | Notes |
|---|---|---|
| Vector index | FAISS `IndexFlatIP` | L2-normalized embeddings for cosine similarity |
| Embeddings | `all-MiniLM-L6-v2` (384d) | 22M params, CPU-friendly |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder precision pass |
| Generator | Qwen 2.5-3B Instruct (Q5_K_M GGUF) | Local, via `llama-cpp-python` |
| API | FastAPI | `/query`, `/health`, `/metrics` endpoints |
| UI | Streamlit (`app/streamlit_app.py`) | Query, evaluation, and debug tabs |
| Data models | Pydantic v2 | Strict validation across the pipeline |

---

## Key Design Decisions

**FAISS over pgvector.** The online retrieval path (`pipeline/retriever_faiss.py`) uses a flat inner-product FAISS index. This removes the PostgreSQL dependency for query serving. Metadata filtering (topic, category, date) is applied as a post-filter over a wider initial candidate set (`search_k = max(top_k * 5, 50)`). Deduplication by `paper_id` prevents returning multiple chunks from the same paper.

**Local LLM via llama.cpp.** Generation (`pipeline/generator.py`) loads a GGUF-quantized model using `llama-cpp-python`. The default is `qwen2.5-3b-instruct-q5_k_m.gguf`. No external API calls. The generator supports both chat completion and text completion fallback depending on model capabilities. On failure, a graceful fallback message is returned instead of an error.

**Two-stage retrieval.** Bi-encoder (MiniLM) retrieves an initial candidate set optimized for recall. Cross-encoder (`pipeline/reranker.py`) rescores each query-document pair for precision. This avoids running the expensive cross-encoder on the full corpus.

**Keyword-based topic mapping.** `ingestion/topic_mapper.py` assigns papers to 12 predefined topics using regex patterns over title + abstract. Deterministic, fast, no API dependency. Each paper gets a primary topic, optional secondary topics, and a reason string for debugging.

**Abstract-level retrieval units.** No chunking in the FAISS-based path. Each abstract is one retrieval unit. Title + abstract captures the key signal for trend-level questions. The `full_text_path` field on `PaperRecord` is reserved for future full-text support.

---

## Fast vs Full Mode

The pipeline supports two query modes, configured via the `mode` parameter on `QueryRequest`:

| Parameter | Fast | Full |
|---|---|---|
| Retrieval candidates (`retrieval_k`) | 8 | 20 |
| Final results (`top_k`) | 3 | 5 |
| Context char limit | 3,500 | 12,000 |
| Max generation tokens | 96 | 384 |
| Temperature | 0.1 | 0.2 |
| LLM context window (`n_ctx`) | 2,048 | 4,096 |
| Prompt style | Outline (max 110 words) | Narrative with trends + evidence |

Fast mode is designed for low-latency interactive use. Full mode provides richer synthesis at higher CPU cost.

---

## How to Run

### Prerequisites

- Python 3.10+
- A GGUF model file (default: `models/qwen2.5-3b-instruct-q5_k_m.gguf`)
- arXiv data in `data/raw/` (JSONL, JSON, or CSV)

### Environment Setup

```bash
cp .env.example .env
# Edit .env — key variables:
#   FAISS_INDEX_PATH=data/index/arxiv_ai.index
#   FAISS_METADATA_PATH=data/index/arxiv_ai_metadata.jsonl
#   LLM_MODEL_PATH=models/qwen2.5-3b-instruct-q5_k_m.gguf
#   LLM_BACKEND=llama_cpp

pip install -r requirements.txt
```

### Ingestion (Offline)

Build the FAISS index from a curated dataset:

```bash
python ingestion/build_faiss_index.py \
    --input data/processed/arxiv_ai_curated.jsonl \
    --index-output data/index/arxiv_ai.index \
    --metadata-output data/index/arxiv_ai_metadata.jsonl
```

This embeds each record's `content` field with MiniLM, L2-normalizes the vectors, and writes a FAISS `IndexFlatIP` index alongside an aligned metadata JSONL file.

### API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On startup, the API warms up the embedding model and FAISS index to reduce first-query latency.

### Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The UI provides three tabs: Query (interactive RAG), Evaluation (smoke and full eval), and Debug (environment and state inspection).

### Evaluation

```bash
# Smoke eval: 5 queries in fast mode, with warmup
python -m evaluation.smoke_eval

# Full eval: 25 queries across all topics
python -m evaluation.simple_eval
```

---

## API Usage Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the recent trends in multimodal AI?",
    "mode": "fast",
    "top_k": 3,
    "filters": {
      "primary_topic": "Multimodal AI"
    }
  }'
```

Response structure:

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
  "latency_ms": 4200.0,
  "model": "models/qwen2.5-3b-instruct-q5_k_m.gguf",
  "retrieval_count": 8,
  "reranked_count": 3,
  "mode": "fast",
  "timings": {
    "embed_ms": 45.2,
    "retrieve_ms": 12.1,
    "rerank_ms": 320.5,
    "prompt_build_ms": 0.8,
    "generate_ms": 3800.0,
    "total_ms": 4200.0
  },
  "prompt_chars": 2100,
  "answer_chars": 380
}
```

---

## Evaluation

Two evaluation modes exist:

**Smoke evaluation** (`evaluation/smoke_eval.py`): Runs a small subset (default 5) of the 25-question test set in fast mode. Includes a non-measured warmup query to isolate model-loading latency. Reports per-query timings (embed, retrieve, rerank, generate), pass/fail status, and aggregate statistics (mean/median latency). Designed for development iteration.

**Full evaluation** (`evaluation/simple_eval.py`): Runs all 25 test cases. Supports live mode (full pipeline) or dry-run mode (validates test set structure without models). Each query is checked against three heuristic metrics defined in `evaluation/metrics.py`:
- `answer_non_empty`: answer has at least 20 characters and 3 words
- `answer_structure_score`: heuristic 0.0-1.0 based on length, paragraph count, specificity keywords, and uncertainty markers
- `latency_acceptable`: response time within threshold (default 10s for full eval, 30s for smoke)

The test set (`evaluation/testset.py`) contains 25 manually-authored business-style questions spanning all 12 topics, each with expected topic labels and relevance keywords.

---

## Known Limitations

- **CPU latency.** LLM generation via llama.cpp on CPU is the dominant bottleneck. Expect 3-15 seconds per query in fast mode, 10-45+ seconds in full mode depending on hardware. The cross-encoder reranking step adds 200-500ms. This is an intentional tradeoff: fully local, no API costs, but slower than GPU or cloud inference.
- **No semantic evaluation metrics.** `semantic_similarity_placeholder` in `evaluation/metrics.py` returns 0.0. BERTScore/RAGAS integration is stubbed but not implemented.
- **Abstract-only corpus.** Full paper text is not parsed. This limits answer depth for questions requiring methodological detail.
- **Keyword-based topic mapping.** Topic assignment uses regex heuristics, not semantic understanding. Edge cases (papers spanning multiple subfields) may be misclassified.
- **Legacy ingestion path.** `ingestion/run.py` still references pgvector and a chunker module. The active offline path uses `ingestion/build_faiss_index.py` directly.
- **No incremental index updates.** The FAISS index must be rebuilt from scratch when new papers are added.

---

## Future Improvements

- **GPU-accelerated inference.** Adding GPU support for llama.cpp would reduce generation latency by an order of magnitude. The `LLM_N_GPU_LAYERS` config already exists but defaults to 0.
- **RAGAS / BERTScore evaluation.** Replace the placeholder semantic similarity metric with proper reference-based evaluation.
- **Incremental FAISS updates.** Support appending new embeddings to the existing index without full rebuild.
- **Full-text PDF parsing.** Extract and chunk full paper content for deeper retrieval. The `PaperRecord.full_text_path` field is already reserved for this.

---

## Repository Structure

```
ai-research-rag/
├── api/
│   ├── main.py               # FastAPI app, /query /health /metrics endpoints
│   ├── schemas.py             # Re-exports Pydantic models from pipeline.models
│   └── middleware.py          # Request logging + timing middleware
├── app/
│   └── streamlit_app.py       # Streamlit UI (query, eval, debug tabs)
├── ingestion/
│   ├── load_arxiv.py          # Multi-format arXiv loader (JSONL/JSON/CSV)
│   ├── filter_papers.py       # Date and AI-category filtering
│   ├── topic_mapper.py        # Keyword-based topic assignment (12 topics)
│   ├── build_documents.py     # PaperRecord -> DocumentRecord normalization
│   ├── embedder.py            # Batch embedding with MiniLM (offline)
│   ├── embedder_optimized.py  # Low-latency embedding (online, with warmup)
│   ├── build_faiss_index.py   # FAISS index builder (CLI tool)
│   └── run.py                 # Legacy ingestion orchestrator (pgvector path)
├── pipeline/
│   ├── models.py              # Pydantic data models (PaperRecord, QueryRequest, etc.)
│   ├── rag_pipeline.py        # Online query orchestrator (embed -> retrieve -> rerank -> generate)
│   ├── retriever_faiss.py     # FAISS search with metadata post-filtering
│   ├── reranker.py            # Cross-encoder reranking
│   ├── generator.py           # llama.cpp generation (GGUF models)
│   └── prompt.py              # Prompt templates (full and fast modes)
├── evaluation/
│   ├── metrics.py             # Heuristic quality metrics
│   ├── testset.py             # 25 business-style test questions
│   ├── simple_eval.py         # Full evaluation runner
│   └── smoke_eval.py          # Fast smoke evaluation runner
├── tests/                     # Unit tests
├── data/
│   ├── raw/                   # Input arXiv data
│   ├── processed/             # Curated datasets
│   ├── index/                 # FAISS index + metadata
│   └── artifacts/             # Evaluation results
├── docs/
│   └── architecture.md        # Technical architecture document
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## License

MIT
