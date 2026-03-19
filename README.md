# AI Research Intelligence RAG

A fully local, CPU-first Retrieval-Augmented Generation system for answering business-oriented questions about recent AI research. It ingests arXiv paper abstracts, indexes them with FAISS, and synthesizes citation-backed answers using a quantized local LLM - with no paid APIs, no GPU requirement, and no external runtime dependencies.

---

## Overview

This system is built for investors, analysts, and strategy teams who need structured answers from the growing volume of AI research output. A user submits a natural language question; the system retrieves semantically relevant papers, reranks them for precision, and generates a grounded, multi-source answer from a local LLM.

The corpus is organized around a 12-topic AI taxonomy covering areas such as Multimodal AI, Agents, Reasoning, and Safety. Ingestion is an offline batch process; query serving is fully stateless and self-contained, with no database or API dependency at runtime.

---

## Quick Start

```bash
# 1. Create required directories
mkdir -p models data/index

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the GGUF model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
    qwen2.5-3b-instruct-q5_k_m.gguf \
    --local-dir models/

# 4. (Optional) Download prebuilt index artifacts - skips ~15–20 min of ingestion
#    See the Prebuilt Artifacts section below

# 5. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 6. Run the test script (in a separate terminal)
python scripts/test_api.py
```

---

## Architecture

The pipeline is split into two phases: an offline ingestion phase that builds the vector index, and an online serving phase that handles queries end-to-end.

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

---

## Key Design Decisions

**FAISS `IndexFlatIP` over approximate indexes (HNSW, IVF).** For a corpus of tens of thousands of paper abstracts, an exact flat index fits comfortably in RAM and returns results in under 20ms. Approximate indexes trade recall for speed - a tradeoff that only becomes necessary at millions of vectors. Using an exact index here means no index tuning, no recall degradation, and no surprise behavior at query time. L2 normalization of embeddings converts inner product to cosine similarity without a separate index type.

**Post-filtering over pre-filtering for metadata constraints.** Metadata filters (topic, date, category) are applied after retrieving an expanded candidate set (`search_k = max(top_k * 5, 50)`) rather than partitioning the index upfront. Pre-filtering with a flat index would require either separate per-partition indexes or a brute-force scan over filtered subsets - both add operational complexity. Post-filtering over a widened retrieval set is simpler, produces consistently well-ranked candidates, and avoids empty-result edge cases when filter selectivity is high.

**Two-stage retrieval: bi-encoder recall + cross-encoder reranking.** The bi-encoder (MiniLM) is fast but encodes query and documents independently, which limits precision. The cross-encoder jointly encodes each query-document pair and produces substantially better relevance scores, but is too slow to run over the full index. The two-stage design gets the best of both: the bi-encoder narrows the candidate set cheaply, and the cross-encoder rescores only the shortlist.

**Qwen 2.5-3B at Q5_K_M quantization over larger models or APIs.** A 3B parameter model quantized to Q5_K_M runs on a standard laptop CPU with 8GB RAM in 3–12 seconds per query. Models larger than 7B become impractically slow on CPU. Commercial API alternatives would violate the local-first constraint and introduce per-query cost and latency variability. The 3B model is sufficient for the task: it synthesizes a short, structured answer from pre-retrieved and pre-ranked context, which is a much easier job than open-domain generation.

**Keyword-based topic mapping over ML classification.** Topic assignment uses regex patterns over title and abstract text. This is deterministic, zero-cost, and fully reproducible - no model to fine-tune, no inference step, no external dependency. For a curated 12-topic taxonomy over AI papers, keyword coverage is high and edge cases are debuggable via the reason string included on each assignment. ML classification would add complexity without meaningfully improving downstream retrieval quality.

**Abstract-level retrieval units with no chunking.** Each paper is a single retrieval unit. Abstracts are dense summaries written to capture a paper's core contribution - the highest-signal text available per paper. For trend-level business questions, abstract-level retrieval is sufficient and avoids the indexing complexity, chunk boundary artifacts, and deduplication overhead that full-text chunking introduces.

---

## System Components

| Component | Implementation | Notes |
|---|---|---|
| Vector index | FAISS `IndexFlatIP` | Exact search; L2-normalized vectors for cosine similarity |
| Embeddings | `all-MiniLM-L6-v2` (384d) | 22M params; fast CPU inference; used offline and online |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-document scoring; precision pass over shortlist |
| Generator | Qwen 2.5-3B Instruct (Q5_K_M GGUF) | Local inference via `llama-cpp-python`; no API required |
| API | FastAPI | `/query`, `/health`, `/metrics` endpoints |
| UI | Streamlit | Optional; query, evaluation, and debug tabs |
| Data models | Pydantic v2 | Strict validation at every pipeline boundary |

---

## Models

**Embedding model - `all-MiniLM-L6-v2`**
Used for both offline document embedding and online query embedding. 22M parameters, 384-dimensional output, fast on CPU. Downloaded automatically via `sentence-transformers`.
[https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Reranker - `cross-encoder/ms-marco-MiniLM-L-6-v2`**
Cross-encoder trained on MS MARCO for passage relevance scoring. Used to rerank the bi-encoder shortlist. Downloaded automatically via `sentence-transformers`.
[https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

**LLM - Qwen 2.5-3B Instruct (GGUF)**
Must be downloaded manually and placed at `models/qwen2.5-3b-instruct-q5_k_m.gguf`. GGUF is the quantized model format consumed by `llama-cpp-python` for CPU inference. The Q5_K_M variant balances answer quality with runtime memory usage (~2.5GB).

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
    qwen2.5-3b-instruct-q5_k_m.gguf \
    --local-dir models/
```

---

## Prebuilt Artifacts

A prebuilt FAISS index and metadata file are available on Hugging Face, covering the curated arXiv dataset used in this project:

[https://huggingface.co/datasets/antalbalint97/ai-research-intelligence-artifacts](https://huggingface.co/datasets/antalbalint97/ai-research-intelligence-artifacts)

Downloading these files skips the ~15–20 minute offline ingestion and embedding step. Place both files into `data/index/` before starting the API:

```
data/index/arxiv_ai.index
data/index/arxiv_ai_metadata.jsonl
```

Running the full ingestion pipeline from raw arXiv data remains supported and is recommended for validating the end-to-end system or building a custom corpus.

---

## Query Modes

The pipeline supports two modes, set via the `mode` field on `QueryRequest`.

| Parameter | Fast | Full |
|---|---|---|
| Retrieval candidates | 8 | 20 |
| Final results | 3 | 5 |
| Context character limit | 3,500 | 12,000 |
| Max generation tokens | 96 | 384 |
| Temperature | 0.1 | 0.2 |
| LLM context window | 2,048 | 4,096 |
| Prompt style | Structured outline (≤110 words) | Narrative with trends and evidence |

**Fast mode** is the default for interactive use. It returns a concise, structured answer in roughly 3–8 seconds on modern laptop hardware. **Full mode** provides deeper synthesis suitable for reports or detailed analysis, at 10–45+ seconds depending on CPU.

---

## How to Run

### Setup

```bash
git clone https://github.com/your-username/ai-research-rag.git
cd ai-research-rag

cp .env.example .env
pip install -r requirements.txt
```

Configure `.env` with the following required variables:

```
FAISS_INDEX_PATH=data/index/arxiv_ai.index
FAISS_METADATA_PATH=data/index/arxiv_ai_metadata.jsonl
LLM_MODEL_PATH=models/qwen2.5-3b-instruct-q5_k_m.gguf
LLM_BACKEND=llama_cpp
LLM_N_GPU_LAYERS=0
```

Set `LLM_N_GPU_LAYERS` to a positive integer to offload layers to GPU if available (CUDA or Metal).

### Ingestion

Build the FAISS index from a curated arXiv dataset. This is a one-time offline step.

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

On startup, the API preloads the embedding model and FAISS index to minimize first-query latency. The reranker and LLM load lazily on the first query, so the initial request will be slower than subsequent ones.

### Quick Test

```bash
python scripts/test_api.py
```

Sends a sample query to the running API and prints the full response with per-stage timings.

### Docker (Optional)

```bash
# Build
docker build -t ai-research-rag .

# Run (models and data must be mounted - the container has neither)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  ai-research-rag
```

The Streamlit UI can be started separately:

```bash
streamlit run app/streamlit_app.py
```

---

## API Example

**Request:**

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

**Response:**

```json
{
  "answer": "Recent multimodal AI research focuses on...",
  "sources": [
    {
      "title": "...",
      "url": "https://arxiv.org/abs/...",
      "published_date": "2025-09-15",
      "primary_topic": "Multimodal AI",
      "relevance_score": 0.92
    }
  ],
  "mode": "fast",
  "retrieval_count": 8,
  "reranked_count": 3,
  "latency_ms": 4200.0,
  "timings": {
    "embed_ms": 45.2,
    "retrieve_ms": 12.1,
    "rerank_ms": 320.5,
    "prompt_build_ms": 0.8,
    "generate_ms": 3800.0,
    "total_ms": 4200.0
  },
  "prompt_chars": 2100,
  "answer_chars": 380,
  "model": "models/qwen2.5-3b-instruct-q5_k_m.gguf"
}
```

---

## Performance and Observability

Every response includes a `timings` object with per-stage latency breakdown:

| Stage | Typical Latency | Notes |
|---|---|---|
| Embedding | 30–80ms | MiniLM on CPU; negligible after warmup |
| FAISS retrieval | 5–20ms | Exact flat search; negligible at this corpus size |
| Cross-encoder reranking | 200–500ms | Scales with shortlist size, not corpus size |
| LLM generation | 3,000–12,000ms | Dominant cost; scales with `max_tokens` and `n_ctx` |
| **Total (fast mode)** | **~3–13s** | Typical on modern laptop CPU |
| **Total (full mode)** | **~10–45s** | Hardware-dependent |

LLM generation accounts for over 85% of end-to-end latency in virtually every query. All other pipeline stages are fast enough that optimizing them has negligible impact on user-perceived response time. The primary lever for reducing latency is GPU offloading via `LLM_N_GPU_LAYERS`, which can bring generation time down by an order of magnitude on CUDA hardware.

The `/metrics` endpoint exposes aggregate query statistics across the server's lifetime. The API warms up the embedding model and FAISS index on startup to separate initialization cost from steady-state performance.

If `llama-cpp-python` is not installed or the GGUF model file is missing, the generator returns a structured fallback message instead of raising an error. Retrieval and reranking continue to function normally in this state, making it possible to validate the full pipeline up to generation without a model present.

---

## Limitations

- **CPU latency is the primary constraint.** Fast mode is usable interactively; full mode is better suited to async or batch workflows. Sub-second latency is not achievable without GPU offloading.
- **Abstract-only corpus.** Full paper text is not parsed. Answer depth is limited for questions that require methodological or experimental detail beyond what an abstract conveys.
- **No incremental index updates.** Adding new papers requires a full offline rebuild of the FAISS index. Acceptable for periodic batch ingestion; unsuitable for streaming updates.
- **Heuristic evaluation only.** Answer quality is assessed via length, structure, and keyword heuristics. The semantic similarity metric (`semantic_similarity_placeholder`) returns 0.0. BERTScore and RAGAS are stubbed but not implemented.
- **Keyword-based topic mapping.** Regex heuristics handle the majority of cases well, but papers spanning multiple subfields or using non-standard terminology may be misclassified. Misclassification affects filter-based queries more than open queries.

---

## Future Work

- **GPU inference.** Enable `LLM_N_GPU_LAYERS` for CUDA/Metal offloading. The config variable already exists; `llama-cpp-python` supports both backends. This is the highest-impact single improvement available.
- **Semantic evaluation.** Replace heuristic metrics with BERTScore or RAGAS for reference-based quality measurement. Requires a labeled evaluation set with ground-truth answers.
- **Incremental FAISS updates.** Support appending new embeddings without a full index rebuild via `IndexIDMap`, with a corresponding metadata store update strategy.
- **Full-text retrieval.** Parse and chunk full PDF content for deeper retrieval. The `PaperRecord.full_text_path` field is already reserved for this extension.

---

## Repository Structure

```
ai-research-rag/
├── api/
│   ├── main.py                   # FastAPI app: /query, /health, /metrics
│   ├── schemas.py                # Pydantic model re-exports
│   └── middleware.py             # Request logging and timing
├── app/
│   └── streamlit_app.py          # Optional Streamlit UI
├── ingestion/
│   ├── load_arxiv.py             # Multi-format loader (JSONL/JSON/CSV)
│   ├── filter_papers.py          # Date and category filtering
│   ├── topic_mapper.py           # Keyword-based topic assignment (12 topics)
│   ├── build_documents.py        # PaperRecord normalization
│   ├── embedder.py               # Batch embedding, offline
│   ├── embedder_optimized.py     # Warmed-up embedding, online
│   └── build_faiss_index.py      # FAISS index builder (active offline path)
├── pipeline/
│   ├── models.py                 # Pydantic data models
│   ├── rag_pipeline.py           # Query orchestrator
│   ├── retriever_faiss.py        # FAISS retrieval with metadata post-filtering
│   ├── reranker.py               # Cross-encoder reranking
│   ├── generator.py              # llama.cpp generation
│   └── prompt.py                 # Prompt templates (fast and full modes)
├── evaluation/
│   ├── metrics.py                # Heuristic quality metrics
│   ├── testset.py                # 25 business-style test questions
│   └── simple_eval.py            # Full evaluation runner
├── scripts/
│   └── test_api.py               # Quick API smoke test
├── tests/                        # Unit tests
├── data/
│   ├── raw/                      # Input arXiv data
│   ├── processed/                # Curated datasets
│   ├── index/                    # FAISS index and metadata
│   └── artifacts/                # Evaluation results
├── docs/
│   └── architecture.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## License

MIT