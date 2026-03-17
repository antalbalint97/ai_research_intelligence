# Architecture -- AI Research Intelligence RAG

## 1. System Overview

A local RAG system that answers business-oriented questions about AI research using arXiv paper abstracts. The system has two distinct pipelines:

- **Offline pipeline**: loads arXiv metadata, filters to AI-relevant papers, assigns topics via keyword heuristics, embeds content with MiniLM-L6-v2, and builds a FAISS index.
- **Online pipeline**: embeds the user query, retrieves candidates from FAISS, reranks with a cross-encoder, assembles a prompt, and generates an answer with a local quantized LLM.

All inference runs on CPU. No external APIs are called at query time. The system is exposed via FastAPI (REST) and Streamlit (interactive UI).

---

## 2. Offline Pipeline

The offline pipeline transforms raw arXiv data into a searchable FAISS index. Two code paths exist: a legacy orchestrator (`ingestion/run.py`, targets pgvector) and the active FAISS builder (`ingestion/build_faiss_index.py`).

### Active path: `ingestion/build_faiss_index.py`

Operates on an already-curated JSONL file (produced by separate data prep scripts).

**Steps:**

1. **Stream records** from input JSONL (`iter_jsonl`). Each record has fields like `paper_id`, `title`, `abstract`, `content`, `categories`, `primary_topic_en`, `published_date`.
2. **Normalize text** (`normalize_text`): collapse whitespace, strip.
3. **Embed** the `content` field in batches of 64 using `sentence-transformers/all-MiniLM-L6-v2`. Output: 384-dimensional float32 vectors.
4. **L2-normalize** embeddings via `faiss.normalize_L2`. This makes inner-product search equivalent to cosine similarity.
5. **Add to FAISS `IndexFlatIP`**. Exact (brute-force) inner product search -- no approximation.
6. **Write aligned metadata** to a JSONL file (`make_metadata_record`). Row N in this file corresponds to row N in the FAISS index.
7. **Write FAISS index** to disk via `faiss.write_index`.

CLI usage:
```bash
python ingestion/build_faiss_index.py \
    --input data/processed/arxiv_ai_curated.jsonl \
    --index-output data/index/arxiv_ai.index \
    --metadata-output data/index/arxiv_ai_metadata.jsonl
```

### Legacy path: `ingestion/run.py`

Orchestrates: `load_arxiv_papers` -> `filter_papers` -> `build_documents` -> `chunk_documents` -> `embed_documents` -> `upsert_documents` (pgvector). This path references modules (`ingestion/chunker.py`, `pipeline/retriever.py`) that may not be present in the current codebase. The FAISS path above is the active retrieval backend.

### Supporting ingestion modules

| Module | Function | Purpose |
|---|---|---|
| `ingestion/load_arxiv.py` | `load_arxiv_papers(path)` | Loads JSONL/JSON/CSV, normalizes field names, returns `list[PaperRecord]` |
| `ingestion/filter_papers.py` | `filter_papers(papers, lookback_months)` | Filters by date window (default 9 months) and AI categories (`cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.RO`, `stat.ML`, `cs.HC`) with keyword fallback |
| `ingestion/topic_mapper.py` | `assign_topics(title, abstract)` | Regex-based topic assignment to 12 predefined topics; returns primary, secondary, and reason |
| `ingestion/build_documents.py` | `build_documents(papers)` | Converts `PaperRecord` to `DocumentRecord` with structured content, SHA-256 doc IDs, and topic metadata |
| `ingestion/embedder.py` | `embed_documents(documents)` | Batch-embeds `DocumentRecord.content` fields, attaches 384d vectors |
| `ingestion/embedder_optimized.py` | `embed_query(query)`, `warmup_embedder()` | Low-latency embedding for online queries; startup warmup to avoid first-query penalty |

---

## 3. Online Query Pipeline

Orchestrated by `pipeline/rag_pipeline.py::run_query()`. Called by both `api/main.py` (FastAPI) and `app/streamlit_app.py` (Streamlit).

**Step-by-step flow:**

1. **Normalize mode** (`_normalize_mode`): resolve to `"fast"` or `"full"`.
2. **Resolve parameters** (`_resolve_top_k`, `_resolve_retrieval_k`): apply mode-specific defaults and clamp to valid ranges.
3. **Embed query** (`ingestion.embedder.embed_query`): encode the query string into a 384d vector using MiniLM-L6-v2.
4. **Retrieve from FAISS** (`pipeline.retriever_faiss.search_documents`):
   - Load cached FAISS index and metadata (lazy-loaded, cached globally).
   - L2-normalize the query vector.
   - Search for `initial_k = max(top_k * 5, 50)` nearest neighbors.
   - Post-filter on metadata: exact match on `primary_topic_en`, substring match on `categories`, date lower bound on `published_date`.
   - Deduplicate by `paper_id`.
   - Return `top_k` results with cosine similarity scores.
5. **Rerank** (`pipeline.reranker.rerank`):
   - Score each (query, document.content) pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Sort descending by cross-encoder score.
   - Return top-k with `rerank_score` attached.
6. **Build prompt** (`pipeline.prompt.build_prompt` or `build_prompt_fast`):
   - Format document blocks: title, topic, date, relevance score, URL, truncated content.
   - Insert into query template with system instructions.
   - Full mode: up to 5 docs, 1200 chars each. Fast mode: up to 3 docs, 700 chars each.
7. **Generate answer** (`pipeline.generator.generate`):
   - Load GGUF model via `llama-cpp-python` (cached globally).
   - Truncate prompt to context char limit (12,000 or 3,500).
   - Call `create_chat_completion` if model supports chat format, else fall back to text completion.
   - Return generated text and model path.
8. **Format response**: assemble `QueryResponse` with answer, `SourceCitation` list, per-stage timings, and metadata.

**Empty result handling**: if FAISS returns zero matches, the pipeline short-circuits and returns a user-friendly message without calling the reranker or generator.

---

## 4. Component Breakdown

### Embedding

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384 dimensions)
- **Offline**: `ingestion/embedder.py::embed_texts()` encodes in batches of 64 with progress bar
- **Online**: `ingestion/embedder_optimized.py::embed_query()` encodes single queries without progress bar, with optional L2 normalization
- **Warmup**: `warmup_embedder()` called on API startup to avoid cold-start latency
- **Why MiniLM**: fast on CPU, well-benchmarked on semantic similarity, 384d is sufficient for abstract-level retrieval

### Retrieval (FAISS)

- **Index type**: `IndexFlatIP` (flat inner product). With L2-normalized vectors, inner product equals cosine similarity.
- **Module**: `pipeline/retriever_faiss.py`
- **Lazy loading**: index and metadata are loaded on first query and cached in module-level globals (`_index`, `_metadata`).
- **Over-retrieval**: searches for `max(top_k * 5, 50)` initial candidates to compensate for post-filtering losses.
- **Metadata filtering**: applied after FAISS search. Filters on `primary_topic_en` (exact), `categories` (substring), `published_date` (lower bound).
- **Deduplication**: by `paper_id` to prevent returning multiple records from the same paper.
- **Why not HNSW/IVF**: flat index is exact (no recall loss) and fast enough for the corpus sizes expected (tens of thousands of abstracts).

### Reranking

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Module**: `pipeline/reranker.py::rerank()`
- **Mechanism**: scores each (query, document) pair jointly. More expensive than bi-encoder but significantly more precise.
- **Lazy loading**: model loaded on first call, cached globally.
- **Output**: adds `rerank_score` field to each document dict, sorts descending, returns top-k.
- **Why cross-encoder**: bi-encoder embeddings optimize for recall (fast, approximate). Cross-encoder directly models query-document relevance for precision. The two-stage approach avoids scoring the full corpus with the slow cross-encoder.

### Generation (llama.cpp)

- **Module**: `pipeline/generator.py`
- **Backend**: `llama-cpp-python` wrapping the C++ llama.cpp library
- **Default model**: `qwen2.5-3b-instruct-q5_k_m.gguf` (Q5_K_M quantization)
- **Configuration**: environment-variable driven (`LLM_MODEL_PATH`, `LLM_N_CTX`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `LLM_N_THREADS`, `LLM_N_GPU_LAYERS`)
- **Chat vs text completion**: prefers `create_chat_completion` (system + user messages). Falls back to rendered plain-text prompt with `System:` / `User:` / `Assistant:` markers.
- **Mode-specific configs** (`_get_mode_config`): fast uses smaller context (2048), fewer tokens (96), lower temperature (0.1). Full uses larger context (4096), more tokens (384), slightly higher temperature (0.2).
- **Graceful failure**: on any exception, returns a fallback message rather than propagating the error.
- **Why GGUF/llama.cpp**: runs quantized models on CPU without GPU. Qwen 2.5-3B at Q5_K_M is small enough for CPU inference while still producing coherent instruction-following responses.

---

## 5. Data Flow Diagram

```
                         OFFLINE
                         ======

  arxiv_ai_curated.jsonl
          │
          ▼
  ┌───────────────────┐
  │ build_faiss_index  │
  │  • stream JSONL    │
  │  • embed (MiniLM)  │
  │  • L2 normalize    │
  │  • IndexFlatIP.add │
  └──────┬──────┬──────┘
         │      │
         ▼      ▼
  arxiv_ai    arxiv_ai_metadata
  .index      .jsonl


                         ONLINE
                         ======

  User Query (POST /query or Streamlit)
          │
          ▼
  ┌──────────────────┐
  │ embed_query       │  MiniLM-L6-v2 → 384d vector
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ search_documents  │  FAISS IndexFlatIP.search
  │  • initial_k      │  + metadata post-filters
  │  • dedup          │  + paper_id dedup
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ rerank            │  cross-encoder/ms-marco-MiniLM
  │  • top_k scored   │  (query, doc) pairs → rerank_score
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ build_prompt      │  Document blocks + system prompt
  │  (fast or full)   │  + query template
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ generate          │  llama.cpp (Qwen 2.5-3B GGUF)
  │  • chat or text   │  → synthesized answer
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ QueryResponse     │  answer + sources + timings
  └──────────────────┘
```

---

## 6. Fast vs Full Mode

Both modes use the same pipeline stages. They differ in parameter settings controlled by `pipeline/rag_pipeline.py` and `pipeline/generator.py`.

### Retrieval differences

| | Fast | Full |
|---|---|---|
| `retrieval_k` (candidates from FAISS) | 8 | 20 |
| `top_k` (final results after reranking) | 3 | 5 |
| `search_k` (FAISS initial search) | max(8*5, 50) = 50 | max(20*5, 50) = 100 |

### Prompt differences

| | Fast | Full |
|---|---|---|
| Docs in prompt | 3 | up to 5 |
| Char limit per doc | 700 | 1,200 |
| Template | `FAST_QUERY_TEMPLATE` (outline, max 110 words) | `QUERY_TEMPLATE` (narrative with trends + evidence + business relevance) |
| System prompt | `FAST_SYSTEM_PROMPT` (outline mode) | `SYSTEM_PROMPT` (synthesis mode) |

### Generation differences

| | Fast | Full |
|---|---|---|
| `n_ctx` | 2,048 | 4,096 |
| `max_tokens` | 96 | 384 |
| `temperature` | 0.1 | 0.2 |
| `top_p` | 0.85 | 0.9 |
| Context char limit | 3,500 | 12,000 |

Fast mode targets interactive use (seconds). Full mode targets richer synthesis (tens of seconds on CPU).

---

## 7. Evaluation Strategy

### Test set

`evaluation/testset.py` defines 25 `TestCase` dataclass instances, each with:
- `query`: business-style natural language question
- `expected_topic`: one of the 12 taxonomy topics
- `relevant_keywords`: hint terms for relevance assessment
- `description`: optional context

All 12 topics are covered. Questions are phrased for an investor/analyst audience.

### Metrics (`evaluation/metrics.py`)

| Metric | Function | Logic |
|---|---|---|
| Retrieval recall | `retrieval_hit_rate(retrieved_ids, relevant_ids)` | Set intersection / relevant count |
| Answer substance | `answer_non_empty(answer)` | >= 20 chars AND >= 3 words |
| Answer quality | `answer_structure_score(answer)` | Heuristic 0-1: length, paragraphs, specificity keywords, uncertainty markers |
| Latency check | `latency_acceptable(latency_ms, threshold_ms)` | Threshold comparison (default 10s) |
| Semantic similarity | `semantic_similarity_placeholder(answer, reference)` | Stub returning 0.0 (future BERTScore/RAGAS) |

A test case passes if `answer_non_empty` is true AND `latency_acceptable` is true.

### Evaluation runners

**`evaluation/simple_eval.py::run_evaluation()`**: runs all 25 queries in full mode (live) or validates test set structure (dry-run if pipeline unavailable). Outputs JSON summary with per-query status, metrics, and aggregate pass rate.

**`evaluation/smoke_eval.py::run_smoke_evaluation()`**: runs a configurable subset (default 5) in fast mode. Executes a non-measured warmup query first to isolate model loading from steady-state timing. Reports per-stage timings (embed, retrieve, rerank, prompt_build, generate) and wall-clock vs pipeline latency gap. Default latency threshold: 30 seconds.

---

## 8. Performance Considerations

### CPU Bottleneck

The system is designed for CPU-only execution. This is the primary performance constraint.

**Latency breakdown (typical, per query, on modern multi-core CPU):**

| Stage | Fast mode | Full mode |
|---|---|---|
| Embedding | ~30-80ms | ~30-80ms |
| FAISS search | ~5-20ms | ~10-30ms |
| Cross-encoder rerank | ~200-500ms | ~300-800ms |
| Prompt assembly | <5ms | <5ms |
| LLM generation | ~3-12s | ~10-40s |
| **Total** | **~4-15s** | **~12-45s** |

LLM generation dominates. The Qwen 2.5-3B model at Q5_K_M quantization is a deliberate balance: small enough for CPU inference, large enough for coherent instruction-following. Generation time scales linearly with `max_tokens`.

### Memory footprint

- FAISS `IndexFlatIP`: 384 floats * 4 bytes * N documents. For 50k papers: ~75 MB.
- Metadata JSONL: loaded entirely into memory as a list of dicts.
- SentenceTransformer (MiniLM): ~90 MB.
- CrossEncoder (ms-marco-MiniLM): ~90 MB.
- Llama.cpp model (Qwen 2.5-3B Q5_K_M): ~2.5 GB.
- Total working set: ~3-4 GB.

### Cold start mitigation

`api/main.py` calls `warmup_embedder()` and `warmup_retriever()` on startup. This preloads the embedding model, FAISS index, and metadata into memory. First real query avoids the model-loading penalty. The reranker and LLM are still lazy-loaded on first query.

### Scalability limits

- `IndexFlatIP` is O(N) per query (brute-force). Adequate for tens of thousands of abstracts. For millions, switch to `IndexIVFFlat` or `IndexHNSWFlat`.
- Single-threaded LLM generation. `LLM_N_THREADS=8` parallelizes within a single inference call, but concurrent requests will queue.
- No async generation. FastAPI endpoint blocks during LLM inference.

---

## 9. Tradeoffs and Design Decisions

**FAISS flat index vs approximate.** `IndexFlatIP` gives exact cosine similarity (no recall loss). For a corpus of ~10-50k abstracts, brute-force search completes in milliseconds. Approximate indexes (IVF, HNSW) add complexity and training requirements without meaningful latency improvement at this scale.

**Post-filtering vs pre-filtering.** Metadata filters (topic, category, date) are applied after FAISS search, not before. This ensures the system always has enough candidates to fill `top_k` even when filters are restrictive. The cost is retrieving more initial candidates (`search_k = max(top_k * 5, 50)`), which is negligible for flat search.

**Quantized 3B model vs larger model or API.** A 3B parameter model at Q5 quantization runs in seconds on CPU. Larger models (7B+) would improve answer quality but push latency to minutes. API-based models eliminate the latency problem but introduce cost, network dependency, and reproducibility concerns. The 3B local model is the sweet spot for a fully local, cost-free system.

**Keyword topic mapping vs LLM/embedding classification.** `ingestion/topic_mapper.py` uses regex patterns. This is deterministic, fast, and requires no model inference. The tradeoff is rigidity: papers that use novel terminology may be missed. Adding new topics requires updating the keyword list. For the 12 broad topics in the taxonomy, keyword coverage is adequate.

**Abstract-only retrieval vs full-text.** Abstracts are short, well-structured, and consistently available. Full-text parsing (LaTeX, PDF) is engineering-heavy and adds noise (references, boilerplate). For trend-level questions, abstracts contain sufficient signal. The `PaperRecord.full_text_path` field is reserved for future extension.

**Separate prompt templates per mode.** Fast mode (`FAST_QUERY_TEMPLATE`) constrains the LLM to a 110-word outline. Full mode (`QUERY_TEMPLATE`) requests structured narrative with trends, evidence, and business relevance. This is cheaper than runtime prompt negotiation and gives consistent output formats.

**Module-level global caching.** The FAISS index, metadata, embedding model, reranker, and LLM are all cached in module-level globals. This avoids repeated disk/model loading across requests. The tradeoff is that the process holds all models in memory simultaneously (~3-4 GB). For a single-server deployment, this is acceptable.

**Pydantic v2 models as the data contract.** `pipeline/models.py` defines strict Pydantic models (`PaperRecord`, `DocumentRecord`, `QueryRequest`, `QueryResponse`, `SourceCitation`, `QueryFilters`) shared across ingestion, pipeline, and API layers. This enforces type safety at boundaries and generates OpenAPI schemas automatically via FastAPI.
