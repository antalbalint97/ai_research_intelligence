# AI Research Intelligence RAG

> A RAG-based assistant for exploring recent AI research trends and translating them into strategic insights for investors and innovation teams.

---

## Executive Summary

**AI Research Intelligence RAG** is a containerized Retrieval-Augmented Generation (RAG) Q&A system designed to help business stakeholders understand recent AI research trends. It ingests arXiv paper metadata, organizes it into a topic taxonomy, and answers natural-language questions with clear, investor-friendly explanations backed by source citations.

The system is built for:
- **Investors** evaluating deep-tech opportunities
- **Deep-tech analysts** tracking emerging methods
- **Innovation / strategy teams** monitoring technology landscapes
- **Consulting teams** researching AI capabilities for client engagements

Instead of simply retrieving papers, the system **synthesizes findings** and explains their commercial relevance in plain business language.

---

## Business Relevance

Scientific-paper intelligence supports critical business workflows:

| Use Case | Value |
|---|---|
| **Investment Research** | Identify accelerating AI subfields before they become mainstream; detect which methods are gaining traction in papers before they appear in products |
| **Deep-Tech Scouting** | Systematic monitoring of emerging techniques across computer vision, NLP, robotics, and more |
| **Corporate Innovation** | Understand which AI capabilities are maturing and could be adopted internally |
| **Technology Landscape Analysis** | Map the competitive dynamics of AI research by topic, method, and institution |

---

## Architecture Overview

The system follows a 5-layer RAG architecture with clear separation between **offline** (data ingestion) and **online** (query) workloads:

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Data Ingestion                        │
│  load_arxiv → filter → topic_mapper → build_doc │
├─────────────────────────────────────────────────┤
│  Layer 2: Chunking + Embedding                  │
│  chunker → embedder (MiniLM-L6-v2)             │
├─────────────────────────────────────────────────┤
│  Layer 3: Retrieval + Reranking                 │
│  pgvector cosine search → cross-encoder rerank  │
├─────────────────────────────────────────────────┤
│  Layer 4: Generation                            │
│  Mistral-7B (HF API) → flan-t5-base fallback   │
├─────────────────────────────────────────────────┤
│  Layer 5: API + Frontend                        │
│  FastAPI → lightweight HTML/JS/CSS UI           │
└─────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|---|---|---|
| Vector Store | PostgreSQL + pgvector | Semantic search with SQL metadata filtering |
| Embeddings | `all-MiniLM-L6-v2` (384d) | Fast, high-quality sentence embeddings |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder precision reranking |
| Generator (primary) | Mistral-7B-Instruct via HF API | High-quality answer generation |
| Generator (fallback) | flan-t5-base (local) | Offline/rate-limit fallback |
| API | FastAPI | REST endpoints with OpenAPI docs |
| Frontend | Vanilla HTML/JS/CSS | Lightweight chat-like interface |

---

## Design Decisions & Trade-offs

### Why pgvector over FAISS?
- Production-grade: supports SQL metadata filtering alongside vector search
- Single Docker service (PostgreSQL) for both structured and vector data
- Persistent storage with ACID guarantees
- HNSW index for fast approximate nearest-neighbor search

### Why bare Python + minimal frameworking?
- Explicit, readable code over framework magic
- Easier to debug, test, and explain
- Each component is independently understandable
- Interview-ready: every design choice is intentional

### Why MiniLM embeddings?
- 384-dimensional vectors: excellent quality-to-size ratio
- Fast inference on CPU (no GPU required for embedding)
- Well-benchmarked on semantic similarity tasks

### Why cross-encoder reranking?
- Bi-encoders (embeddings) optimize for recall; cross-encoders optimize for precision
- Two-stage retrieval: fast recall → precise reranking = best of both worlds

### Why Mistral via HF API + local fallback?
- Mistral-7B-Instruct produces high-quality, instruction-following answers
- HF Inference API avoids GPU infrastructure requirements
- Automatic fallback to flan-t5-base ensures the system never completely fails

### Why abstract-first corpus (no full PDF for MVP)?
- Title + abstract captures 80%+ of a paper's key signal
- Full PDF parsing is complex (LaTeX, figures, tables) and not needed for trend analysis
- Architecture is extensible: `full_text_path` field is reserved for future use

---

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local development)
- A Hugging Face API token (optional, for Mistral-7B)

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd ai-research-rag

# 2. Set up environment
cp .env.example .env
# Edit .env with your HF_API_TOKEN and data path

# 3. Start services
docker compose up -d

# 4. Run ingestion (with arXiv data file in data/raw/)
python -m ingestion.run

# 5. Query the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the recent trends in multimodal AI?"}'

# 6. Open the frontend
open http://localhost:8000/
```

### Data Format

Place your arXiv data file in `data/raw/`. Supported formats:
- `.jsonl` / `.ndjson` – one JSON object per line
- `.json` – JSON array of objects
- `.csv` – CSV with header row

Expected fields (flexible naming):
- `id` – arXiv paper ID
- `title` – paper title
- `abstract` – paper abstract
- `authors` – author names
- `categories` – arXiv category codes (space-separated)
- `update_date` / `published` / `updated` – publication date

---

## Example Queries

```
What are the recent trends in multimodal AI?
What methods are becoming popular in AI agents research?
What are the limitations of current RAG research?
What commercially promising directions are emerging in robotics AI?
How is AI being used in healthcare research recently?
What instruction tuning methods are gaining traction in LLM research?
How effective are current jailbreak defense methods for LLMs?
What quantization techniques are most effective for deploying LLMs?
```

---

## Topic Taxonomy

The system maps papers to a predefined AI topic taxonomy:

| Topic (EN) | Translation (HU) |
|---|---|
| Large Language Models | Nagy nyelvi modellek |
| Multimodal AI | Multimodális mesterséges intelligencia |
| AI Agents | AI ügynökök / autonóm ügynökök |
| Retrieval-Augmented Generation | Visszakereséssel bővített generálás |
| Reinforcement Learning | Megerősítéses tanulás |
| Graph Neural Networks | Gráf neurális hálók |
| AI for Healthcare | AI az egészségügyben |
| AI for Robotics | AI a robotikában |
| AI Safety / Alignment | AI biztonság és igazítás |
| Efficient AI / Model Compression | Hatékony AI / modellkompresszió |
| Synthetic Data | Szintetikus adatgenerálás |
| Foundation Models | Alapmodellek |

---

## Evaluation Approach

The evaluation pipeline provides lightweight quality checks:

1. **Retrieval hit-rate** – fraction of relevant documents retrieved
2. **Answer non-emptiness** – structural validation of generated answers
3. **Answer quality score** – heuristic scoring based on length, structure, and specificity
4. **Latency check** – response time within acceptable bounds

A manually-authored test set of 25 business-style questions spans all major AI topics. Run evaluation:

```bash
python -m evaluation.simple_eval
```

---

## Repository Structure

```
ai-research-rag/
├── api/                 # FastAPI application
├── ingestion/           # Data loading, filtering, topic mapping, chunking, embedding
├── pipeline/            # Retriever, reranker, generator, prompt templates, data models
├── evaluation/          # Metrics, test set, evaluation runner
├── frontend/            # Lightweight HTML/JS/CSS chat interface
├── scripts/             # Shell scripts for ingestion, indexing, evaluation
├── tests/               # Unit tests
├── data/                # Raw, processed, and artifact directories
├── docs/                # Architecture documentation
├── docker-compose.yml   # Container orchestration
├── Dockerfile           # API service image
├── requirements.txt     # Python dependencies
└── pyproject.toml       # Project and tool configuration
```

---

## Future Improvements

- **Full-text PDF extraction** – Parse LaTeX/PDF for deeper content
- **Trend dashboards** – Visualize topic momentum over time
- **Topic clustering** – Unsupervised discovery of emerging sub-topics
- **Institution / author analytics** – Track which labs and researchers are leading
- **Scheduled digests** – Weekly email or Slack summaries of new research
- **RAGAS evaluation** – Integrate RAGAS for more sophisticated RAG evaluation
- **Multi-language support** – Translate insights for non-English stakeholders
- **Fine-tuned embeddings** – Domain-adapted embeddings for AI research terminology

---

## License

MIT
