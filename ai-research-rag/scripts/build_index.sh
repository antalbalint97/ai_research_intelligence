#!/usr/bin/env bash
# Build index – ensures database schema and HNSW index exist
set -euo pipefail

echo "=== AI Research RAG – Build Index ==="

cd "$(dirname "$0")/.."
python -c "
from pipeline.retriever import ensure_schema
ensure_schema()
print('Database schema and HNSW index created successfully.')
"

echo "=== Index build complete ==="
