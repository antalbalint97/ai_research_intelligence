#!/usr/bin/env bash
# Ingestion script – runs the full offline pipeline
set -euo pipefail

echo "=== AI Research RAG – Ingestion ==="
echo "Data path: ${ARXIV_DATA_PATH:-data/raw/arxiv-metadata.jsonl}"
echo "Lookback:  ${LOOKBACK_MONTHS:-9} months"

cd "$(dirname "$0")/.."
python -m ingestion.run

echo "=== Ingestion complete ==="
