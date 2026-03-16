#!/usr/bin/env bash
# Evaluation script – runs the evaluation test set
set -euo pipefail

echo "=== AI Research RAG – Evaluation ==="

cd "$(dirname "$0")/.."
python -m evaluation.simple_eval

echo "=== Evaluation complete ==="
