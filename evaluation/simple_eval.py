"""Simple evaluation runner for the AI Research RAG system.

Runs the evaluation test set against the pipeline and produces a summary report.
Designed for MVP deterministic evaluation; does not require external services.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from evaluation.metrics import (
    answer_non_empty,
    answer_structure_score,
    latency_acceptable,
)
from evaluation.testset import get_eval_test_set

logger = logging.getLogger(__name__)


def run_evaluation(output_path: str | None = None) -> dict:
    """Run the evaluation test set and produce a summary.

    If the full pipeline is available (database + models), runs live queries.
    Otherwise, performs a dry-run validating test set structure.

    Args:
        output_path: Optional path to write JSON results.

    Returns:
        Summary dict with pass/fail counts and scores.
    """
    test_set = get_eval_test_set()
    results: list[dict] = []
    live_mode = False

    # Try to import the pipeline for live evaluation
    try:
        from pipeline.rag_pipeline import run_query

        live_mode = True
        logger.info("Live evaluation mode: pipeline available")
    except Exception:
        logger.info("Dry-run evaluation mode: pipeline not available")

    total = len(test_set)
    passed = 0
    total_latency = 0.0

    for i, tc in enumerate(test_set, 1):
        result: dict = {
            "index": i,
            "query": tc.query,
            "expected_topic": tc.expected_topic,
            "status": "skip",
        }

        if live_mode:
            try:
                start = time.time()
                response = run_query(query=tc.query, top_k=5)
                elapsed = (time.time() - start) * 1000

                non_empty = answer_non_empty(response.answer)
                structure = answer_structure_score(response.answer)
                latency_ok = latency_acceptable(response.latency_ms)

                result.update(
                    {
                        "status": "pass" if non_empty and latency_ok else "fail",
                        "answer_non_empty": non_empty,
                        "structure_score": round(structure, 3),
                        "latency_ms": round(response.latency_ms, 1),
                        "latency_ok": latency_ok,
                        "model": response.model,
                        "retrieval_count": response.retrieval_count,
                        "reranked_count": response.reranked_count,
                    }
                )
                total_latency += response.latency_ms
                if non_empty and latency_ok:
                    passed += 1
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                logger.warning("Test %d failed: %s", i, e)
        else:
            # Dry-run: validate test case structure
            result["status"] = "dry_run"
            result["valid_structure"] = bool(tc.query and tc.expected_topic)
            if tc.query and tc.expected_topic:
                passed += 1

        results.append(result)
        logger.info("[%d/%d] %s – %s", i, total, tc.expected_topic, result["status"])

    summary = {
        "mode": "live" if live_mode else "dry_run",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
        "avg_latency_ms": round(total_latency / total, 1) if live_mode and total > 0 else 0.0,
        "results": results,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Evaluation results written to %s", output_path)

    return summary


def main() -> None:
    """CLI entry point for evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    summary = run_evaluation(output_path="data/artifacts/eval_results.json")
    print(f"\nEvaluation complete: {summary['passed']}/{summary['total']} passed "
          f"({summary['pass_rate']:.0%}) [{summary['mode']} mode]")


if __name__ == "__main__":
    main()
