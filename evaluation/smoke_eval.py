"""Fast smoke evaluation runner for the AI Research RAG system.

Goal:
- run only a small subset of test queries
- perform one non-measured warmup query first
- measure steady-state timing per query
- keep evaluation cheap and informative during development
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    answer_non_empty,
    answer_structure_score,
    latency_acceptable,
)
from evaluation.testset import get_eval_test_set

logger = logging.getLogger(__name__)


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _truncate(text: str, limit: int = 200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def run_smoke_evaluation(
    output_path: str | None = None,
    max_cases: int = 5,
    top_k: int = 3,
    retrieval_k: int | None = 8,
    mode: str = "fast",
    latency_threshold_ms: float = 30000.0,
    run_warmup: bool = True,
    warmup_query: str = "What are the recent trends in multimodal AI research?",
) -> dict:
    """Run a small live smoke evaluation against the RAG pipeline.

    Args:
        output_path: Optional JSON output file.
        max_cases: Number of test queries to run from the test set.
        top_k: Final reranked result count.
        retrieval_k: Initial retrieval candidate count.
        mode: Query mode, e.g. 'fast' or 'full'.
        latency_threshold_ms: Steady-state acceptable latency threshold.
        run_warmup: Whether to execute one non-measured warmup query first.
        warmup_query: Query used for warmup.

    Returns:
        Summary dict.
    """
    test_set = get_eval_test_set()[:max_cases]
    results: list[dict] = []

    try:
        from pipeline.rag_pipeline import run_query
    except Exception as e:
        raise RuntimeError(
            "Smoke eval requires live pipeline import: pipeline.rag_pipeline.run_query"
        ) from e

    warmup_info: dict[str, Any] | None = None

    if run_warmup:
        logger.info("Running non-measured warmup query before smoke evaluation")
        try:
            warmup_start = time.perf_counter()
            warmup_response = run_query(
                query=warmup_query,
                top_k=top_k,
                retrieval_k=retrieval_k,
                mode=mode,
            )
            warmup_wall_clock_ms = (time.perf_counter() - warmup_start) * 1000
            warmup_info = {
                "ran": True,
                "query": warmup_query,
                "wall_clock_ms": round(warmup_wall_clock_ms, 1),
                "pipeline_latency_ms": round(
                    float(_safe_getattr(warmup_response, "latency_ms", warmup_wall_clock_ms)), 1
                ),
                "model": _safe_getattr(warmup_response, "model", "unknown"),
                "mode": mode,
            }
        except Exception as e:
            warmup_info = {
                "ran": False,
                "query": warmup_query,
                "error": str(e),
                "mode": mode,
            }
            logger.exception("Warmup run failed")

    total = len(test_set)
    passed = 0
    passed_latency = 0
    wall_clock_latencies: list[float] = []
    pipeline_latencies: list[float] = []

    for i, tc in enumerate(test_set, 1):
        logger.info("Running smoke test %d/%d: %s", i, total, tc.query)

        row: dict[str, Any] = {
            "index": i,
            "query": tc.query,
            "expected_topic": tc.expected_topic,
            "status": "error",
        }

        try:
            wall_start = time.perf_counter()
            response = run_query(
                query=tc.query,
                top_k=top_k,
                retrieval_k=retrieval_k,
                mode=mode,
            )
            wall_elapsed_ms = (time.perf_counter() - wall_start) * 1000

            answer = _safe_getattr(response, "answer", "")
            pipeline_latency_ms = _safe_getattr(response, "latency_ms", wall_elapsed_ms)
            retrieval_count = _safe_getattr(response, "retrieval_count", None)
            reranked_count = _safe_getattr(response, "reranked_count", None)
            model = _safe_getattr(response, "model", "unknown")
            timings = _safe_getattr(response, "timings", {}) or {}
            prompt_chars = _safe_getattr(response, "prompt_chars", 0)
            answer_chars = _safe_getattr(response, "answer_chars", len((answer or "").strip()))

            non_empty = answer_non_empty(answer)
            structure_score = answer_structure_score(answer)
            latency_ok = latency_acceptable(
                float(pipeline_latency_ms), threshold_ms=latency_threshold_ms
            )

            status = "pass" if (non_empty and latency_ok) else "fail"

            if status == "pass":
                passed += 1
            if latency_ok:
                passed_latency += 1

            row.update(
                {
                    "status": status,
                    "answer_non_empty": non_empty,
                    "structure_score": round(structure_score, 3),
                    "latency_ok": latency_ok,
                    "wall_clock_ms": round(wall_elapsed_ms, 1),
                    "pipeline_latency_ms": round(float(pipeline_latency_ms), 1),
                    "latency_gap_ms": round(abs(float(pipeline_latency_ms) - wall_elapsed_ms), 1),
                    "embed_ms": timings.get("embed_ms"),
                    "retrieve_ms": timings.get("retrieve_ms"),
                    "rerank_ms": timings.get("rerank_ms"),
                    "prompt_build_ms": timings.get("prompt_build_ms"),
                    "generate_ms": timings.get("generate_ms"),
                    "total_ms": timings.get("total_ms"),
                    "model": model,
                    "mode": mode,
                    "retrieval_count": retrieval_count,
                    "reranked_count": reranked_count,
                    "prompt_chars": prompt_chars,
                    "answer_chars": answer_chars,
                    "answer_preview": _truncate(answer, 240),
                }
            )

            wall_clock_latencies.append(wall_elapsed_ms)
            pipeline_latencies.append(float(pipeline_latency_ms))

        except Exception as e:
            row["error"] = str(e)
            logger.exception("Smoke test %d failed", i)

        results.append(row)

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    def _median(values: list[float]) -> float:
        return round(statistics.median(values), 1) if values else 0.0

    summary = {
        "mode": "smoke_live",
        "query_mode": mode,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "latency_passed": passed_latency,
        "latency_pass_rate": round(passed_latency / total, 3) if total else 0.0,
        "config": {
            "max_cases": max_cases,
            "top_k": top_k,
            "retrieval_k": retrieval_k,
            "mode": mode,
            "latency_threshold_ms": latency_threshold_ms,
            "run_warmup": run_warmup,
        },
        "warmup": warmup_info,
        "timing_summary": {
            "avg_wall_clock_ms": _avg(wall_clock_latencies),
            "median_wall_clock_ms": _median(wall_clock_latencies),
            "avg_pipeline_latency_ms": _avg(pipeline_latencies),
            "median_pipeline_latency_ms": _median(pipeline_latencies),
        },
        "results": results,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Smoke eval results written to %s", output_path)

    return summary