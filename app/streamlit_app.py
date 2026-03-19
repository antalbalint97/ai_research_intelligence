import os
import time
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


FAST_DEFAULT_TOP_K = 3
FAST_DEFAULT_RETRIEVAL_K = 8

FULL_DEFAULT_TOP_K = 5
FULL_DEFAULT_RETRIEVAL_K = 20


st.set_page_config(
    page_title="AI Research Intelligence",
    page_icon="",
    layout="wide",
)

st.title("AI Research Intelligence")
st.caption("Local RAG pipeline over curated arXiv AI research abstracts")


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def format_score(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def render_sources(sources: list[Any]) -> None:
    if not sources:
        st.warning("No sources found.")
        return

    for i, src in enumerate(sources, start=1):
        title = get_value(src, "title", "Untitled")
        topic = get_value(src, "primary_topic") or get_value(src, "topic") or "N/A"
        published = get_value(src, "published_date", "N/A")
        score = (
            get_value(src, "relevance_score")
            or get_value(src, "score")
            or get_value(src, "similarity")
            or 0.0
        )
        url = get_value(src, "url") or get_value(src, "arxiv_abs_url")
        abstract = get_value(src, "abstract", "")

        with st.expander(f"[{i}] {title}", expanded=(i <= 2)):
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"**Topic:** {topic}")
            c2.markdown(f"**Published:** {published}")
            c3.markdown(f"**Score:** {format_score(score)}")

            if url:
                st.markdown(f"[Open paper]({url})")

            if abstract:
                st.markdown("**Abstract**")
                st.write(abstract)


def format_duration_ms(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "N/A"

    if value >= 1000:
        return f"{value / 1000:.1f} s"
    return f"{value:.1f} ms"


def render_metrics(response: Any, wall_clock_ms: float | None = None) -> None:
    latency_ms = get_value(response, "latency_ms", None)
    retrieval_count = get_value(response, "retrieval_count", 0)
    reranked_count = get_value(response, "reranked_count", 0)
    model = get_value(response, "model", "N/A")
    mode = get_value(response, "mode", "N/A")
    timings = get_value(response, "timings", {}) or {}
    prompt_chars = get_value(response, "prompt_chars", 0)
    answer_chars = get_value(response, "answer_chars", 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pipeline latency", format_duration_ms(latency_ms))
    col2.metric("Retrieved docs", retrieval_count)
    col3.metric("Reranked docs", reranked_count)
    col4.metric("Model", str(model))
    col5.metric("Mode", str(mode))

    if wall_clock_ms is not None:
        st.caption(f"Observed wall-clock time in UI: {format_duration_ms(wall_clock_ms)}")

    if timings:
        st.markdown("**Stage timings**")
        t1, t2, t3 = st.columns(3)
        t1.metric("Embed", format_duration_ms(timings.get("embed_ms")))
        t2.metric("Retrieve", format_duration_ms(timings.get("retrieve_ms")))
        t3.metric("Rerank", format_duration_ms(timings.get("rerank_ms")))

        t4, t5, t6 = st.columns(3)
        t4.metric("Prompt build", format_duration_ms(timings.get("prompt_build_ms")))
        t5.metric("Generate", format_duration_ms(timings.get("generate_ms")))
        t6.metric("Total", format_duration_ms(timings.get("total_ms")))

    s1, s2 = st.columns(2)
    s1.metric("Prompt chars", prompt_chars)
    s2.metric("Answer chars", answer_chars)


def build_filters(primary_topic: str | None, category: str | None, date_from: str | None):
    filters = {
        "primary_topic": primary_topic or None,
        "category": category or None,
        "date_from": date_from or None,
    }
    return {k: v for k, v in filters.items() if v is not None}


def resolve_query_params(
    query_mode: str,
    top_k: int,
    retrieval_k: int,
    use_mode_defaults: bool,
) -> tuple[int, int]:
    if not use_mode_defaults:
        return top_k, retrieval_k

    if query_mode == "fast":
        return FAST_DEFAULT_TOP_K, FAST_DEFAULT_RETRIEVAL_K

    return FULL_DEFAULT_TOP_K, FULL_DEFAULT_RETRIEVAL_K

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.header("Query Settings")

default_query = "What are the main trends in multimodal AI?"
query = st.sidebar.text_area("Query", value=default_query, height=100)

query_mode = st.sidebar.radio(
    "Response mode",
    options=["fast", "full"],
    index=0,
    help=(
        "Fast mode uses a shorter prompt and tighter generation budget for lower latency. "
        "Full mode aims for richer answers but may be much slower on CPU-only hardware."
    ),
)

use_mode_defaults = st.sidebar.checkbox(
    "Use recommended mode defaults",
    value=True,
    help="Automatically use lower retrieval settings in fast mode and standard settings in full mode.",
)

top_k = st.sidebar.slider("Final top_k", min_value=1, max_value=10, value=5)
retrieval_k = st.sidebar.slider("Initial retrieval_k", min_value=5, max_value=50, value=20, step=5)

effective_top_k, effective_retrieval_k = resolve_query_params(
    query_mode=query_mode,
    top_k=top_k,
    retrieval_k=retrieval_k,
    use_mode_defaults=use_mode_defaults,
)

st.sidebar.caption(
    f"Effective query settings → mode={query_mode}, top_k={effective_top_k}, retrieval_k={effective_retrieval_k}"
)

topic_options = [
    "",
    "Large Language Models",
    "Multimodal AI",
    "AI Agents",
    "Retrieval-Augmented Generation",
    "Reinforcement Learning",
    "Graph Neural Networks",
    "AI for Healthcare",
    "AI for Robotics",
    "AI Safety / Alignment",
    "Efficient AI / Model Compression",
    "Synthetic Data",
    "Foundation Models",
]
selected_topic = st.sidebar.selectbox("Primary topic filter", options=topic_options, index=0)
category_filter = st.sidebar.text_input("Category filter (optional)", value="", placeholder="e.g. cs.CL")
date_from = st.sidebar.text_input("Published after (optional)", value="", placeholder="YYYY-MM-DD")

show_debug = st.sidebar.checkbox("Show debug tab content", value=False)

run_query_btn = st.sidebar.button("Run query", type="primary")

# st.sidebar.markdown("---")
# st.sidebar.subheader("Evaluation Controls")

# eval_mode = st.sidebar.radio(
#     "Smoke eval mode",
#     options=["fast", "full"],
#     index=0,
#     help="Use fast smoke eval for development-time iteration.",
# )

# smoke_cases = st.sidebar.slider("Smoke eval cases", min_value=1, max_value=10, value=5)
# smoke_top_k = st.sidebar.slider("Smoke eval top_k", min_value=1, max_value=10, value=3)
# smoke_latency_threshold_ms = st.sidebar.number_input(
#     "Smoke latency threshold (ms)",
#     min_value=1000,
#     max_value=600000,
#     value=300000,
#     step=1000,
# )

# run_smoke_eval_btn = st.sidebar.button("Run smoke eval")
# run_eval_btn = st.sidebar.button("Run full evaluation suite")


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_query, tab_eval, tab_debug = st.tabs(["Query", "Evaluation", "Debug"])


# -------------------------------------------------------------------
# Query tab
# -------------------------------------------------------------------
with tab_query:
    st.subheader("Research question")
    st.write(query)

    if query_mode == "fast":
        st.info("Fast mode returns a shorter, more outline-style answer with lower latency.")
    else:
        st.info("Full mode aims for a richer answer and may be significantly slower on CPU-only hardware.")

    if run_query_btn:
        filters = build_filters(
            primary_topic=selected_topic,
            category=category_filter,
            date_from=date_from,
        )

        payload = {
            "query": query,
            "filters": filters or None,
            "top_k": effective_top_k,
            "retrieval_k": effective_retrieval_k,
            "mode": query_mode,
        }

        with st.spinner(f"Running {query_mode} retrieval, reranking, and generation..."):
            wall_start = time.time()
            try:
                api_response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=payload,
                    timeout=600,
                )
                api_response.raise_for_status()
                response = api_response.json()
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
                st.stop()

            wall_clock_ms = (time.time() - wall_start) * 1000

        st.subheader("Synthesized Answer")
        answer = get_value(response, "answer", "")
        if answer:
            st.markdown(answer)
        else:
            st.warning("No answer was generated.")

        st.subheader("System Metrics")
        render_metrics(response, wall_clock_ms=wall_clock_ms)

        st.subheader("Top Sources")
        sources = get_value(response, "sources", []) or []
        render_sources(sources)

        st.session_state["last_response"] = response
        st.session_state["last_query"] = query
        st.session_state["last_query_mode"] = query_mode
        st.session_state["last_effective_top_k"] = effective_top_k
        st.session_state["last_effective_retrieval_k"] = effective_retrieval_k
        st.session_state["last_wall_clock_ms"] = wall_clock_ms

    elif "last_response" in st.session_state:
        st.info("Showing previous result from this session.")
        response = st.session_state["last_response"]

        st.subheader("Synthesized Answer")
        st.markdown(get_value(response, "answer", "No answer available."))

        st.subheader("System Metrics")
        render_metrics(
            response,
            wall_clock_ms=st.session_state.get("last_wall_clock_ms"),
        )

        st.subheader("Top Sources")
        render_sources(get_value(response, "sources", []) or [])
    else:
        st.info("Set your query in the sidebar and click **Run query**.")

# -------------------------------------------------------------------
# Evaluation tab
# -------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation")
    st.info("Evaluation is currently run from the backend/CLI, not from the Streamlit UI in Docker.")

# with tab_eval:
#     st.subheader("Evaluation")
#     st.write(
#         "Use smoke evaluation for fast iteration and profiling, "
#         "or run the full evaluation suite for broader end-to-end testing."
#     )

#     eval_col1, eval_col2 = st.columns(2)

#     with eval_col1:
#         st.markdown("### Smoke evaluation")
#         st.caption(
#             "Fast development-time check over a small subset of queries. "
#             "Useful for profiling and optimization."
#         )

#     with eval_col2:
#         st.markdown("### Full evaluation")
#         st.caption(
#             "Runs the full manually-authored evaluation set. "
#             "This is slower on CPU-only generation."
#         )

#     if run_smoke_eval_btn:
#         with st.spinner(f"Running smoke evaluation in {eval_mode} mode..."):
#             try:
#                 summary = run_smoke_evaluation(
#                     output_path="data/artifacts/smoke_eval_results.json",
#                     max_cases=smoke_cases,
#                     top_k=smoke_top_k,
#                     latency_threshold_ms=float(smoke_latency_threshold_ms),
#                     mode=eval_mode,
#                 )
#             except TypeError:
#                 summary = run_smoke_evaluation(
#                     output_path="data/artifacts/smoke_eval_results.json",
#                     max_cases=smoke_cases,
#                     top_k=smoke_top_k,
#                     latency_threshold_ms=float(smoke_latency_threshold_ms),
#                 )
#                 summary["requested_mode"] = eval_mode

#         st.success("Smoke evaluation finished.")
#         st.session_state["last_smoke_eval_summary"] = summary

#     elif run_eval_btn:
#         with st.spinner("Running full evaluation suite... This may take a long time on CPU-only setup."):
#             summary = run_evaluation(output_path="data/artifacts/eval_results.json")

#         st.success("Full evaluation finished.")
#         st.session_state["last_full_eval_summary"] = summary

#     smoke_summary = st.session_state.get("last_smoke_eval_summary")
#     full_summary = st.session_state.get("last_full_eval_summary")

#     # ---------------------------
#     # Smoke eval display
#     # ---------------------------
#     st.markdown("---")
#     st.subheader("⚡ Latest smoke evaluation")

#     if smoke_summary:
#         timing = smoke_summary.get("timing_summary", {})

#         c1, c2, c3, c4, c5 = st.columns(5)
#         c1.metric("Mode", smoke_summary.get("mode", "N/A"))
#         c2.metric("Requested mode", smoke_summary.get("requested_mode", smoke_summary.get("config", {}).get("mode", "N/A")))
#         c3.metric("Pass rate", f"{smoke_summary.get('pass_rate', 0.0) * 100:.1f}%")
#         c4.metric("Passed", smoke_summary.get("passed", 0))
#         c5.metric("Failed", smoke_summary.get("failed", 0))

#         c6, c7, c8, c9 = st.columns(4)
#         c6.metric("Avg wall-clock (ms)", format_duration_ms(timing.get("avg_wall_clock_ms", 0.0)))
#         c7.metric("Median wall-clock (ms)", format_duration_ms(timing.get("median_wall_clock_ms", 0.0)))
#         c8.metric("Avg pipeline latency (ms)", format_duration_ms(timing.get("avg_pipeline_latency_ms", 0.0)))
#         c9.metric("Median pipeline latency (ms)", format_duration_ms(timing.get("median_pipeline_latency_ms", 0.0)))

#         results = smoke_summary.get("results", [])
#         if results:
#             st.subheader("Smoke per-query results")
#             st.dataframe(results, use_container_width=True)

#         with st.expander("Raw smoke evaluation summary"):
#             st.json(smoke_summary)
#     else:
#         st.info("No smoke evaluation run yet.")

#     # ---------------------------
#     # Full eval display
#     # ---------------------------
#     st.markdown("---")
#     st.subheader("Latest full evaluation")

#     if full_summary:
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Mode", full_summary.get("mode", "N/A"))
#         c2.metric("Pass rate", f"{full_summary.get('pass_rate', 0.0) * 100:.1f}%")
#         c3.metric("Passed", full_summary.get("passed", 0))
#         c4.metric("Avg latency (ms)", format_duration_ms(full_summary.get("avg_latency_ms", 0.0)))

#         results = full_summary.get("results", [])
#         if results:
#             st.subheader("Full evaluation per-query results")
#             st.dataframe(results, use_container_width=True)

#         with st.expander("Raw full evaluation summary"):
#             st.json(full_summary)
#     else:
#         st.info("No full evaluation run yet.")


# -------------------------------------------------------------------
# Debug tab
# -------------------------------------------------------------------
with tab_debug:
    st.subheader("Debug")

    if not show_debug:
        st.info("Enable **Show debug tab content** in the sidebar to view internals.")
    else:
        st.markdown("**Current environment**")
        st.code(
            "\n".join(
                [
                    f"API_BASE_URL = {API_BASE_URL}",
                    f"Selected query mode = {query_mode}",
                    f"Effective top_k = {effective_top_k}",
                    f"Effective retrieval_k = {effective_retrieval_k}",
                ]
            )
        )

        st.markdown("**Last query**")
        st.code(st.session_state.get("last_query", "No query run yet."))

        st.markdown("**Last response object**")
        if "last_response" in st.session_state:
            last_response = st.session_state["last_response"]
            if isinstance(last_response, dict):
                st.json(last_response)
            else:
                debug_payload = {
                    "answer": get_value(last_response, "answer"),
                    "mode": get_value(last_response, "mode", "N/A"),
                    "model": get_value(last_response, "model"),
                    "latency_ms": get_value(last_response, "latency_ms"),
                    "retrieval_count": get_value(last_response, "retrieval_count"),
                    "reranked_count": get_value(last_response, "reranked_count"),
                    "timings": get_value(last_response, "timings", {}),
                    "prompt_chars": get_value(last_response, "prompt_chars", 0),
                    "answer_chars": get_value(last_response, "answer_chars", 0),
                    "sources_count": len(get_value(last_response, "sources", []) or []),
                }
                st.json(debug_payload)
        else:
            st.write("No response available yet.")

        st.markdown("**Last smoke evaluation summary**")
        if "last_smoke_eval_summary" in st.session_state:
            st.json(st.session_state["last_smoke_eval_summary"])
        else:
            st.write("No smoke evaluation run yet.")

        st.markdown("**Last full evaluation summary**")
        if "last_full_eval_summary" in st.session_state:
            st.json(st.session_state["last_full_eval_summary"])
        else:
            st.write("No full evaluation run yet.")
