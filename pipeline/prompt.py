"""
Prompt templates for the AI Research Intelligence RAG system.

Includes:
- full mode for richer synthesis
- fast mode for low-latency outline-style answers
"""

from __future__ import annotations

SYSTEM_PROMPT = """
You are an AI Research Intelligence Assistant.

Your role is to help investors, deep-tech analysts, innovation teams, and consultants
understand recent trends in artificial intelligence research based only on retrieved paper abstracts.

Core rules:
- Use ONLY the provided research context.
- Do NOT invent facts, trends, results, or citations.
- Synthesize across multiple sources instead of summarizing only one paper.
- Prefer recurring patterns that appear across several papers.
- If the evidence is weak or fragmented, say so clearly.
- Explain technical topics in clear, business-friendly language.
- When relevant, comment briefly on commercial relevance, product implications, or strategic importance.
- Keep the answer concise but insightful.
""".strip()

FAST_SYSTEM_PROMPT = """
You are an AI Research Intelligence Assistant in fast-response mode.

Rules:
- Use ONLY the provided research context.
- Write an evidence-backed outline, not a long narrative.
- Prefer trends that appear across multiple papers.
- If the evidence is weak or narrow, say so clearly.
- Keep the answer practical for a business audience.
""".strip()

QUERY_TEMPLATE = """
You are given research context from multiple retrieved AI papers.

Your task:
1. Identify the main research trends relevant to the user's question.
2. Synthesize across the documents instead of describing papers one by one.
3. Highlight 3 to 5 major themes, methods, or directions if the context supports them.
4. Mention concrete examples from the papers when useful.
5. Note uncertainty when a trend appears in only one paper or is weakly supported.
6. End with a short business relevance or strategic implication section when appropriate.

Important constraints:
- Do not focus on a single paper unless the context is too narrow.
- Do not produce a generic answer detached from the sources.
- Do not claim "the field is moving toward X" unless the retrieved context supports it.
- If the retrieved papers are mixed or partially relevant, say that the answer is based on the retrieved sample.

Research Context:
---
{context}
---

User Question:
{query}

Return the answer in this structure:

Summary:
2-4 sentences answering the question directly.

Main Trends:
- Trend 1: ...
- Trend 2: ...
- Trend 3: ...
(Include up to 5 trends only if supported.)

Evidence from Retrieved Papers:
Briefly reference which papers or themes support the trends.

Business Relevance:
2-4 sentences on why these trends matter in practice.
""".strip()

FAST_QUERY_TEMPLATE = """
Use the short research context below to answer the business question quickly.

Constraints:
- Use only the provided context.
- Keep the answer brief and concrete.
- Maximum 110 words.
- Give an outline-style answer, not an essay.
- Mention uncertainty if the retrieved sample is narrow.
- Do not invent details.

Research Context:
---
{context}
---

User Question:
{query}

Return the answer in exactly this structure:

Top Trends:
- ...
- ...
- ...

Business Implication:
1-2 sentences.

Evidence Base:
- ...
- ...
- ...
""".strip()


def _get_doc_text(doc: dict, content_key: str = "content") -> str:
    return (doc.get(content_key) or doc.get("abstract") or "").strip()


def _format_doc_block(
    doc: dict,
    index: int,
    content_key: str = "content",
    include_url: bool = True,
    include_score: bool = True,
    char_limit: int | None = None,
) -> str:
    title = doc.get("title", "Untitled")
    topic = doc.get("primary_topic", "")
    published = doc.get("published_date", "")
    score = doc.get("rerank_score", doc.get("similarity", 0.0))
    url = doc.get("url") or doc.get("arxiv_abs_url") or ""
    content = _get_doc_text(doc, content_key=content_key)

    if char_limit is not None and char_limit > 0:
        content = content[:char_limit].rstrip()

    lines = [f"[{index}] Title: {title}"]

    if topic:
        lines.append(f"Topic: {topic}")
    if published:
        lines.append(f"Published: {published}")
    if include_score:
        lines.append(f"Relevance Score: {score:.2f}")
    if include_url and url:
        lines.append(f"URL: {url}")

    lines.append("Content:")
    lines.append(content)

    return "\n".join(lines)


def build_prompt(query: str, context_docs: list[dict], content_key: str = "content") -> str:
    """Assemble the full prompt from query and retrieved context documents."""
    context_blocks: list[str] = [
        _format_doc_block(
            doc,
            i,
            content_key=content_key,
            include_url=True,
            include_score=True,
            char_limit=1200,
        )
        for i, doc in enumerate(context_docs, start=1)
    ]

    context_text = "\n\n".join(context_blocks)
    return QUERY_TEMPLATE.format(context=context_text, query=query)


def build_prompt_fast(query: str, context_docs: list[dict], content_key: str = "content") -> str:
    """Assemble a shorter fast-mode prompt for evidence-backed outline responses."""
    context_blocks: list[str] = [
        _format_doc_block(
            doc,
            i,
            content_key=content_key,
            include_url=False,
            include_score=False,
            char_limit=700,
        )
        for i, doc in enumerate(context_docs[:3], start=1)
    ]

    context_text = "\n\n".join(context_blocks)
    return FAST_QUERY_TEMPLATE.format(context=context_text, query=query)


def get_system_prompt() -> str:
    return SYSTEM_PROMPT


def get_system_prompt_fast() -> str:
    return FAST_SYSTEM_PROMPT
