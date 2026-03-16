"""Prompt templates for the AI Research Intelligence RAG system.

Defines the system prompt and context-assembly logic for generating
investor-friendly, business-relevant answers about AI research trends.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an AI Research Intelligence Assistant. Your role is to help
investors, deep-tech analysts, innovation teams, and consultants understand recent trends
in artificial intelligence research.

Instructions:
- Answer ONLY based on the provided research context below.
- Explain technical concepts in clear, business-friendly language.
- Highlight emerging trends and their potential commercial relevance.
- If the context is insufficient to answer fully, say so honestly.
- Include brief business relevance commentary when possible.
- Mention specific papers or findings from the context to support your answer.
- Be concise but thorough. Aim for 3-6 paragraphs.
- Do NOT fabricate information not present in the context.
"""

QUERY_TEMPLATE = """Research Context:
---
{context}
---

User Question: {query}

Please provide a clear, well-structured answer based on the research context above.
Focus on trends, methods, and commercial implications."""


def build_prompt(query: str, context_docs: list[dict], content_key: str = "content") -> str:
    """Assemble the full prompt from query and retrieved context documents.

    Args:
        query: User's natural-language question.
        context_docs: List of retrieved/reranked document dicts.
        content_key: Key in document dict containing the text.

    Returns:
        Fully assembled prompt string.
    """
    context_blocks: list[str] = []
    for i, doc in enumerate(context_docs, 1):
        title = doc.get("title", "Untitled")
        content = doc.get(content_key, "")
        score = doc.get("rerank_score", doc.get("similarity", 0.0))
        context_blocks.append(f"[{i}] {title} (relevance: {score:.2f})\n{content}")

    context_text = "\n\n".join(context_blocks)
    return QUERY_TEMPLATE.format(context=context_text, query=query)


def get_system_prompt() -> str:
    """Return the system prompt for the generator."""
    return SYSTEM_PROMPT.strip()
