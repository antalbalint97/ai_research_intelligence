"""
Local text generator using llama-cpp-python + GGUF quantized instruct models.

Supports two inference modes:
- full: better answer depth, higher latency
- fast: lower latency, shorter outputs
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_BACKEND = os.environ.get("LLM_BACKEND", "llama_cpp")

DEFAULT_MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH", "models/qwen2.5-3b-instruct-q5_k_m.gguf"
)
FAST_MODEL_PATH = os.environ.get("LLM_MODEL_PATH_FAST", DEFAULT_MODEL_PATH)

DEFAULT_N_CTX = int(os.environ.get("LLM_N_CTX", "4096"))
FAST_N_CTX = int(os.environ.get("LLM_N_CTX_FAST", "2048"))

DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "384"))
FAST_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS_FAST", "96"))

DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
FAST_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE_FAST", "0.1"))

DEFAULT_TOP_P = float(os.environ.get("LLM_TOP_P", "0.9"))
FAST_TOP_P = float(os.environ.get("LLM_TOP_P_FAST", "0.85"))

DEFAULT_N_THREADS = int(os.environ.get("LLM_N_THREADS", "8"))
DEFAULT_N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", "0"))

DEFAULT_CONTEXT_CHAR_LIMIT = int(os.environ.get("LLM_CONTEXT_CHAR_LIMIT", "12000"))
FAST_CONTEXT_CHAR_LIMIT = int(os.environ.get("LLM_CONTEXT_CHAR_LIMIT_FAST", "3500"))

DEFAULT_VERBOSE = os.environ.get("LLM_VERBOSE", "false").lower() in ("1", "true", "yes")

_llm = None
_loaded_model_path: str | None = None


def _get_mode_config(mode: str) -> dict[str, Any]:
    mode = (mode or "full").strip().lower()
    if mode == "fast":
        return {
            "mode": "fast",
            "model_path": FAST_MODEL_PATH,
            "n_ctx": FAST_N_CTX,
            "max_tokens": FAST_MAX_TOKENS,
            "temperature": FAST_TEMPERATURE,
            "top_p": FAST_TOP_P,
            "context_char_limit": FAST_CONTEXT_CHAR_LIMIT,
        }

    return {
        "mode": "full",
        "model_path": DEFAULT_MODEL_PATH,
        "n_ctx": DEFAULT_N_CTX,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "context_char_limit": DEFAULT_CONTEXT_CHAR_LIMIT,
    }


def _truncate_prompt(prompt: str, max_chars: int) -> str:
    prompt = (prompt or "").strip()
    if len(prompt) <= max_chars:
        return prompt

    logger.warning(
        "Prompt too long (%d chars), truncating to %d chars",
        len(prompt),
        max_chars,
    )
    return prompt[:max_chars].rstrip()


def _fallback_answer(mode: str) -> str:
    if (mode or "").lower() == "fast":
        return (
            "The local fast mode could not produce a reliable short answer from the "
            "retrieved context. Please review the sources or try a narrower question."
        )

    return (
        "I could not generate a sufficiently strong synthesized answer from the retrieved "
        "context on the current local setup. Please review the retrieved sources directly "
        "or try a more specific query."
    )


def _load_llama_cpp(model_path: str, n_ctx: int):
    global _llm, _loaded_model_path

    if _llm is not None and _loaded_model_path == model_path:
        return _llm, _loaded_model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GGUF model file not found: {model_path}. "
            "Set LLM_MODEL_PATH or LLM_MODEL_PATH_FAST to your local GGUF file."
        )

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required. Install with: pip install llama-cpp-python"
        ) from exc

    logger.info("Loading GGUF model with llama.cpp: %s", model_path)
    logger.info(
        "llama.cpp config: n_ctx=%d, n_threads=%d, n_gpu_layers=%d",
        n_ctx,
        DEFAULT_N_THREADS,
        DEFAULT_N_GPU_LAYERS,
    )

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=DEFAULT_N_THREADS,
        n_gpu_layers=DEFAULT_N_GPU_LAYERS,
        verbose=DEFAULT_VERBOSE,
    )

    _llm = llm
    _loaded_model_path = model_path
    logger.info("GGUF model loaded successfully")
    return _llm, _loaded_model_path


def _supports_chat_format(llm: Any) -> bool:
    return hasattr(llm, "create_chat_completion")


def _render_fallback_prompt(system_prompt: str, prompt: str) -> str:
    system_prompt = (system_prompt or "").strip()
    prompt = (prompt or "").strip()

    if system_prompt:
        return f"System:\n{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"

    return f"User:\n{prompt}\n\nAssistant:\n"


def generate_with_llama_cpp(
    prompt: str,
    system_prompt: str = "",
    mode: str = "full",
) -> tuple[str, str]:
    """Generate an answer using llama-cpp-python from a local GGUF model."""
    cfg = _get_mode_config(mode)
    llm, loaded_model_path = _load_llama_cpp(
        model_path=cfg["model_path"],
        n_ctx=cfg["n_ctx"],
    )
    prompt = _truncate_prompt(prompt, max_chars=cfg["context_char_limit"])

    if _supports_chat_format(llm):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        logger.info("Generating response via llama.cpp chat completion | mode=%s", cfg["mode"])
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )

        text = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return text, loaded_model_path

    rendered_prompt = _render_fallback_prompt(system_prompt=system_prompt, prompt=prompt)

    logger.info("Generating response via llama.cpp text completion | mode=%s", cfg["mode"])
    response = llm(
        rendered_prompt,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        stop=["User:", "System:"],
    )

    text = response.get("choices", [{}])[0].get("text", "").strip()
    return text, loaded_model_path


def generate(prompt: str, system_prompt: str = "", mode: str = "full") -> tuple[str, str]:
    """Main generator entry point used by the RAG pipeline."""
    backend = DEFAULT_BACKEND.lower().strip()
    mode = (mode or "full").strip().lower()

    try:
        if backend != "llama_cpp":
            raise ValueError(
                f"Unsupported backend: {backend}. "
                "This generator.py is configured for llama_cpp."
            )

        text, model_name = generate_with_llama_cpp(
            prompt=prompt,
            system_prompt=system_prompt,
            mode=mode,
        )

        if text:
            return text, model_name

        logger.warning("Generator returned empty text | mode=%s", mode)
        return _fallback_answer(mode), model_name

    except Exception as exc:
        logger.exception("Local generation failed | mode=%s: %s", mode, exc)
        return _fallback_answer(mode), "unavailable"
