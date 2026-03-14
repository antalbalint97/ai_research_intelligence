"""Text generator – produces answers from context using LLM inference.

Primary: Mistral-7B-Instruct-v0.2 via Hugging Face Inference API.
Fallback: google/flan-t5-base locally via transformers pipeline.

Implements robust timeout handling and automatic fallback on API errors.
"""

from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TIMEOUT = 60  # seconds
MISTRAL_MODEL_NAME = "mistral-7b-instruct-v0.2"
FLAN_MODEL_NAME = "google/flan-t5-base"

# Lazy-loaded local fallback model
_local_pipeline = None


def _get_hf_token() -> str | None:
    """Retrieve Hugging Face API token from environment."""
    return os.environ.get("HF_API_TOKEN")


def _should_fallback() -> bool:
    """Check if local fallback is enabled."""
    return os.environ.get("FALLBACK_TO_FLAN", "true").lower() in ("true", "1", "yes")


def _get_local_pipeline():
    """Lazy-load the local flan-t5-base pipeline."""
    global _local_pipeline
    if _local_pipeline is None:
        try:
            from transformers import pipeline

            logger.info("Loading local fallback model: %s", FLAN_MODEL_NAME)
            _local_pipeline = pipeline(
                "text2text-generation",
                model=FLAN_MODEL_NAME,
                max_new_tokens=512,
            )
        except ImportError:
            raise ImportError(
                "transformers is required for local fallback. "
                "Install with: pip install transformers"
            )
    return _local_pipeline


def generate_with_hf_api(prompt: str, system_prompt: str = "") -> tuple[str, str]:
    """Generate a response using the Hugging Face Inference API.

    Args:
        prompt: The assembled prompt with context and query.
        system_prompt: System-level instruction for the model.

    Returns:
        Tuple of (generated_text, model_name).

    Raises:
        requests.HTTPError: On non-retryable API errors.
        requests.Timeout: On timeout.
    """
    token = _get_hf_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Format for Mistral instruct
    full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.3,
            "top_p": 0.9,
            "return_full_text": False,
        },
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=HF_TIMEOUT)
    response.raise_for_status()

    result = response.json()
    if isinstance(result, list) and result:
        text = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        text = result.get("generated_text", "")
    else:
        text = str(result)

    return text.strip(), MISTRAL_MODEL_NAME


def generate_with_local(prompt: str, system_prompt: str = "") -> tuple[str, str]:
    """Generate a response using the local flan-t5-base model.

    Args:
        prompt: The assembled prompt.
        system_prompt: System-level instruction (prepended to prompt).

    Returns:
        Tuple of (generated_text, model_name).
    """
    pipe = _get_local_pipeline()
    full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    # Truncate input to avoid OOM on small models
    max_input_chars = 2048
    if len(full_input) > max_input_chars:
        full_input = full_input[:max_input_chars]

    outputs = pipe(full_input)
    text = outputs[0]["generated_text"] if outputs else ""
    return text.strip(), FLAN_MODEL_NAME


def generate(prompt: str, system_prompt: str = "") -> tuple[str, str]:
    """Generate an answer with automatic fallback.

    Tries the HF Inference API first. On 429/503/timeout errors, falls back
    to local flan-t5-base if FALLBACK_TO_FLAN is enabled.

    Args:
        prompt: Assembled prompt with context and query.
        system_prompt: System-level instruction.

    Returns:
        Tuple of (generated_text, model_name).
    """
    # Try HF API first
    try:
        text, model = generate_with_hf_api(prompt, system_prompt)
        if text:
            return text, model
        logger.warning("HF API returned empty response, trying fallback")
    except requests.exceptions.Timeout:
        logger.warning("HF API timed out after %ds", HF_TIMEOUT)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        if status in (429, 503):
            logger.warning("HF API returned %d, trying fallback", status)
        else:
            logger.error("HF API error: %s", e)
    except requests.exceptions.ConnectionError:
        logger.warning("HF API connection failed, trying fallback")
    except Exception as e:
        logger.error("Unexpected HF API error: %s", e)

    # Fallback to local model
    if _should_fallback():
        logger.info("Falling back to local model: %s", FLAN_MODEL_NAME)
        try:
            return generate_with_local(prompt, system_prompt)
        except Exception as e:
            logger.error("Local fallback also failed: %s", e)

    return (
        "I'm sorry, the generation service is temporarily unavailable. "
        "Please try again in a moment.",
        "unavailable",
    )
