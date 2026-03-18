FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Optional runtime defaults; override in docker-compose or docker run
ENV LLM_MODEL_PATH=/app/models/qwen2.5-3b-instruct-q5_k_m.gguf
ENV LLM_MODEL_PATH_FAST=/app/models/qwen2.5-3b-instruct-q5_k_m.gguf
ENV LLM_N_THREADS=8
ENV LLM_MAX_TOKENS_FAST=96
ENV LLM_MAX_TOKENS_FULL=256
ENV LLM_N_CTX_FAST=2048
ENV LLM_N_CTX_FULL=4096

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]