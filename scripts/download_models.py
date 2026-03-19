from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

QWEN_TARGET = MODELS_DIR / "qwen2.5-3b-instruct-q5_k_m.gguf"
EMBED_DIR = MODELS_DIR / "embeddings" / "all-MiniLM-L6-v2"
RERANK_DIR = MODELS_DIR / "rerankers" / "ms-marco-MiniLM-L-6-v2"

QWEN_REPO_ID = os.getenv("QWEN_GGUF_REPO_ID", "bartowski/Qwen2.5-3B-Instruct-GGUF")
QWEN_FILENAME = os.getenv("QWEN_GGUF_FILENAME", "Qwen2.5-3B-Instruct-Q5_K_M.gguf")

EMBED_REPO_ID = os.getenv("EMBEDDING_REPO_ID", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_REPO_ID = os.getenv("RERANKER_REPO_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN") or None
MAX_WORKERS = int(os.getenv("HF_MAX_WORKERS", "8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _marker_exists(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() or (model_dir / "modules.json").exists()


def download_qwen() -> None:
    _ensure_dir(MODELS_DIR)

    if QWEN_TARGET.exists():
        print(f"[skip] Qwen already exists: {QWEN_TARGET}")
        return

    print(f"[info] Downloading Qwen GGUF: {QWEN_REPO_ID}/{QWEN_FILENAME}")
    downloaded = hf_hub_download(
        repo_id=QWEN_REPO_ID,
        filename=QWEN_FILENAME,
        token=HF_TOKEN,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != QWEN_TARGET.resolve():
        shutil.move(str(downloaded_path), str(QWEN_TARGET))

    print(f"[ok] Saved Qwen GGUF to {QWEN_TARGET}")


def download_snapshot(repo_id: str, target_dir: Path, label: str) -> None:
    _ensure_dir(target_dir)

    if _marker_exists(target_dir):
        print(f"[skip] {label} already exists: {target_dir}")
        return

    print(f"[info] Downloading {label}: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        token=HF_TOKEN,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=MAX_WORKERS,
    )
    print(f"[ok] Saved {label} to {target_dir}")


def main() -> None:
    download_qwen()
    download_snapshot(EMBED_REPO_ID, EMBED_DIR, "embedding model")
    download_snapshot(RERANK_REPO_ID, RERANK_DIR, "reranker model")

    print("\n[done] All local pretrained assets are ready.")
    print(f"       Qwen:      {QWEN_TARGET}")
    print(f"       Embedding: {EMBED_DIR}")
    print(f"       Reranker:  {RERANK_DIR}")


if __name__ == "__main__":
    main()