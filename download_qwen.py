from pathlib import Path
from huggingface_hub import hf_hub_download

out_dir = Path("models")
out_dir.mkdir(exist_ok=True)

path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q5_k_m.gguf",
    local_dir=str(out_dir),
)

print(path)