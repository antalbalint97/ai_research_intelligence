import json
import time
import requests

BASE_URL = "http://localhost:8000"


def main():
    print("Testing /health ...")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(r.status_code, r.json())

    print("\nTesting /metrics ...")
    r = requests.get(f"{BASE_URL}/metrics", timeout=10)
    print(r.status_code, r.json())

    payload = {
        "query": "What are the main trends in multimodal AI?",
        "mode": "fast",
        "top_k": 3,
        "retrieval_k": 8,
    }

    print("\nTesting /query ...")
    start = time.perf_counter()
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=120)
    elapsed = time.perf_counter() - start

    print("status:", r.status_code)
    print("elapsed_s:", round(elapsed, 2))

    data = r.json()
    print(json.dumps({
        "mode": data.get("mode"),
        "latency_ms": data.get("latency_ms"),
        "retrieval_count": data.get("retrieval_count"),
        "reranked_count": data.get("reranked_count"),
        "model": data.get("model"),
        "timings": data.get("timings"),
        "answer_preview": data.get("answer", "")[:300],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()