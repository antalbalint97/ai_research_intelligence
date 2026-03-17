import requests

BASE_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_metrics():
    r = requests.get(f"{BASE_URL}/metrics", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_query_fast():
    payload = {
        "query": "What are the main trends in multimodal AI?",
        "mode": "fast",
        "top_k": 3,
        "retrieval_k": 8,
    }
    r = requests.post(f"{BASE_URL}/query", json=payload, timeout=120)
    assert r.status_code == 200
    data = r.json()

    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"].strip()) > 0

    assert "sources" in data
    assert isinstance(data["sources"], list)

    assert "latency_ms" in data
    assert "mode" in data
    assert data["mode"] == "fast"

    assert "timings" in data
    assert "generate_ms" in data["timings"]