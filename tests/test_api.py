"""API endpoint tests verifying OpenAI compatibility, modes, and failure resiliency."""

import pytest
from fastapi.testclient import TestClient
from contextpilot.app.main import app
from contextpilot.app.db.session import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ContextPilot"


def test_chat_completions_openai_compatible():
    client = TestClient(app)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is ContextPilot?"}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Standard OpenAI format fields
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]

    # ContextPilot report extensions
    assert "contextpilot" in data
    cp = data["contextpilot"]
    assert cp["decision"] in ("skipped", "optimized", "fallback")
    assert "tokens_original" in cp
    assert "tokens_sent" in cp
    assert "reduction_pct" in cp
    assert "latency_overhead_ms" in cp


def test_chat_completions_guarded_mode():
    client = TestClient(app)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Where were the 2016 Olympics held?"}
        ],
        "contextpilot": {
            "mode": "guarded",
            "token_budget": 500,
            "skip_threshold_tokens": 10,
            "passages": [
                {"id": "p_olympics", "text": "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil."},
                {"id": "p_pasta", "text": "Pasta is typically made from durum wheat semolina dough."}
            ]
        }
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    cp = response.json()["contextpilot"]
    assert cp["decision"] == "optimized"
    assert cp["evidence_check"] == "passed"
    assert any(k["id"] == "p_olympics" for k in cp["kept"])


def test_chat_completions_aggressive_mode():
    client = TestClient(app)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is the chemical symbol for gold?"}
        ],
        "contextpilot": {
            "mode": "aggressive",
            "token_budget": 200,
            "skip_threshold_tokens": 10,
            "passages": [
                {
                    "id": "p_gold",
                    "text": "Gold is a chemical element with symbol Au and atomic number 79. Gold is used in jewelry and bullion. It has high malleability."
                }
            ]
        }
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    cp = response.json()["contextpilot"]
    assert cp["decision"] == "optimized"


def test_chat_completions_shadow_mode():
    client = TestClient(app)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is the boiling point of water at sea level?"}
        ],
        "contextpilot": {
            "mode": "shadow",
            "skip_threshold_tokens": 10,
            "passages": [
                {"id": "p_water", "text": "At sea level, water boils at 212 degrees Fahrenheit."}
            ]
        }
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    cp = response.json()["contextpilot"]
    assert "shadow" in cp
    assert cp["shadow"] is not None
    assert "answer_match" in cp["shadow"]
    assert "risk_flag" in cp["shadow"]
    assert 0.0 <= cp["shadow"]["answer_match"] <= 1.0


def test_resiliency_failed_log_does_not_break_response(monkeypatch):
    """Verify that if database log write fails in BackgroundTasks, the client response is unaffected."""
    client = TestClient(app)

    import contextpilot.app.db.session as session_module

    def failing_factory():
        raise RuntimeError("Database pool connection timed out!")

    monkeypatch.setattr(session_module, "get_session_factory", failing_factory)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hello resiliency test"}]
    }
    # Must succeed with 200 OK even when database logging fails
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "choices" in response.json()


def test_requests_and_stats_endpoints():
    client = TestClient(app)
    stats_res = client.get("/v1/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert "total_requests" in stats

    reqs_res = client.get("/v1/requests")
    assert reqs_res.status_code == 200
    reqs = reqs_res.json()
    assert "items" in reqs


def test_collections_ingest_endpoint():
    client = TestClient(app)
    payload = {
        "texts": [
            "ContextPilot documentation text snippet 1.",
            "ContextPilot documentation text snippet 2."
        ],
        "chunk_size": 200,
        "chunk_overlap": 20
    }
    res = client.post("/v1/collections/test_coll/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["collection"] == "test_coll"
    assert data["chunks_ingested"] >= 2
    assert data["status"] == "success"
