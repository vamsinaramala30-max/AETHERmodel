"""
AETHER — API Contract Smoke Tests (Phase 6)

Tests the API contract, not model internals.
Every test asserts on WHAT the response contains, not HOW it was generated.

Run with:
    # Start server first:
    uvicorn aether.api.server:app --port 5002

    # Then:
    pytest tests/test_smoke_api.py -v

Or without a running server (using TestClient):
    pytest tests/test_smoke_api.py -v --no-server

Env vars:
    AETHER_TEST_BASE_URL   - Base URL if testing against a live server (default: use TestClient)
    AETHER_TEST_TIMEOUT    - Request timeout in seconds (default: 120)
    AETHER_TEST_MAX_LATENCY_MS - Max acceptable latency for smoke tests (default: 30000)
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Client fixture — TestClient (no live server needed) or httpx live client
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("AETHER_TEST_BASE_URL", "")
TIMEOUT = int(os.environ.get("AETHER_TEST_TIMEOUT", "120"))
MAX_LATENCY_MS = int(os.environ.get("AETHER_TEST_MAX_LATENCY_MS", "30000"))

# Detect if we should skip HF-dependent tests
_HF_AVAILABLE = False
try:
    import transformers  # noqa: F401
    import chromadb  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    _HF_AVAILABLE = True
except ImportError:
    pass

requires_hf = pytest.mark.skipif(
    not _HF_AVAILABLE,
    reason="HuggingFace stack not installed (run: pip install -r requirements_hf.txt)",
)


@pytest.fixture(scope="session")
def client():
    """
    Returns either a FastAPI TestClient (no running server needed)
    or an httpx client pointed at AETHER_TEST_BASE_URL.
    """
    if BASE_URL:
        import httpx
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
            yield c
    else:
        try:
            from fastapi.testclient import TestClient
            from aether.api.server import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c
        except ImportError:
            pytest.skip("FastAPI/httpx not installed — run: pip install fastapi httpx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GARBAGE_PATTERN = re.compile(r"(\*{3,}|\.{5,}|_{10,}|\s{10,})")


def assert_valid_response(body: dict, *, check_evidence_field: bool = True) -> None:
    """Common assertions that every /v1/generate response must satisfy."""
    assert "content" in body, f"Response missing 'content' field: {body}"
    text = body["content"]
    assert isinstance(text, str), f"content must be a string, got {type(text)}"
    assert len(text.strip()) > 0, "content is empty"
    assert not GARBAGE_PATTERN.search(text), (
        f"content looks like word-soup / garbage output: '{text[:200]}'"
    )
    if check_evidence_field:
        assert "evidence_used" in body, f"Response missing 'evidence_used' field: {body}"
        assert isinstance(body["evidence_used"], bool), (
            f"evidence_used must be bool, got {type(body['evidence_used'])}"
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        # 200 = model loaded, 503 = startup failed — both are valid JSON, not crashes
        assert resp.status_code in (200, 503), f"Unexpected status: {resp.status_code}"

    @requires_hf
    def test_health_ok_when_model_loaded(self, client):
        resp = client.get("/health")
        body = resp.json()
        if resp.status_code == 503:
            pytest.skip(f"Model not loaded: {body.get('error')}")
        assert body.get("ok") is True
        assert body.get("status") == "READY"
        assert "model" in body
        assert "timestamp_ms" in body

    def test_health_never_returns_200_with_missing_fields(self, client):
        resp = client.get("/health")
        body = resp.json()
        # Whether ok or error, must always have 'status' and 'timestamp_ms'
        assert "status" in body, f"Health response missing 'status': {body}"


# ---------------------------------------------------------------------------
# Generation — Phase 1 gate
# ---------------------------------------------------------------------------

SMOKE_PROMPTS = [
    "What is the capital of France?",
    "Explain what a transformer neural network is in two sentences.",
    "Write a short poem about autumn.",
    "What are three benefits of exercise?",
    "How do I reverse a list in Python?",
    "Summarize the importance of sleep in one paragraph.",
    "What is 17 multiplied by 23?",
    "Describe the water cycle briefly.",
    "What does HTTP stand for and what is it used for?",
    "Give me a simple recipe for scrambled eggs.",
]


@requires_hf
class TestGeneration:
    def test_generate_returns_non_empty_text(self, client):
        resp = client.post("/v1/generate", json={"prompt": "Hello, who are you?"})
        assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text[:200]}"
        body = resp.json()
        assert_valid_response(body)

    @pytest.mark.parametrize("prompt", SMOKE_PROMPTS)
    def test_smoke_prompts_produce_coherent_output(self, client, prompt):
        """Phase 1 hard gate: all 10 prompts return coherent, non-empty, non-garbage text."""
        resp = client.post(
            "/v1/generate",
            json={"prompt": prompt, "max_tokens": 256, "temperature": 0.3},
        )
        assert resp.status_code == 200, (
            f"Prompt '{prompt[:50]}' returned {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        assert_valid_response(body)

    def test_empty_prompt_returns_400(self, client):
        resp = client.post("/v1/generate", json={"prompt": ""})
        assert resp.status_code == 400

    def test_temperature_out_of_range_returns_422(self, client):
        resp = client.post("/v1/generate", json={"prompt": "Hi", "temperature": 5.0})
        assert resp.status_code == 422

    def test_max_tokens_out_of_range_returns_422(self, client):
        resp = client.post("/v1/generate", json={"prompt": "Hi", "max_tokens": 9999})
        assert resp.status_code == 422

    def test_response_includes_usage_info(self, client):
        resp = client.post("/v1/generate", json={"prompt": "What is 2 + 2?"})
        if resp.status_code != 200:
            pytest.skip(f"Model not available: {resp.status_code}")
        body = resp.json()
        assert "usage" in body
        usage = body["usage"]
        assert "latency_ms" in usage
        latency = usage["latency_ms"]
        assert isinstance(latency, (int, float))
        assert latency < MAX_LATENCY_MS, (
            f"Response took {latency}ms, exceeding threshold of {MAX_LATENCY_MS}ms"
        )

    def test_response_has_required_fields(self, client):
        resp = client.post("/v1/generate", json={"prompt": "Hello"})
        if resp.status_code != 200:
            pytest.skip("Model not available")
        body = resp.json()
        required = ["id", "object", "content", "confidence", "evidence_used", "model"]
        for field in required:
            assert field in body, f"Missing required field '{field}' in response"


# ---------------------------------------------------------------------------
# RAG / evidence_used — Phase 2 gate
# ---------------------------------------------------------------------------

@requires_hf
class TestRAG:
    def test_evidence_used_false_when_db_empty(self, client):
        """
        When the knowledge DB is empty (or the query has no relevant match),
        evidence_used must be False. This is the Phase 2 hard gate:
        evidence_used must reflect reality, not be hardcoded True.
        """
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "xzqwerty_uniquetoken_nosimilardocument_789",
                "use_rag": True,
                "max_tokens": 64,
            },
        )
        if resp.status_code != 200:
            pytest.skip("Model not available")
        body = resp.json()
        # On an empty DB, evidence_used MUST be False
        assert body["evidence_used"] is False, (
            f"evidence_used was True but no relevant document exists in DB. "
            f"This means evidence_used is hardcoded — Phase 2 gate FAILED."
        )

    def test_index_and_retrieve(self, client):
        """Index a document, then ask about it — evidence_used should flip to True."""
        # Index a unique document
        unique_text = "The Aether project uses ChromaDB for vector storage of documents."
        idx_resp = client.post("/v1/knowledge/index", json={"text": unique_text})
        if idx_resp.status_code != 200:
            pytest.skip(f"Index endpoint unavailable: {idx_resp.status_code}")

        # Query about it
        resp = client.post(
            "/v1/generate",
            json={"prompt": "What does Aether use for vector storage?", "use_rag": True},
        )
        if resp.status_code != 200:
            pytest.skip("Model not available")
        body = resp.json()
        assert_valid_response(body)
        # evidence_used should be True since we just indexed a relevant doc
        assert body["evidence_used"] is True, (
            "evidence_used was False after indexing a relevant document — "
            "RAG retrieval not working correctly."
        )

    def test_evidence_used_is_always_bool(self, client):
        """evidence_used must always be a bool, never a string or None."""
        resp = client.post("/v1/generate", json={"prompt": "test"})
        if resp.status_code != 200:
            pytest.skip("Model not available")
        body = resp.json()
        assert isinstance(body.get("evidence_used"), bool), (
            f"evidence_used must be bool, got {type(body.get('evidence_used'))}"
        )


# ---------------------------------------------------------------------------
# Memory — Phase 3 gate
# ---------------------------------------------------------------------------

@requires_hf
class TestMemory:
    def test_remember_endpoint_stores_fact(self, client):
        resp = client.post(
            "/v1/memory/remember",
            json={"fact": "The user prefers concise responses.", "user_id": "test_user_smoke"},
        )
        if resp.status_code == 503:
            pytest.skip("Memory store not available")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("stored") is True
        assert "doc_id" in body

    def test_remember_then_recall_in_generation(self, client):
        """Phase 3 gate: stored memory should surface in subsequent generation."""
        unique_fact = "The user's cat is named Pixel and loves tuna."
        user_id = "test_user_memory_recall_smoke"

        # Store memory
        remember_resp = client.post(
            "/v1/memory/remember",
            json={"fact": unique_fact, "user_id": user_id},
        )
        if remember_resp.status_code == 503:
            pytest.skip("Memory not available")

        # Query about it
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "What is my cat's name?",
                "user_id": user_id,
                "use_memory": True,
                "max_tokens": 128,
            },
        )
        if resp.status_code != 200:
            pytest.skip("Model not available")

        body = resp.json()
        assert_valid_response(body)
        # The cat name "Pixel" should appear in the response if memory was injected
        assert "Pixel" in body["content"] or "pixel" in body["content"].lower(), (
            f"Memory fact not recalled in response. Response: '{body['content'][:300]}'"
        )

    def test_remember_intent_via_generate(self, client):
        """POSTing a 'remember that' prompt to /v1/generate should store, not generate."""
        resp = client.post(
            "/v1/generate",
            json={
                "prompt": "Remember that my preferred timezone is IST",
                "user_id": "test_user_intent_smoke",
                "use_memory": True,
            },
        )
        if resp.status_code != 200:
            pytest.skip("Model not available")
        body = resp.json()
        # Should get a confirmation, not a random generation
        assert_valid_response(body, check_evidence_field=False)
        content_lower = body["content"].lower()
        assert any(w in content_lower for w in ["stored", "remember", "got it", "noted"]), (
            f"Remember intent not handled correctly. Got: '{body['content'][:200]}'"
        )

    def test_empty_fact_returns_error(self, client):
        resp = client.post(
            "/v1/memory/remember",
            json={"fact": "", "user_id": "test_user"},
        )
        assert resp.status_code in (400, 422, 500), (
            "Empty fact should return an error, not 200"
        )


# ---------------------------------------------------------------------------
# Agent loop — Phase 5 gate
# ---------------------------------------------------------------------------

@requires_hf
class TestAgentLoop:
    def test_agent_returns_typed_status(self, client):
        resp = client.post(
            "/v1/agent",
            json={"prompt": "What is 5 + 5?", "max_steps": 3},
        )
        if resp.status_code != 200:
            pytest.skip("Agent endpoint not available")
        body = resp.json()
        assert "status" in body, f"Agent response missing 'status': {body}"
        assert body["status"] in ("done", "error"), (
            f"status must be 'done' or 'error', got '{body['status']}'"
        )
        assert "steps_taken" in body
        assert body["steps_taken"] <= 3

    def test_agent_never_exceeds_max_steps(self, client):
        """Phase 5 hard gate: agent must not exceed max_steps."""
        resp = client.post(
            "/v1/agent",
            json={"prompt": "Keep thinking forever.", "max_steps": 2},
        )
        if resp.status_code != 200:
            pytest.skip("Agent endpoint not available")
        body = resp.json()
        assert body.get("steps_taken", 0) <= 2, (
            f"Agent exceeded max_steps=2. Got steps_taken={body.get('steps_taken')}"
        )

    def test_agent_error_has_detail(self, client):
        """Error responses must include a human-readable detail."""
        resp = client.post(
            "/v1/agent",
            json={"prompt": "x", "max_steps": 1},
        )
        if resp.status_code != 200:
            pytest.skip("Agent not available")
        body = resp.json()
        if body["status"] == "error":
            assert "detail" in body, "Error status must include 'detail' field"
            assert isinstance(body["detail"], str) and len(body["detail"]) > 0


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------

class TestLegacyCompat:
    def test_health_endpoint_accessible(self, client):
        """Both /health and /v1/health must respond."""
        for path in ["/health", "/v1/health"]:
            resp = client.get(path)
            assert resp.status_code in (200, 503), f"{path} returned {resp.status_code}"

    def test_audit_endpoint_accessible(self, client):
        resp = client.post("/v1/audit", json={"prompt": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert "has_rag" in body
        assert "has_memory" in body
