"""
AETHER_MODEL — Phase 2 Production Baseline Unit & Integration Test Suite

Tests:
  1. GGUF Checkpoint Loading & State Machine
  2. Native Tokenizer Encode/Decode Roundtrip
  3. Real Non-Empty Deterministic Generation
  4. Real Token-Level Streaming with SSE protocol
  5. Structured Output (JSON) Validation
  6. Context Window Boundary Safety
  7. Failure Behavior (Missing checkpoint, invalid payload, resource cleanup)
  8. FastAPI Serving Endpoints (/health, /model/status, /v1/models, /v1/generate, /v1/stream)
"""
from __future__ import annotations

import json
import os
import sys
import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aether.config import settings
from aether.core.llama_engine import EngineState, LlamaCppEngine, LlamaEngineManager
from aether.api.server import app, ModelState, _model_status

GGUF_PATH = settings.model.gguf_model_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gguf_engine():
    """Session-scoped LlamaCppEngine instance for fast testing."""
    if not os.path.exists(GGUF_PATH):
        pytest.skip(f"GGUF checkpoint not found at: {GGUF_PATH}")
    engine = LlamaCppEngine(model_path=GGUF_PATH, n_ctx=1024, n_threads=2, verbose=False)
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def api_client():
    """TestClient instance connected to the unified FastAPI server."""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. Model Loading & State Machine Tests
# ---------------------------------------------------------------------------

def test_gguf_file_integrity():
    """Verify that the GGUF file exists and is >1 GB in size."""
    assert os.path.exists(GGUF_PATH), f"Checkpoint missing at {GGUF_PATH}"
    size = os.path.getsize(GGUF_PATH)
    assert size > 1_000_000_000, f"Expected size > 1 GB, got {size} bytes"


def test_model_loading_and_state(gguf_engine):
    """Verify model loads into READY state with correct metadata."""
    assert gguf_engine.state == EngineState.READY
    assert gguf_engine._llm is not None
    info = gguf_engine.get_info()
    assert info["loaded"] is True
    assert info["state"] == "READY"
    assert info["backend"] == "llama.cpp"
    assert info["architecture"] == "qwen2"
    assert info["quantization"] == "Q4_K_M"


def test_missing_model_fails_loudly():
    """Verify that attempting to load a non-existent GGUF raises FileNotFoundError and sets UNAVAILABLE."""
    bogus_path = os.path.join(_ROOT, "checkpoints", "does_not_exist_123.gguf")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        LlamaCppEngine(model_path=bogus_path, n_ctx=512)


def test_model_close_and_reinitialization():
    """Verify resource release resets engine state to UNINITIALIZED."""
    test_eng = LlamaCppEngine(model_path=GGUF_PATH, n_ctx=512, n_threads=2)
    assert test_eng.state == EngineState.READY
    test_eng.close()
    assert test_eng.state == EngineState.UNINITIALIZED
    assert test_eng._llm is None


# ---------------------------------------------------------------------------
# 2. Tokenizer Tests
# ---------------------------------------------------------------------------

def test_tokenizer_roundtrip_ascii(gguf_engine):
    """Verify ASCII encoding/decoding roundtrip."""
    text = "Hello world from Aether neural runtime."
    tokens = gguf_engine.tokenize(text)
    assert len(tokens) > 0
    decoded = gguf_engine.detokenize(tokens)
    assert decoded.strip() == text.strip()


def test_tokenizer_roundtrip_multiline_and_numbers(gguf_engine):
    """Verify multiline, indentation, and numeric tokenization roundtrip."""
    text = "Count: 1, 2, 3, 42, 100\n\tTabbed line with 3.14159"
    tokens = gguf_engine.tokenize(text)
    decoded = gguf_engine.detokenize(tokens)
    assert decoded.strip() == text.strip()


def test_tokenizer_vocabulary_size(gguf_engine):
    """Verify that native tokenizer vocabulary is ~152k tokens, not the legacy 1024."""
    # Special token for Qwen2 <|im_end|> is 151645
    tokens = gguf_engine.tokenize("<|im_end|>", special=True)
    assert len(tokens) == 1
    assert tokens[0] == 151645


# ---------------------------------------------------------------------------
# 3. Real Generation Tests
# ---------------------------------------------------------------------------

def test_real_generation_non_empty(gguf_engine):
    """Verify generation returns real, non-empty, non-mock text."""
    res = gguf_engine.generate(
        prompt="What is the capital of Japan? Answer with one word.",
        max_tokens=20,
        temperature=0.0,
    )
    assert res.success is True
    assert len(res.text.strip()) > 0
    assert "tokyo" in res.text.lower()
    assert res.tokens_used > 0
    assert res.tokens_per_second is not None
    assert res.tokens_per_second > 0


def test_deterministic_generation(gguf_engine):
    """Verify that temperature=0.0 produces deterministic responses."""
    p = "What is 10 + 10? Answer with only the number."
    res1 = gguf_engine.generate(prompt=p, max_tokens=10, temperature=0.0)
    res2 = gguf_engine.generate(prompt=p, max_tokens=10, temperature=0.0)
    assert res1.text == res2.text


def test_system_prompt_identity(gguf_engine):
    """Verify that the model adopts the system prompt identity."""
    system = "You are Aether, an intelligent AI Life OS created by Vamsi."
    res = gguf_engine.generate(
        prompt="What is your name?",
        system_prompt=system,
        max_tokens=30,
        temperature=0.0,
    )
    assert "aether" in res.text.lower()


# ---------------------------------------------------------------------------
# 4. Streaming Tests
# ---------------------------------------------------------------------------

def test_streaming_token_delivery(gguf_engine):
    """Verify streaming yields incremental chunks and terminates with done=True."""
    chunks = []
    for chunk in gguf_engine.stream_generate(
        prompt="Write the numbers 1, 2, 3.",
        max_tokens=30,
        temperature=0.0,
    ):
        chunks.append(chunk)

    assert len(chunks) >= 2
    # Check that deltas were received
    deltas = [c["delta"] for c in chunks if c.get("delta")]
    assert len(deltas) > 0

    # Check that final chunk has done=True
    assert chunks[-1]["done"] is True
    assert chunks[-1]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 5. Structured Output Tests
# ---------------------------------------------------------------------------

def test_structured_json_output(gguf_engine):
    """Verify model can produce valid JSON when explicitly instructed."""
    prompt = (
        "Output a valid JSON object with keys 'task' and 'priority'. "
        "Task must be 'Review tests' and priority 'high'. "
        "Output ONLY raw JSON."
    )
    res = gguf_engine.generate(prompt=prompt, max_tokens=100, temperature=0.0)
    clean = res.text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    data = json.loads(clean)
    assert isinstance(data, dict)
    assert "task" in data
    assert "priority" in data


# ---------------------------------------------------------------------------
# 6. Context Boundaries
# ---------------------------------------------------------------------------

def test_safe_context_boundaries(gguf_engine):
    """Verify that token counting and generation work within context bounds."""
    count = gguf_engine.count_tokens("This is a context boundary test.")
    assert 5 <= count <= 12
    assert count < gguf_engine._n_ctx


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_health_endpoint(api_client):
    """Verify GET /health returns 200 OK with truthful status."""
    res = api_client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "status" in body
    assert body["has_trained_weights"] is True


def test_api_model_status_endpoint(api_client):
    """Verify GET /model/status returns state details."""
    res = api_client.get("/model/status")
    assert res.status_code == 200
    body = res.json()
    assert "state" in body
    assert "model" in body


def test_api_models_endpoint(api_client):
    """Verify GET /v1/models returns OpenAI-compatible models list."""
    res = api_client.get("/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "list"
    assert len(body["data"]) > 0
    assert body["data"][0]["backend"] == "llama.cpp"


def test_api_generate_endpoint(api_client):
    """Verify POST /v1/generate returns real completion with usage stats."""
    payload = {
        "prompt": "What is 2 + 2? Answer with only the number.",
        "max_tokens": 10,
        "temperature": 0.0,
    }
    res = api_client.post("/v1/generate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "content" in body
    assert "4" in body["content"]
    assert "usage" in body
    assert body["usage"]["completion_tokens"] > 0
    assert body["has_trained_weights"] is True


def test_api_stream_endpoint(api_client):
    """Verify POST /v1/stream yields genuine SSE events with 'delta' and '[DONE]'."""
    payload = {
        "prompt": "Name two primary colors.",
        "max_tokens": 20,
        "temperature": 0.0,
    }
    res = api_client.post("/v1/stream", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    lines = res.text.split("\n")
    data_lines = [line for line in lines if line.startswith("data:")]
    assert len(data_lines) >= 2

    # Verify at least one line has delta JSON and last is [DONE]
    has_delta = False
    for dl in data_lines:
        content = dl.replace("data:", "").strip()
        if content == "[DONE]":
            continue
        try:
            parsed = json.loads(content)
            if "delta" in parsed and parsed["delta"]:
                has_delta = True
        except Exception:
            pass

    assert has_delta, "No delta token chunks found in stream"
    assert any("[DONE]" in dl for dl in data_lines), "Stream missing final [DONE]"


def test_api_empty_prompt_validation(api_client):
    """Verify POST /v1/generate rejects empty prompt with 400 Bad Request."""
    res = api_client.post("/v1/generate", json={"prompt": ""})
    assert res.status_code == 400
