# -*- coding: utf-8 -*-
"""
AETHER_MODEL Phase 4: Dedicated Verification & Regression Test Suite

Validates:
1. Model Registry integrity & role assignments
2. Hardware Feasibility Guard behavior
3. Tokenizer Special Tokens & 256-byte lossless coverage
4. Causal LM next-token shift, prompt masking, and gradient math
5. Track A Architecture specification & parameter count (~3.68M)
6. Baseline GGUF integrity & cryptographic SHA-256 preservation
7. Candidate checkpoint loading & deterministic inference equivalence
8. Fast-path API contract compliance (/health, /model/status, /v1/models)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aether.config import settings
from aether.api.server import app, ModelState, _model_status
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import (
    AetherTokenizer,
    PAD_TOKEN_ID,
    EOS_TOKEN_ID,
    SPECIAL_TOKENS,
    SPECIAL_TOKEN_IDS,
)
from training.loss import CausalCrossEntropyLoss
from scripts.train_aether_lora import check_hardware_feasibility


# ---------------------------------------------------------------------------
# 1. Model Registry Tests
# ---------------------------------------------------------------------------

def test_model_registry_schema_and_roles():
    """Verify checkpoints/model_registry.json registers baseline and candidate with correct roles."""
    reg_path = os.path.join(ROOT_DIR, "checkpoints", "model_registry.json")
    assert os.path.exists(reg_path), f"Registry missing: {reg_path}"

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert registry.get("registry_version") == "1.0.0"
    assert registry.get("active_production_model") == "qwen2.5-1.5b-instruct-q4km"
    assert registry.get("authoritative_baseline") == "qwen2.5-1.5b-instruct-q4km"

    models = registry.get("models", {})
    assert "qwen2.5-1.5b-instruct-q4km" in models
    assert "aether-small-v0.1" in models

    baseline = models["qwen2.5-1.5b-instruct-q4km"]
    assert baseline["status"] == "VALIDATED_BASELINE"
    assert baseline["role"] == "authoritative_production_default"
    assert baseline["production_ready"] is True
    assert baseline["quantization"] == "Q4_K_M"

    candidate = models["aether-small-v0.1"]
    assert candidate["status"] == "EXPERIMENTAL_CANDIDATE"
    assert candidate["role"] == "experimental_neural_candidate"
    assert candidate["production_ready"] is False
    assert candidate["parameters"] == 3682304
    assert "TRAINING_BLOCKED_BY_HARDWARE" in candidate["feasibility_verdict"]


# ---------------------------------------------------------------------------
# 2. Hardware Guard Tests
# ---------------------------------------------------------------------------

def test_hardware_feasibility_guard_prevents_unsafe_cpu_training():
    """Verify hardware guard correctly flags low-RAM host and prevents OOM backprop."""
    feasible, msg = check_hardware_feasibility(min_ram_gb=8.0)
    assert feasible is False
    assert "HARDWARE INSUFFICIENT" in msg
    assert "Total System RAM" in msg
    assert "No CUDA GPU" in msg


# ---------------------------------------------------------------------------
# 3. Tokenizer Special Tokens & Byte Coverage Tests
# ---------------------------------------------------------------------------

def test_phase4_tokenizer_special_tokens_and_byte_coverage():
    """Verify tokenizer has 9 special tokens, vocab 1024, and 100% byte coverage."""
    bpe_path = os.path.join(ROOT_DIR, "checkpoints", "aether_bpe_tokenizer.json")
    tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)

    assert tokenizer.is_bpe is True
    assert tokenizer.vocab_size == 1024
    assert len(SPECIAL_TOKENS) == 9

    expected_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<system>", "<user>", "<assistant>", "<tool>", "<evidence>"]
    for idx, tok in enumerate(expected_tokens):
        assert tokenizer.token_to_id.get(tok) == idx
        assert tokenizer.decode([idx], skip_special_tokens=False) == tok

    # All 256 bytes lossless roundtrip
    raw_bytes_str = bytes(range(256)).decode("latin-1")
    encoded = tokenizer.encode(raw_bytes_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == raw_bytes_str


# ---------------------------------------------------------------------------
# 4. Causal Loss & Prompt Masking Math Tests
# ---------------------------------------------------------------------------

def test_causal_lm_next_token_shift_and_masking():
    """Verify mathematical next-token shift, prompt masking, and pad masking."""
    criterion = CausalCrossEntropyLoss(pad_token_id=0, eos_token_id=3)
    vocab = 1024
    seq = 8
    asst_idx = 4

    logits = np.random.randn(seq, vocab) * 2.0
    targets = np.array([10, 20, 30, 40, 50, 60, 0, 3], dtype=np.int64)  # 0 is pad, 3 is eos

    loss, grad_logits, metrics = criterion(logits, targets, asst_start_idx=asst_idx)
    assert math.isfinite(loss)
    assert loss > 0.0
    assert grad_logits.shape == (seq, vocab)

    # Prompt tokens before asst_idx must have zero gradient
    prompt_grad = grad_logits[:asst_idx - 1]
    assert np.max(np.abs(prompt_grad)) == 0.0

    # Pad token at index 6 must have zero gradient
    assert np.all(grad_logits[6] == 0.0)


# ---------------------------------------------------------------------------
# 5. Track A Architecture & Parameter Tests
# ---------------------------------------------------------------------------

def test_track_a_architecture_parameter_count():
    """Verify Track A architecture (v2_scaled) has exactly 3,682,304 parameters."""
    cfg = ModelConfig.v2_scaled()
    assert cfg.n_layers == 6
    assert cfg.n_heads == 8
    assert cfg.d_model == 256
    assert cfg.d_ff == 512
    assert cfg.vocab_size == 1024

    model = AetherModel(cfg, skip_checkpoint=True)
    named = model.architecture.get_named_parameters()
    total_params = sum(np.prod(p.shape) for _, p, _ in named)
    assert total_params == 3682304


# ---------------------------------------------------------------------------
# 6. Baseline GGUF Preservation Tests
# ---------------------------------------------------------------------------

def test_authoritative_baseline_gguf_unaltered():
    """Verify production baseline GGUF exists, size matches, and SHA-256 is preserved."""
    gguf_path = settings.model.gguf_model_path
    assert os.path.exists(gguf_path)
    size = os.path.getsize(gguf_path)
    assert size == 1117320736

    # Verify SHA-256 prefix
    hasher = hashlib.sha256()
    with open(gguf_path, "rb") as f:
        chunk = f.read(65536)
        while chunk:
            hasher.update(chunk)
            chunk = f.read(65536)
    digest = hasher.hexdigest()
    assert digest == "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"


# ---------------------------------------------------------------------------
# 7. Candidate Checkpoint Integrity Tests
# ---------------------------------------------------------------------------

def test_candidate_checkpoint_reload_and_determinism():
    """Verify candidate checkpoint loads and produces deterministic forward pass."""
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "aether_checkpoint_phase4_candidate.json")
    if not os.path.exists(ckpt_path):
        pytest.skip("Candidate checkpoint not yet saved")

    cfg = ModelConfig.v2_scaled()
    model1 = AetherModel(cfg, skip_checkpoint=True)
    assert model1.load_checkpoint(ckpt_path) is True

    model2 = AetherModel(cfg, skip_checkpoint=True)
    assert model2.load_checkpoint(ckpt_path) is True

    test_tokens = [2, 15, 88, 105, 3]
    out1 = np.array(model1.forward_all(test_tokens))
    out2 = np.array(model2.forward_all(test_tokens))
    diff = float(np.max(np.abs(out1 - out2)))
    assert diff < 1e-6


# ---------------------------------------------------------------------------
# 8. Fast-Path API Conformance Tests
# ---------------------------------------------------------------------------

def test_api_serving_contracts():
    """Verify FastAPI endpoints conform to model-independent contracts."""
    with TestClient(app, raise_server_exceptions=False) as client:
        # /health
        res_health = client.get("/health")
        assert res_health.status_code == 200
        body_health = res_health.json()
        assert "ok" in body_health
        assert "status" in body_health

        # /model/status
        res_status = client.get("/model/status")
        assert res_status.status_code == 200
        body_status = res_status.json()
        assert "state" in body_status
        assert "model" in body_status

        # /v1/models
        res_models = client.get("/v1/models")
        assert res_models.status_code == 200
        body_models = res_models.json()
        assert body_models.get("object") == "list"
        assert len(body_models.get("data", [])) >= 1
        model_info = body_models["data"][0]
        assert model_info.get("backend") == "llama.cpp"
        assert model_info.get("architecture") == "qwen2"
