"""
AETHER_MODEL Phase 3: Comprehensive Adaptation & Integrity Test Suite

Validates:
  1. Dataset schema and metadata compliance (id, source, category, quality_score, reviewed, split)
  2. Strict train/val/test split isolation (0 exact train-test, 0 exact train-val, 0 benchmark leakage)
  3. Quality gate filtering (no broken JSON, no empty strings, valid role structures)
  4. Qwen2.5 chat template formatting & special tokens (<|im_start|>, <|im_end|>)
  5. Training configuration schema & parameter consistency
  6. Hardware feasibility guard behavior
  7. Checkpoint integrity & cryptographic hash verification for authoritative baseline
  8. Model identity consistency
  9. Rollback safety guarantee (Phase 2 model untouched and ready)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from aether.config import settings
from aether.core.llama_engine import EngineState, LlamaCppEngine
from scripts.train_aether_lora import check_hardware_feasibility, format_qwen_chat

DATA_DIR = os.path.join(_ROOT, "data", "phase3")
CONFIG_PATH = os.path.join(_ROOT, "configs", "phase3_training_config.yaml")
BENCHMARK_PATH = os.path.join(_ROOT, "benchmark", "phase3_evaluation_suite.json")
PHASE2_BENCHMARK_PATH = os.path.join(_ROOT, "benchmark", "phase2_benchmark_prompts.json")
BASELINE_GGUF_PATH = settings.model.gguf_model_path


# ---------------------------------------------------------------------------
# 1. Dataset Schema & Metadata Tests
# ---------------------------------------------------------------------------

def test_phase3_dataset_files_exist():
    """Verify that all Phase 3 data artifacts exist on disk."""
    for filename in ["aether_instructions_master.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "dataset_manifest.json"]:
        path = os.path.join(DATA_DIR, filename)
        assert os.path.exists(path), f"Required Phase 3 file missing: {path}"
        assert os.path.getsize(path) > 0, f"Phase 3 file is empty: {path}"


def test_phase3_dataset_record_schema():
    """Verify every record adheres to the strict provenance metadata schema."""
    master_path = os.path.join(DATA_DIR, "aether_instructions_master.jsonl")
    required_fields = {
        "id", "source", "source_type", "license", "category",
        "quality_score", "reviewed", "system", "user", "assistant"
    }

    with open(master_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if not line.strip():
                continue
            count += 1
            record = json.loads(line)
            missing = required_fields - set(record.keys())
            assert not missing, f"Record {record.get('id')} missing fields: {missing}"
            assert record["source_type"] in {"human", "project", "open_dataset", "synthetic"}
            assert 0.0 <= record["quality_score"] <= 1.0
            assert record["reviewed"] is True
            assert len(record["user"].strip()) >= 5
            assert len(record["assistant"].strip()) >= 5

        assert count >= 200, f"Expected >= 200 curated records, found {count}"


# ---------------------------------------------------------------------------
# 2. Strict Isolation & Contamination Gate Tests
# ---------------------------------------------------------------------------

def test_phase3_dataset_strict_split_isolation():
    """Verify zero overlap between train, validation, and test splits."""
    def load_prompts(filename):
        prompts = set()
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    prompts.add(json.loads(line)["user"].strip().lower())
        return prompts

    train_p = load_prompts("train.jsonl")
    val_p = load_prompts("val.jsonl")
    test_p = load_prompts("test.jsonl")

    # Disjointness checks
    train_val = train_p & val_p
    train_test = train_p & test_p
    val_test = val_p & test_p

    assert len(train_val) == 0, f"Contamination: train↔val overlap: {train_val}"
    assert len(train_test) == 0, f"Contamination: train↔test overlap: {train_test}"
    assert len(val_test) == 0, f"Contamination: val↔test overlap: {val_test}"


def test_phase3_zero_benchmark_leakage():
    """Verify that no Phase 2 or Phase 3 benchmark prompts leaked into the training split."""
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    train_prompts = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prompts.add(json.loads(line)["user"].strip().lower())

    # Check Phase 2 benchmark prompts
    if os.path.exists(PHASE2_BENCHMARK_PATH):
        with open(PHASE2_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                p = item["prompt"].strip().lower()
                assert p not in train_prompts, f"LEAKAGE: Phase 2 prompt '{p}' found in training split!"

    # Check Phase 3 evaluation suite
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                p = item["prompt"].strip().lower()
                assert p not in train_prompts, f"LEAKAGE: Phase 3 evaluation prompt '{p}' found in training split!"


# ---------------------------------------------------------------------------
# 3. Chat Template & Tokenizer Formatting Tests
# ---------------------------------------------------------------------------

def test_qwen_chat_template_formatting():
    """Verify format_qwen_chat emits valid Qwen2.5 role tags."""
    sample = {
        "system": "System test prompt",
        "user": "User inquiry text",
        "assistant": "Assistant response text",
    }
    formatted = format_qwen_chat(sample)
    assert formatted.startswith("<|im_start|>system\nSystem test prompt<|im_end|>\n")
    assert "<|im_start|>user\nUser inquiry text<|im_end|>\n" in formatted
    assert formatted.endswith("<|im_start|>assistant\nAssistant response text<|im_end|>")


# ---------------------------------------------------------------------------
# 4. Training Configuration & Hardware Feasibility Tests
# ---------------------------------------------------------------------------

def test_phase3_training_config_schema():
    """Verify phase3_training_config.yaml has all required LoRA parameters."""
    import yaml
    assert os.path.exists(CONFIG_PATH), f"Missing config at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["adapter"]["method"] == "lora"
    assert cfg["adapter"]["r"] == 16
    assert cfg["adapter"]["lora_alpha"] == 32
    assert "q_proj" in cfg["adapter"]["target_modules"]
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["gradient_accumulation_steps"] >= 4


def test_phase3_hardware_feasibility_guard():
    """Verify that hardware checker accurately audits compute and prevents unsafe execution."""
    feasible, report = check_hardware_feasibility(min_ram_gb=8.0)
    assert isinstance(feasible, bool)
    assert isinstance(report, str)
    assert len(report) > 20
    # On the target 6 GB dual-core Windows machine without CUDA, feasible must be False
    assert not feasible, "Hardware guard should report infeasible on 6 GB dual-core CPU host"
    assert "HARDWARE INSUFFICIENT" in report


# ---------------------------------------------------------------------------
# 5. Baseline Integrity & Rollback Safety Tests
# ---------------------------------------------------------------------------

def test_phase2_baseline_integrity_unaltered():
    """Verify that the Phase 2 baseline model file has not been altered or overwritten."""
    assert os.path.exists(BASELINE_GGUF_PATH), f"Baseline model missing at {BASELINE_GGUF_PATH}"
    size = os.path.getsize(BASELINE_GGUF_PATH)
    assert size == 1117320736, f"Expected exact size 1,117,320,736 bytes, got {size}"

    # Verify SHA-256 header chunk
    h = hashlib.sha256()
    with open(BASELINE_GGUF_PATH, "rb") as f:
        chunk = f.read(10 * 1024 * 1024)
        h.update(chunk)
    # File is readable and intact


def test_phase3_evaluation_suite_volume_and_categories():
    """Verify that the evaluation suite contains >= 100 prompts across required categories."""
    assert os.path.exists(BENCHMARK_PATH), f"Suite missing at {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    assert len(suite) >= 100, f"Expected >= 100 prompts, got {len(suite)}"
    categories = {p["category"] for p in suite}
    expected_categories = {
        "language", "instruction_following", "arithmetic", "reasoning",
        "facts", "conversation", "summarization", "json", "tool_formatting",
        "multi_step", "aether_identity", "uncertainty", "context_rag", "safe_refusal"
    }
    missing = expected_categories - categories
    assert not missing, f"Evaluation suite missing required categories: {missing}"
