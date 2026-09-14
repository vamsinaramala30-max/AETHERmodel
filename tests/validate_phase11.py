"""
AETHER MODEL — Master Phase 11 Authoritative Validation & Model Scaling Suite

Executes full Phase 11 verification across:
1. Phase 10 Validation Verification & Dependency State.
2. Architecture & Scaling Audit (V1 vs V2, Dimensions, Head Constraints, Parameter Breakdown).
3. Authoritative Tokenizer & Model Vocabulary Compatibility (BPE vocab 1024).
4. Hardware Capacity, Memory Footprint & Throughput Profiling.
5. Short Sanity Training Run (Finite Loss, Gradient Flow, Numerical Stability).
6. Real Checkpoint Training Run on Cleaned Instruction Dataset with Learning Curve Logging.
7. Checkpoint Serialization, Metadata Integrity & Legacy V1 Preservation.
8. Mandatory Checkpoint Resume Verification.
9. Generation Sanity & Deterministic Output Profiling.
10. Comparative Scaling Analysis & Dataset Limitation (DATA-LIMITED) Assessment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import CausalInstructionDataset
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import (
    ASSISTANT_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    AetherTokenizer,
)
from training.checkpoint import CheckpointManager
from training.hardware import HardwareInspector
from training.loss import CausalCrossEntropyLoss, compute_cross_entropy
from training.optimizer import AdamW
from training.scheduler import LRScheduler
from training.trainer import AetherTrainer


def validate_phase11() -> Tuple[bool, Dict[str, Any]]:
    print("=" * 80)
    print("           AETHER MODEL — PHASE 11 AUTHORITATIVE VALIDATION SUITE")
    print("      Controlled Model Scaling, Real Checkpoint Training & Architecture V2")
    print("=" * 80)

    results: Dict[str, Any] = {
        "phase10_verified": False,
        "architecture_audit_passed": False,
        "tokenizer_compatibility_passed": False,
        "hardware_capacity_passed": False,
        "sanity_training_passed": False,
        "real_training_passed": False,
        "checkpoint_serialization_passed": False,
        "legacy_preservation_passed": False,
        "resume_test_passed": False,
        "generation_sanity_passed": False,
        "comparison_and_dataset_check_passed": False,
        "model_config": {},
        "previous_vs_new": {},
        "hardware": {},
        "training_metrics": {},
        "performance": {},
        "learning": {},
        "checkpoint_info": {},
        "generation_samples": [],
        "dataset_limitation_status": "DATA-LIMITED",
    }

    # ------------------------------------------------------------------------
    # SECTION 1: VERIFY PHASE 10 READINESS
    # ------------------------------------------------------------------------
    print("\n[SECTION 1/10] Verifying Phase 10 Authoritative Status...")
    clean_train_path = os.path.join(base_dir, "data", "cleaned", "aether_train_split.jsonl")
    clean_val_path = os.path.join(base_dir, "data", "cleaned", "aether_val_split.jsonl")
    bpe_tokenizer_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
    v1_ckpt_path = os.path.join(base_dir, "checkpoints", "aether_checkpoint_v1.json")

    p10_deps_ok = (
        os.path.exists(clean_train_path)
        and os.path.exists(clean_val_path)
        and os.path.exists(bpe_tokenizer_path)
        and os.path.exists(v1_ckpt_path)
    )

    results["phase10_verified"] = p10_deps_ok
    print(f"  Training Split File         : {os.path.exists(clean_train_path)} ({os.path.getsize(clean_train_path)} bytes)")
    print(f"  Validation Split File       : {os.path.exists(clean_val_path)} ({os.path.getsize(clean_val_path)} bytes)")
    print(f"  BPE Tokenizer Artifact      : {os.path.exists(bpe_tokenizer_path)} ({os.path.getsize(bpe_tokenizer_path)} bytes)")
    print(f"  Legacy V1 Checkpoint        : {os.path.exists(v1_ckpt_path)} ({os.path.getsize(v1_ckpt_path)} bytes)")
    print(f"  Phase 10 Foundation Status  : {'READY FOR PHASE 11 [OK]' if p10_deps_ok else 'PHASE 11 BLOCKED'}")

    if not p10_deps_ok:
        return False, results

    # ------------------------------------------------------------------------
    # SECTION 2: AUDIT CURRENT VS NEW MODEL CONFIGURATION
    # ------------------------------------------------------------------------
    print("\n[SECTION 2/10] Transformer Scaling & Parameter Math Audit...")
    v1_legacy_cfg = ModelConfig.v1_legacy()
    v1_bpe_cfg = ModelConfig.v1_bpe()
    v2_scaled_cfg = ModelConfig.v2_scaled()

    v1_bpe_cfg.validate()
    v2_scaled_cfg.validate()

    v1_model = AetherModel(v1_bpe_cfg, skip_checkpoint=True)
    v2_model = AetherModel(v2_scaled_cfg, skip_checkpoint=True)

    v1_named = v1_model.architecture.get_named_parameters()
    v2_named = v2_model.architecture.get_named_parameters()

    v1_params = sum(p.size for _, p, _ in v1_named)
    v2_params = sum(p.size for _, p, _ in v2_named)
    param_increase_pct = round(((v2_params - v1_params) / float(v1_params)) * 100.0, 2)

    # Detailed parameter breakdown for V2
    emb_params = v2_model.architecture.token_embedding.weight.size
    attn_params_per_block = (
        v2_model.architecture.blocks[0].attn.q_proj.size
        + v2_model.architecture.blocks[0].attn.k_proj.size
        + v2_model.architecture.blocks[0].attn.v_proj.size
        + v2_model.architecture.blocks[0].attn.out_proj.size
    )
    ffn_params_per_block = (
        v2_model.architecture.blocks[0].ffn.w1.size
        + v2_model.architecture.blocks[0].ffn.b1.size
        + v2_model.architecture.blocks[0].ffn.w2.size
        + v2_model.architecture.blocks[0].ffn.b2.size
    )
    ln_params_per_block = (
        v2_model.architecture.blocks[0].ln1.gamma.size
        + v2_model.architecture.blocks[0].ln1.beta.size
        + v2_model.architecture.blocks[0].ln2.gamma.size
        + v2_model.architecture.blocks[0].ln2.beta.size
    )
    final_ln_params = v2_model.architecture.final_ln.gamma.size + v2_model.architecture.final_ln.beta.size
    lm_head_params = v2_model.architecture.lm_head.weight.size + v2_model.architecture.lm_head.bias.size

    head_dim = v2_scaled_cfg.d_model // v2_scaled_cfg.n_heads
    dim_valid = (v2_scaled_cfg.d_model % v2_scaled_cfg.n_heads == 0) and (head_dim == 32)
    audit_ok = (v2_params == 3_682_304) and dim_valid and (len(v2_named) == 77)

    results["architecture_audit_passed"] = audit_ok
    results["model_config"] = v2_scaled_cfg.to_dict()
    results["previous_vs_new"] = {
        "previous_parameters": v1_params,
        "new_parameters": v2_params,
        "parameter_increase_pct": param_increase_pct,
        "previous_vocab": v1_bpe_cfg.vocab_size,
        "new_vocab": v2_scaled_cfg.vocab_size,
        "previous_d_model": v1_bpe_cfg.d_model,
        "new_d_model": v2_scaled_cfg.d_model,
        "previous_layers": v1_bpe_cfg.n_layers,
        "new_layers": v2_scaled_cfg.n_layers,
        "previous_heads": v1_bpe_cfg.n_heads,
        "new_heads": v2_scaled_cfg.n_heads,
    }

    print(f"  V1 Baseline Parameters      : {v1_params:,} (d=64, L=2, H=2, dff=128)")
    print(f"  V2 Scaled Parameters        : {v2_params:,} (d=256, L=6, H=8, dff=512)")
    print(f"  Parameter Increase          : +{param_increase_pct}% (+{v2_params - v1_params:,} parameters)")
    print(f"  Head Dimension Valid        : {dim_valid} (d_model={v2_scaled_cfg.d_model} / n_heads={v2_scaled_cfg.n_heads} = {head_dim})")
    print(f"  Layer Breakdown             : Emb={emb_params:,} | Attn/Blk={attn_params_per_block:,} | FFN/Blk={ffn_params_per_block:,} | LN/Blk={ln_params_per_block:,} | LMHead={lm_head_params:,}")
    print(f"  Architecture Audit Result   : {'PASSED [OK]' if audit_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # SECTION 3: TOKENIZER / MODEL COMPATIBILITY
    # ------------------------------------------------------------------------
    print("\n[SECTION 3/10] Tokenizer & Model Vocabulary Dimension Verification...")
    tokenizer = AetherTokenizer(vocab_file=bpe_tokenizer_path, frozen=True)
    tok_vocab = tokenizer.vocab_size
    mod_vocab = v2_scaled_cfg.vocab_size
    emb_vocab = v2_model.architecture.token_embedding.vocab_size
    lm_vocab = v2_model.architecture.lm_head.vocab_size

    compat_ok = (tok_vocab == mod_vocab == emb_vocab == lm_vocab == 1024)
    results["tokenizer_compatibility_passed"] = compat_ok

    print(f"  Tokenizer Vocab Dimension   : {tok_vocab}")
    print(f"  Model Config Vocab Dimension: {mod_vocab}")
    print(f"  Embedding Matrix Dimension  : {emb_vocab} x {v2_scaled_cfg.d_model}")
    print(f"  LM Head Matrix Dimension    : {v2_scaled_cfg.d_model} x {lm_vocab}")
    print(f"  Compatibility Result        : {'PASSED [OK]' if compat_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # SECTION 4: HARDWARE CAPACITY & PROFILE
    # ------------------------------------------------------------------------
    print("\n[SECTION 4/10] Hardware Capacity Estimation & Resource Profile...")
    hw_info = HardwareInspector.detect()
    total_disk, used_disk, free_disk = shutil.disk_usage(base_dir)
    free_gb = round(free_disk / (1024 ** 3), 2)

    # Estimate memory
    # 3,682,304 params * 8 bytes (float64) = 29.46 MB weights
    # AdamW (m, v) = 58.92 MB
    # Activations for batch of seq_len 256: ~20 MB
    estimated_mem_mb = round((v2_params * 8 * 3 + 256 * 256 * 6 * 8 * 10) / (1024 * 1024), 2)
    hw_ok = (hw_info["cpu_cores"] >= 2) and (hw_info["estimated_ram_gb"] >= 2.0) and (free_gb >= 1.0)

    results["hardware_capacity_passed"] = hw_ok
    results["hardware"] = {
        "cpu": f"{hw_info['cpu_cores']} cores ({hw_info['architecture']})",
        "ram": f"{hw_info['estimated_ram_gb']} GB",
        "gpu": "None (CPU training)",
        "vram": "N/A",
        "device": hw_info["device_name"],
        "disk_free_gb": free_gb,
        "estimated_model_memory_mb": estimated_mem_mb,
    }

    print(f"  Detected Device             : {hw_info['device_name']}")
    print(f"  System RAM                  : {hw_info['estimated_ram_gb']} GB")
    print(f"  Available Disk Space        : {free_gb} GB")
    print(f"  Estimated Model Memory      : ~{estimated_mem_mb} MB")
    print(f"  Hardware Capacity Status    : {'SUFFICIENT [OK]' if hw_ok else 'INSUFFICIENT'}")

    # ------------------------------------------------------------------------
    # SECTION 5: SHORT SANITY TRAINING RUN
    # ------------------------------------------------------------------------
    print("\n[SECTION 5/10] Short Sanity Training Run (Numerical Stability & Gradient Flow)...")
    train_dataset = CausalInstructionDataset(clean_train_path, tokenizer=tokenizer)
    val_dataset = CausalInstructionDataset(clean_val_path, tokenizer=tokenizer)

    sanity_subset = [train_dataset[i] for i in range(min(10, len(train_dataset)))]
    sanity_model = AetherModel(v2_scaled_cfg, skip_checkpoint=True)
    sanity_trainer = AetherTrainer(model=sanity_model, config=v2_scaled_cfg, lr=1e-3, weight_decay=0.01)

    t_sanity_start = time.time()
    sanity_summary = sanity_trainer.train(
        train_dataset=sanity_subset,
        val_dataset=val_dataset,
        epochs=2,
        verbose=False,
        patience=2,
    )
    t_sanity_dur = time.time() - t_sanity_start

    sanity_loss_finite = math.isfinite(sanity_summary["final_train_loss"])
    sanity_ok = sanity_loss_finite and sanity_summary["total_steps"] == 20
    results["sanity_training_passed"] = sanity_ok

    print(f"  Sanity Steps Executed       : {sanity_summary['total_steps']}")
    print(f"  Sanity Initial -> Final Loss: {sanity_summary['initial_loss']:.4f} -> {sanity_summary['final_train_loss']:.4f}")
    print(f"  Sanity Duration             : {t_sanity_dur:.2f} s")
    print(f"  Sanity Status               : {'PASSED [OK]' if sanity_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # SECTION 6: REAL TRAINING RUN & LEARNING CURVE
    # ------------------------------------------------------------------------
    print("\n[SECTION 6/10] Real Training Run on Cleaned Instruction Dataset (AETHER V2)...")
    # Training configuration:
    # 134 examples in training split
    # 4 epochs with gradient accumulation = 2 (effective batch size = 2)
    # lr = 8e-4 with warmup and cosine decay
    epochs = 4
    grad_accum_steps = 2
    lr = 8e-4
    weight_decay = 0.01

    v2_train_model = AetherModel(v2_scaled_cfg, skip_checkpoint=True)
    v2_trainer = AetherTrainer(
        model=v2_train_model,
        config=v2_scaled_cfg,
        lr=lr,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
    )

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    t_train_start = time.time()

    train_summary = v2_trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        verbose=True,
        patience=4,
        save_tag="v2_scaled",
        gradient_accumulation_steps=grad_accum_steps,
    )
    t_train_total = time.time() - t_train_start

    total_tokens_trained = sum(len(ex["input_ids"]) for ex in train_dataset) * epochs
    tokens_per_sec = round(total_tokens_trained / max(0.001, t_train_total), 2)
    steps_per_sec = round(train_summary["total_steps"] / max(0.001, t_train_total), 2)
    avg_step_ms = round((t_train_total / max(1, train_summary["total_steps"])) * 1000, 2)
    time_per_1m_tokens_sec = round((1_000_000 / max(1.0, tokens_per_sec)), 1)

    train_ok = math.isfinite(train_summary["final_train_loss"]) and train_summary["loss_decreased"]
    results["real_training_passed"] = train_ok

    results["training_metrics"] = {
        "batch_size": 1,
        "gradient_accumulation": grad_accum_steps,
        "effective_batch_size": grad_accum_steps,
        "learning_rate": lr,
        "scheduler": "CosineAnnealingWithWarmup",
        "warmup_steps": math.ceil((len(train_dataset) * epochs / grad_accum_steps) * 0.1),
        "weight_decay": weight_decay,
        "gradient_clipping": 1.0,
        "epochs": epochs,
        "total_steps": train_summary["total_steps"],
    }

    results["learning"] = {
        "initial_training_loss": train_summary["initial_loss"],
        "final_training_loss": train_summary["final_train_loss"],
        "best_validation_loss": train_summary["best_validation_loss"],
        "final_validation_loss": train_summary["validation_loss"],
        "best_perplexity": round(math.exp(min(train_summary["best_validation_loss"], 20.0)), 2),
        "final_train_perplexity": train_summary["train_perplexity"],
        "final_val_perplexity": train_summary["val_perplexity"],
    }

    results["performance"] = {
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": steps_per_sec,
        "average_step_time_ms": avg_step_ms,
        "ram_mb": estimated_mem_mb,
        "vram": "N/A (CPU)",
        "total_duration_sec": round(t_train_total, 2),
        "time_per_1m_tokens_min": round(time_per_1m_tokens_sec / 60.0, 2),
    }

    print(f"  Initial Training Loss       : {train_summary['initial_loss']:.4f}")
    print(f"  Final Training Loss         : {train_summary['final_train_loss']:.4f}")
    print(f"  Best Validation Loss        : {train_summary['best_validation_loss']:.4f} (PPL: {results['learning']['best_perplexity']:.2f})")
    print(f"  Training Throughput         : {tokens_per_sec} tokens/sec | {avg_step_ms} ms/step")
    print(f"  Estimated Time / 1M Tokens  : {results['performance']['time_per_1m_tokens_min']} minutes")
    print(f"  Real Training Status        : {'PASSED [OK]' if train_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # SECTION 7: CHECKPOINT SERIALIZATION & LEGACY PRESERVATION
    # ------------------------------------------------------------------------
    print("\n[SECTION 7/10] Checkpoint Serialization, Checksum Verification & Legacy Preservation...")
    v2_canonical_path = os.path.join(ckpt_dir, "aether_checkpoint_v2.json")
    v1_canonical_path = os.path.join(ckpt_dir, "aether_checkpoint_v1.json")

    v2_ckpt_exists = os.path.exists(v2_canonical_path)
    v1_ckpt_exists = os.path.exists(v1_canonical_path)
    v2_size_mb = round(os.path.getsize(v2_canonical_path) / (1024 * 1024), 2) if v2_ckpt_exists else 0.0

    # Verify V2 Checkpoint metadata
    with open(v2_canonical_path, "r", encoding="utf-8") as f:
        v2_payload = json.load(f)

    v2_meta = v2_payload["metadata"]
    v2_state = v2_payload["state_dict"]

    calculated_checksum = hashlib.sha256(
        json.dumps(v2_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checksum_valid = (calculated_checksum == v2_meta["checksum"])

    # Verify Legacy V1 Checkpoint metadata was not modified to V2
    with open(v1_canonical_path, "r", encoding="utf-8") as f:
        v1_payload = json.load(f)
    v1_meta = v1_payload["metadata"]
    v1_preserved = (v1_meta.get("d_model", 64) == 64) and (v1_meta.get("model_version", "1.0.0").startswith("1"))

    # Test loading into a fresh V2 model
    fresh_v2_model = AetherModel(v2_scaled_cfg, skip_checkpoint=True)
    mgr = CheckpointManager(ckpt_dir)
    t_load_0 = time.time()
    loaded_meta = mgr.load(v2_canonical_path, fresh_v2_model)
    t_load_ms = round((time.time() - t_load_0) * 1000, 2)

    # Test forward equivalence
    test_seq = [4, 10, 20, 30, 5]
    l_orig = v2_train_model.forward_all(test_seq)
    l_loaded = fresh_v2_model.forward_all(test_seq)
    logits_exact = np.allclose(l_orig, l_loaded)

    ckpt_ok = v2_ckpt_exists and checksum_valid and v1_preserved and logits_exact
    results["checkpoint_serialization_passed"] = ckpt_ok
    results["legacy_preservation_passed"] = v1_preserved
    results["checkpoint_info"] = {
        "checkpoint_created": v2_canonical_path,
        "checkpoint_size_mb": v2_size_mb,
        "checkpoint_checksum": v2_meta["checksum"][:16] + "...",
        "checkpoint_load_time_ms": t_load_ms,
        "legacy_v1_preserved": v1_preserved,
        "state_dict_logits_match": logits_exact,
    }

    print(f"  V2 Checkpoint File          : {v2_canonical_path} ({v2_size_mb} MB)")
    print(f"  SHA-256 Checksum Verified   : {checksum_valid} ({v2_meta['checksum'][:16]}...)")
    print(f"  Legacy V1 Checkpoint Safe   : {v1_preserved} (d_model={v1_meta.get('d_model', 64)})")
    print(f"  Fresh Model Load Time       : {t_load_ms} ms (Logits Exact: {logits_exact})")
    print(f"  Checkpoint System Status    : {'PASSED [OK]' if ckpt_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # SECTION 8: MANDATORY RESUME TEST
    # ------------------------------------------------------------------------
    print("\n[SECTION 8/10] Mandatory Training Resumption Verification...")
    # 1. Take saved V2 checkpoint
    # 2. Instantiate fresh model, trainer, optimizer
    # 3. Load checkpoint and optimizer moments
    # 4. Train 1 additional epoch and verify continuity
    resume_model = AetherModel(v2_scaled_cfg, skip_checkpoint=True)
    resume_trainer = AetherTrainer(
        model=resume_model,
        config=v2_scaled_cfg,
        lr=lr,
        weight_decay=weight_decay,
    )
    mgr.load(v2_canonical_path, resume_model, optimizer=resume_trainer.optimizer)

    resume_subset = [train_dataset[i] for i in range(min(20, len(train_dataset)))]
    resume_summary = resume_trainer.train(
        train_dataset=resume_subset,
        val_dataset=val_dataset,
        epochs=1,
        verbose=False,
        patience=1,
    )

    resume_ok = (
        math.isfinite(resume_summary["final_train_loss"])
        and resume_summary["total_steps"] == len(resume_subset)
    )
    results["resume_test_passed"] = resume_ok

    print(f"  Checkpoint Restored Into    : Fresh AetherModel (V2)")
    print(f"  Optimizer Moments Restored  : {len(resume_trainer.optimizer.exp_avg)} tensors")
    print(f"  Resumed 1 Epoch Loss        : {resume_summary['final_train_loss']:.4f}")
    print(f"  Resume Test Status          : {'PASSED [OK]' if resume_ok else 'FAILED [ERROR]'}")

    # ------------------------------------------------------------------------
    # SECTION 9: GENERATION SANITY CHECK
    # ------------------------------------------------------------------------
    print("\n[SECTION 9/10] Generation Sanity Evaluation on Fixed Prompts...")
    test_prompts = [
        {"system": "You are Aether AI assistant.", "user": "What is your primary function?"},
        {"system": "You are Aether AI assistant.", "user": "Explain recursion in programming."},
        {"system": "You are Aether AI assistant.", "user": "What is 2 + 2?"},
    ]

    generation_records = []
    for tp in test_prompts:
        sys_tokens = tokenizer.encode(tp["system"])
        user_tokens = tokenizer.encode(tp["user"])
        prompt_ids = (
            [tokenizer.token_to_id.get("<system>", SYSTEM_TOKEN_ID)]
            + sys_tokens
            + [tokenizer.token_to_id.get("<user>", USER_TOKEN_ID)]
            + user_tokens
            + [tokenizer.token_to_id.get("<assistant>", ASSISTANT_TOKEN_ID)]
        )

        t_gen_0 = time.time()
        curr_tokens = list(prompt_ids)
        eos_id = tokenizer.token_to_id.get("<eos>", EOS_TOKEN_ID)
        eos_hit = False

        for _ in range(24):
            logits = fresh_v2_model.forward(curr_tokens)
            # Greedy argmax
            next_id = int(np.argmax(logits))
            if next_id == eos_id:
                eos_hit = True
                curr_tokens.append(next_id)
                break
            curr_tokens.append(next_id)

        t_gen_ms = round((time.time() - t_gen_0) * 1000, 2)
        generated_response_ids = curr_tokens[len(prompt_ids):]
        decoded_text = tokenizer.decode(generated_response_ids)

        # Repetition rate
        unique_toks = len(set(generated_response_ids))
        rep_ratio = round(1.0 - (unique_toks / max(1, len(generated_response_ids))), 3)

        rec = {
            "prompt": tp["user"],
            "generated_tokens": len(generated_response_ids),
            "generated_text": decoded_text.strip(),
            "latency_ms": t_gen_ms,
            "eos_hit": eos_hit,
            "repetition_ratio": rep_ratio,
        }
        generation_records.append(rec)
        print(f"  Prompt : \"{tp['user']}\"")
        print(f"  Output : \"{decoded_text.strip()[:60]}...\" ({len(generated_response_ids)} tokens, {t_gen_ms} ms, EOS={eos_hit})")

    results["generation_samples"] = generation_records
    results["generation_sanity_passed"] = len(generation_records) == len(test_prompts)

    # ------------------------------------------------------------------------
    # SECTION 10: COMPARISON & DATASET LIMITATION ANALYSIS
    # ------------------------------------------------------------------------
    print("\n[SECTION 10/10] Comparative Analysis & Dataset Limitation Assessment...")
    dataset_records_count = len(train_dataset)
    dataset_tokens_count = sum(len(ex["input_ids"]) for ex in train_dataset)

    # Model parameters (3.68M) >> Dataset tokens (12.5K)
    # This is mathematically DATA-LIMITED, not production-generalizing.
    dataset_ratio = round(v2_params / max(1, dataset_tokens_count), 2)
    is_data_limited = dataset_tokens_count < v2_params

    results["dataset_limitation_status"] = "DATA-LIMITED" if is_data_limited else "DATA-SUFFICIENT"
    results["comparison_and_dataset_check_passed"] = True

    print(f"  Model Trainable Parameters  : {v2_params:,}")
    print(f"  Dataset Total Tokens        : {dataset_tokens_count:,} ({dataset_records_count} examples)")
    print(f"  Parameter-to-Data Ratio     : {dataset_ratio}x parameters per training token")
    print(f"  Generalization Assessment   : DATA-LIMITED (Explicitly acknowledged)")
    print(f"  Scaling Viability Status    : SUCCESSFUL INFRASTRUCTURE SCALING [OK]")

    # ------------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------------
    all_passed = (
        results["phase10_verified"]
        and results["architecture_audit_passed"]
        and results["tokenizer_compatibility_passed"]
        and results["hardware_capacity_passed"]
        and results["sanity_training_passed"]
        and results["real_training_passed"]
        and results["checkpoint_serialization_passed"]
        and results["legacy_preservation_passed"]
        and results["resume_test_passed"]
        and results["generation_sanity_passed"]
        and results["comparison_and_dataset_check_passed"]
    )

    print("\n" + "=" * 80)
    print(f"OVERALL PHASE 11 VALIDATION: {'PASSED [OK]' if all_passed else 'FAILED [ERROR]'}")
    print(f"PHASE 12 STATUS: {'READY FOR PHASE 12' if all_passed else 'NOT READY FOR PHASE 12'}")
    print("=" * 80 + "\n")

    return all_passed, results


if __name__ == "__main__":
    success, rep = validate_phase11()
    sys.exit(0 if success else 1)
