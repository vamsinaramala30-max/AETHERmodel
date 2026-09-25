# -*- coding: utf-8 -*-
"""
AETHER_MODEL Phase 4: Track A Candidate Controlled Training Experiment
Trains the Track A custom architecture (~3.68M parameters) on the full Phase 3
training split (218 records) with epoch-level validation on val.jsonl (36 records).

Logs:
- Step loss, Epoch loss, Validation loss, Perplexity, Token accuracy
- Real hardware telemetry: Elapsed time, CPU utilization, Resident RAM (MB)
- Saves deterministic checkpoint: checkpoints/aether_checkpoint_phase4_candidate.json
- Verifies reload determinism (< 1e-6 logit diff)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer, EOS_TOKEN_ID, PAD_TOKEN_ID
from training.checkpoint import CheckpointManager
from training.dataset import InstructionDataset
from training.loss import CausalCrossEntropyLoss
from training.optimizer import AdamW
from training.scheduler import LRScheduler


def get_process_ram_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)


def get_cpu_percent() -> float:
    return psutil.cpu_percent(interval=None)


def run_training_experiment(epochs: int = 3, max_seq_len: int = 96) -> Dict[str, Any]:
    print("================================================================================")
    print("AETHER_MODEL PHASE 4: TRACK A CONTROLLED TRAINING EXPERIMENT")
    print("================================================================================")

    np.random.seed(42)
    start_time = time.time()

    # 1. Load Tokenizer
    bpe_path = os.path.join(ROOT_DIR, "checkpoints", "aether_bpe_tokenizer.json")
    assert os.path.exists(bpe_path), f"BPE artifact not found: {bpe_path}"
    tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
    pad_id = tokenizer.token_to_id.get("<pad>", PAD_TOKEN_ID)
    eos_id = tokenizer.token_to_id.get("<eos>", EOS_TOKEN_ID)

    # 2. Load Datasets
    train_file = os.path.join(ROOT_DIR, "data", "phase3", "train.jsonl")
    val_file = os.path.join(ROOT_DIR, "data", "phase3", "val.jsonl")

    train_ds = InstructionDataset(file_path=train_file, tokenizer=tokenizer, max_seq_len=max_seq_len)
    val_ds = InstructionDataset(file_path=val_file, tokenizer=tokenizer, max_seq_len=max_seq_len)

    print(f"Datasets Loaded:")
    print(f"  Train: {len(train_ds)} samples (max_seq_len={max_seq_len})")
    print(f"  Val:   {len(val_ds)} samples (max_seq_len={max_seq_len})")

    # 3. Model Architecture Setup
    config = ModelConfig.v2_scaled(max_seq_len=max_seq_len)
    model = AetherModel(config, skip_checkpoint=True)

    named_params = model.architecture.get_named_parameters()
    total_params = int(sum(np.prod(p.shape) for _, p, _ in named_params))
    print(f"\nModel Configuration: {config.model_name} (v{config.model_version})")
    print(f"  Layers: {config.n_layers}, Heads: {config.n_heads}, d_model: {config.d_model}, d_ff: {config.d_ff}, Vocab: {config.vocab_size}")
    print(f"  Total Trainable Parameters: {total_params:,}")

    # 4. Training Hyperparameters & Optimizer
    base_lr = 1.5e-3
    min_lr = 1e-4
    total_steps = len(train_ds) * epochs
    warmup_steps = 15

    optimizer = AdamW(named_parameters=named_params, lr=base_lr, weight_decay=0.01, max_grad_norm=1.0)
    scheduler = LRScheduler(base_lr=base_lr, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
    criterion = CausalCrossEntropyLoss(pad_token_id=pad_id, eos_token_id=eos_id)

    # 5. Baseline Evaluation (Pre-training / Epoch 0)
    print("\nEvaluating initial untrained model on validation set...")
    val_losses_0 = []
    val_acc_0 = []
    for item in val_ds:
        logits = model.forward_all(item["input_ids"])
        loss, _, metrics = criterion(logits, item["target_ids"], asst_start_idx=item["asst_start_idx"])
        val_losses_0.append(loss)
        val_acc_0.append(metrics["token_accuracy"])

    initial_val_loss = float(np.mean(val_losses_0))
    initial_val_ppl = float(math.exp(min(initial_val_loss, 20)))
    initial_val_acc = float(np.mean(val_acc_0))

    print(f"Initial Val Loss: {initial_val_loss:.4f} | Perplexity: {initial_val_ppl:.2f} | Token Acc: {initial_val_acc*100:.2f}%")

    # 6. Training Loop
    step_metrics_log = []
    epoch_metrics_log = []
    step_count = 0
    best_val_loss = float("inf")

    print("\nStarting Training Execution...")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        epoch_train_losses = []
        epoch_train_acc = []
        epoch_grad_norms = []

        # Train on all 218 samples
        for idx, item in enumerate(train_ds):
            step_count += 1
            input_ids = item["input_ids"]
            target_ids = item["target_ids"]
            asst_start_idx = item["asst_start_idx"]

            # Forward pass
            logits = model.forward_all(input_ids)

            # Causal Loss computation
            loss, grad_logits, metrics = criterion(logits, target_ids, asst_start_idx=asst_start_idx)
            epoch_train_losses.append(loss)
            epoch_train_acc.append(metrics["token_accuracy"])

            # Backward pass & Gradient accumulation
            model.zero_grad()
            model.architecture.backward(grad_logits)

            # Gradient clipping & Optimizer update
            total_norm_sq = 0.0
            for _, _, grad in named_params:
                total_norm_sq += float(np.sum(grad * grad))
            grad_norm = math.sqrt(total_norm_sq)
            epoch_grad_norms.append(grad_norm)

            current_lr = scheduler.get_lr(step_count)
            optimizer.step(lr_override=current_lr)

            # Log periodically
            if (idx + 1) % 50 == 0 or idx == len(train_ds) - 1:
                step_metrics_log.append({
                    "epoch": epoch,
                    "step": step_count,
                    "sample_idx": idx + 1,
                    "loss": round(float(loss), 4),
                    "token_acc": round(float(metrics["token_accuracy"]), 4),
                    "grad_norm": round(grad_norm, 4),
                    "lr": round(current_lr, 6),
                    "ram_mb": get_process_ram_mb(),
                })
                print(f"  [Epoch {epoch} | Step {step_count:3d}/{total_steps}] Sample {idx+1:3d}/{len(train_ds)} | Loss: {loss:.4f} | Acc: {metrics['token_accuracy']*100:5.1f}% | GradNorm: {grad_norm:.3f} | LR: {current_lr:.6f}", flush=True)

        # Validation after epoch
        val_losses = []
        val_acc = []
        for item in val_ds:
            logits = model.forward_all(item["input_ids"])
            loss, _, metrics = criterion(logits, item["target_ids"], asst_start_idx=item["asst_start_idx"])
            val_losses.append(loss)
            val_acc.append(metrics["token_accuracy"])

        avg_train_loss = float(np.mean(epoch_train_losses))
        avg_train_acc = float(np.mean(epoch_train_acc))
        avg_train_ppl = float(math.exp(min(avg_train_loss, 20)))
        avg_val_loss = float(np.mean(val_losses))
        avg_val_acc = float(np.mean(val_acc))
        avg_val_ppl = float(math.exp(min(avg_val_loss, 20)))
        epoch_duration = time.time() - epoch_start_time

        epoch_summary = {
            "epoch": epoch,
            "duration_s": round(epoch_duration, 2),
            "train_loss": round(avg_train_loss, 4),
            "train_perplexity": round(avg_train_ppl, 2),
            "train_token_acc": round(avg_train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_perplexity": round(avg_val_ppl, 2),
            "val_token_acc": round(avg_val_acc, 4),
            "avg_grad_norm": round(float(np.mean(epoch_grad_norms)), 4),
            "final_lr": round(current_lr, 6),
            "process_ram_mb": get_process_ram_mb(),
            "cpu_percent": get_cpu_percent(),
        }
        epoch_metrics_log.append(epoch_summary)

        print(f"\n--- Epoch {epoch}/{epochs} Complete ({epoch_duration:.1f}s) ---", flush=True)
        print(f"  Train Loss: {avg_train_loss:.4f} (PPL: {avg_train_ppl:.2f}) | Train Acc: {avg_train_acc*100:5.1f}%", flush=True)
        print(f"  Val Loss:   {avg_val_loss:.4f} (PPL: {avg_val_ppl:.2f}) | Val Acc:   {avg_val_acc*100:5.1f}%", flush=True)
        print(f"  Process RAM: {get_process_ram_mb():.1f} MB | CPU: {get_cpu_percent():.1f}%\n", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    total_training_duration = time.time() - start_time

    # 7. Checkpoint Saving with Full Metadata and SHA-256
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
    ckpt_mgr = CheckpointManager(ckpt_dir, model_name="aether-small-v0.1")
    ckpt_filename = "aether_checkpoint_phase4_candidate.json"
    ckpt_path, checksum = ckpt_mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step_count,
        epoch=epochs,
        metrics={
            "train_loss": avg_train_loss,
            "train_perplexity": avg_train_ppl,
            "val_loss": avg_val_loss,
            "val_perplexity": avg_val_ppl,
            "val_token_acc": avg_val_acc,
            "training_duration_s": round(total_training_duration, 2),
            "total_params": int(total_params),
        },
        filename=ckpt_filename,
    )
    ckpt_size_bytes = os.path.getsize(ckpt_path)
    print(f"[PASS] Saved candidate checkpoint: {ckpt_filename}")
    print(f"  Path: {ckpt_path}")
    print(f"  Size: {ckpt_size_bytes:,} bytes")
    print(f"  SHA-256: {checksum}")

    # 8. Checkpoint Reload Verification (Deterministic Equivalence)
    print("\nVerifying Checkpoint Reload Determinism...")
    reloaded_model = AetherModel(config, skip_checkpoint=True)
    load_success = reloaded_model.load_checkpoint(ckpt_path)
    assert load_success, "Failed to reload saved checkpoint!"

    # Test equivalence on first 5 validation samples
    max_logit_diff = 0.0
    for i in range(min(5, len(val_ds))):
        test_ids = val_ds[i]["input_ids"]
        logits_orig = np.array(model.forward_all(test_ids))
        logits_reloaded = np.array(reloaded_model.forward_all(test_ids))
        diff = float(np.max(np.abs(logits_orig - logits_reloaded)))
        if diff > max_logit_diff:
            max_logit_diff = diff

    assert max_logit_diff < 1e-6, f"Checkpoint reload produced differing logits: {max_logit_diff}"
    print(f"[PASS] Deterministic Checkpoint Reload Equivalence: Max Logit Diff = {max_logit_diff:.2e} (< 1e-6)")

    # 9. Assemble Summary & Export Artifact
    summary = {
        "status": "COMPLETED",
        "phase": "AETHER_MODEL_PHASE4",
        "candidate_model": {
            "name": "aether-small-v0.1",
            "role": "experimental_neural_candidate",
            "architecture": "transformer_decoder_v2",
            "trainable_parameters": total_params,
            "checkpoint_path": ckpt_path,
            "checkpoint_filename": ckpt_filename,
            "file_size_bytes": ckpt_size_bytes,
            "sha256": checksum,
            "reload_max_logit_diff": max_logit_diff,
        },
        "training_data": {
            "train_file": "data/phase3/train.jsonl",
            "train_samples": len(train_ds),
            "val_file": "data/phase3/val.jsonl",
            "val_samples": len(val_ds),
            "epochs": epochs,
            "total_steps": step_count,
            "max_seq_len": max_seq_len,
        },
        "metrics": {
            "initial_val_loss": round(initial_val_loss, 4),
            "initial_val_perplexity": round(initial_val_ppl, 2),
            "initial_val_token_acc": round(initial_val_acc, 4),
            "final_train_loss": round(avg_train_loss, 4),
            "final_train_perplexity": round(avg_train_ppl, 2),
            "final_train_token_acc": round(avg_train_acc, 4),
            "final_val_loss": round(avg_val_loss, 4),
            "final_val_perplexity": round(avg_val_ppl, 2),
            "final_val_token_acc": round(avg_val_acc, 4),
            "loss_drop_pct": round((initial_val_loss - avg_val_loss) / initial_val_loss * 100, 2),
        },
        "telemetry": {
            "total_duration_s": round(total_training_duration, 2),
            "epoch_history": epoch_metrics_log,
            "step_samples": step_metrics_log,
            "final_process_ram_mb": get_process_ram_mb(),
            "cpu_percent": get_cpu_percent(),
        },
    }

    metrics_out_path = os.path.join(ROOT_DIR, "data", "phase4_training_metrics.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining metrics saved to: {metrics_out_path}")
    print("================================================================================")
    return summary


if __name__ == "__main__":
    run_training_experiment(epochs=3, max_seq_len=96)
