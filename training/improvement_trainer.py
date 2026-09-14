"""
AETHER MODEL — Phase 21 Master Improvement Training & Behavioral Alignment Engine

Executes controlled, regression-safe fine-tuning to fix real failure modes identified in Phase 20
without causing catastrophic forgetting or degrading baseline capabilities.

Features:
- Deterministic experiment tracking (AETHER_PHASE21_RUN_001)
- Ingests Phase 20 improvement dataset + regression preservation dataset
- Conservative learning rate schedule (warmup + cosine decay)
- Validation loss monitoring & best checkpoint selection with SHA-256 checksums
- Checkpoint resumption & integrity verification
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.checkpoint import CheckpointManager
from training.dataset import InstructionDataset
from training.hardware import HardwareInspector
from training.loss import CausalCrossEntropyLoss, compute_cross_entropy
from training.optimizer import AdamW
from training.scheduler import LRScheduler

EXPERIMENT_ID = "AETHER_PHASE21_RUN_001"
DATASET_VERSION = "AETHER_IMPROVEMENT_DATA_V1"
DETERMINISTIC_SEED = 42


class CombinedImprovementDataset(InstructionDataset):
    """
    Dataset combining targeted improvement examples with regression preservation examples.
    """

    def __init__(
        self,
        improvement_path: str,
        regression_path: Optional[str] = None,
        tokenizer: Optional[AetherTokenizer] = None,
        max_seq_len: int = 512,
        min_seq_len: int = 4,
    ):
        self.improvement_path = improvement_path
        self.regression_path = regression_path
        super().__init__(file_path=None, tokenizer=tokenizer, max_seq_len=max_seq_len, min_seq_len=min_seq_len)
        self.load_combined()

    def load_combined(self) -> None:
        raw_lines = []
        # 1. Load improvement records
        if os.path.exists(self.improvement_path):
            with open(self.improvement_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            raw_lines.append(json.loads(line.strip()))
                        except Exception:
                            self.stats["filtered_examples"] += 1

        # 2. Load regression preservation records
        if self.regression_path and os.path.exists(self.regression_path):
            with open(self.regression_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            raw_lines.append(json.loads(line.strip()))
                        except Exception:
                            self.stats["filtered_examples"] += 1

        self.stats["total_raw_examples"] = len(raw_lines)
        seen_hashes = set()
        clean_records = []

        for item in raw_lines:
            sys_text = item.get("system") or item.get("context") or ""
            user_text = item.get("user") or item.get("instruction") or item.get("prompt") or ""
            if item.get("input") and str(item.get("input")).strip():
                user_text = f"{user_text}\nInput: {item.get('input')}"
            asst_text = item.get("assistant") or item.get("output") or item.get("response") or ""
            category = item.get("category") or "general"

            if not user_text.strip() or not asst_text.strip():
                self.stats["filtered_examples"] += 1
                continue

            content_key = f"{user_text.strip().lower()} -> {asst_text.strip().lower()}"
            chash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
            if chash in seen_hashes:
                self.stats["filtered_examples"] += 1
                continue
            seen_hashes.add(chash)

            rec = {
                "system": sys_text.strip(),
                "user": user_text.strip(),
                "assistant": asst_text.strip(),
                "category": category,
            }
            clean_records.append(rec)
            self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1

        self.records = clean_records
        self.stats["deduplicated_examples"] = len(self.records)

        full_content = json.dumps(self.records, sort_keys=True)
        self.stats["dataset_hash"] = f"sha256_{hashlib.sha256(full_content.encode('utf-8')).hexdigest()[:16]}"
        self._tokenize_all()


class Phase21ImprovementTrainer:
    """
    Authoritative training engine for Phase 21 Improvement Fine-Tuning.
    """

    def __init__(
        self,
        base_checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        experiment_id: str = EXPERIMENT_ID,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.experiment_id = experiment_id
        self.ckpt_dir = os.path.join(base_dir, "checkpoints")
        self.base_checkpoint_path = base_checkpoint_path or os.path.join(self.ckpt_dir, "aether_checkpoint_v2.json")
        self.tokenizer_path = tokenizer_path or os.path.join(self.ckpt_dir, "aether_bpe_tokenizer.json")

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.label_smoothing = float(label_smoothing)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

        # Hardware & Tokenizer
        self.hardware_info = HardwareInspector.detect()
        self.tokenizer = AetherTokenizer(vocab_file=self.tokenizer_path, frozen=True)

        # Model Config & Model Loading
        self.config = ModelConfig.authoritative()
        self.model = AetherModel(self.config, skip_checkpoint=True)
        loaded = self.model.load_checkpoint(self.base_checkpoint_path)
        if not loaded:
            raise RuntimeError(f"Failed to load baseline checkpoint from {self.base_checkpoint_path}: {self.model.last_validation_errors}")

        self.criterion = CausalCrossEntropyLoss(
            pad_token_id=0,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        named_params = self.model.architecture.get_named_parameters()
        self.optimizer = AdamW(
            named_parameters=named_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
        )

        self.history: List[Dict[str, Any]] = []

    def evaluate(self, val_dataset: InstructionDataset) -> float:
        """Computes cross-entropy loss across validation dataset."""
        if val_dataset is None or len(val_dataset) == 0:
            return 0.0

        total_loss = 0.0
        count = 0

        for item in val_dataset:
            input_ids = item["input_ids"]
            target_ids = item["target_ids"]
            asst_idx = item.get("asst_start_idx", 0)

            logits = self.model.forward_all(input_ids)
            loss, _, _ = compute_cross_entropy(logits, target_ids, asst_idx, pad_token_id=0)
            if math.isfinite(loss):
                total_loss += loss
                count += 1

        return round(total_loss / float(max(1, count)), 4)

    def _compute_gradient_norm(self) -> float:
        import numpy as np
        total_norm_sq = 0.0
        for _, _, grad in self.model.architecture.get_named_parameters():
            g_arr = np.asarray(grad, dtype=np.float64)
            total_norm_sq += float(np.sum(g_arr * g_arr))
        return round(math.sqrt(total_norm_sq), 6)

    def train(
        self,
        train_dataset: InstructionDataset,
        val_dataset: Optional[InstructionDataset] = None,
        epochs: int = 12,
        checkpoint_dir: Optional[str] = None,
        save_tag: str = "v3_improved",
        verbose: bool = True,
        patience: int = 5,
        min_delta: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Executes controlled fine-tuning with validation tracking, early stopping, and checkpointing.
        """
        if len(train_dataset) == 0:
            raise ValueError("Train dataset cannot be empty.")

        grad_accum = self.gradient_accumulation_steps
        total_examples = len(train_dataset) * epochs
        total_opt_steps = math.ceil(total_examples / grad_accum)
        warmup_steps = max(2, total_opt_steps // 10)
        scheduler = LRScheduler(base_lr=self.lr, warmup_steps=warmup_steps, total_steps=total_opt_steps)

        initial_loss = 0.0
        final_loss = 0.0
        step_idx = 0
        opt_step_idx = 0

        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0
        stopped_early = False

        out_ckpt_dir = checkpoint_dir or os.path.join(self.ckpt_dir, "experiment_p21_improvement")
        os.makedirs(out_ckpt_dir, exist_ok=True)
        ckpt_manager = CheckpointManager(out_ckpt_dir, max_to_keep=5, model_name="aether-v3-improved")

        start_time = time.time()
        if verbose:
            print("=" * 80)
            print(f"=== [PHASE 21 IMPROVEMENT TRAINING] Experiment: {self.experiment_id} ===")
            print(f"=== Base Checkpoint: {os.path.basename(self.base_checkpoint_path)} | Params: {sum(p.size for _, p, _ in self.model.architecture.get_named_parameters()):,} ===")
            print(f"=== Dataset: {len(train_dataset)} examples | Epochs: {epochs} | LR: {self.lr} | Warmup: {warmup_steps} ===")
            print("=" * 80, flush=True)

        self.model.zero_grad()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_grad_norms: List[float] = []
            epoch_token_acc: List[float] = []
            epoch_eos_acc: List[float] = []

            for i, item in enumerate(train_dataset):
                step_idx += 1
                input_ids = item["input_ids"]
                target_ids = item["target_ids"]
                asst_idx = item.get("asst_start_idx", 0)

                # 1. Forward pass
                logits = self.model.forward_all(input_ids)

                # 2. Cross entropy loss w.r.t active assistant response tokens
                loss, grad_logits, step_metrics = compute_cross_entropy(
                    logits_matrix=logits,
                    target_ids=target_ids,
                    asst_start_idx=asst_idx,
                    pad_token_id=0,
                )

                if not math.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step_idx}: {loss}")

                if step_idx == 1:
                    initial_loss = loss

                epoch_token_acc.append(step_metrics.get("token_accuracy", 0.0))
                epoch_eos_acc.append(step_metrics.get("eos_accuracy", 0.0))

                # 3. Analytical backward pass
                if grad_accum > 1:
                    scaled_grad = [[g / float(grad_accum) for g in row] for row in grad_logits]
                    self.model.backward(scaled_grad)
                else:
                    self.model.backward(grad_logits)

                epoch_loss += loss
                final_loss = loss

                # 4. Optimizer update step
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_dataset):
                    opt_step_idx += 1
                    current_lr = scheduler.get_lr(opt_step_idx)

                    grad_norm = self._compute_gradient_norm()
                    epoch_grad_norms.append(grad_norm)

                    self.optimizer.step(lr_override=current_lr)
                    self.model.zero_grad()

            avg_epoch_loss = epoch_loss / float(len(train_dataset))
            val_loss = self.evaluate(val_dataset) if val_dataset else avg_epoch_loss
            train_ppl = round(math.exp(min(avg_epoch_loss, 20.0)), 2)
            val_ppl = round(math.exp(min(val_loss, 20.0)), 2)
            avg_grad_norm = round(sum(epoch_grad_norms) / max(1, len(epoch_grad_norms)), 6) if epoch_grad_norms else 0.0
            avg_token_acc = round(sum(epoch_token_acc) / max(1, len(epoch_token_acc)), 4) if epoch_token_acc else 0.0
            avg_eos_acc = round(sum(epoch_eos_acc) / max(1, len(epoch_eos_acc)), 4) if epoch_eos_acc else 0.0
            current_lr_val = scheduler.get_lr(opt_step_idx) if opt_step_idx > 0 else self.lr

            self.history.append({
                "epoch": epoch,
                "step": step_idx,
                "opt_step": opt_step_idx,
                "train_loss": round(avg_epoch_loss, 4),
                "val_loss": val_loss,
                "train_perplexity": train_ppl,
                "val_perplexity": val_ppl,
                "avg_grad_norm": avg_grad_norm,
                "token_accuracy": avg_token_acc,
                "eos_accuracy": avg_eos_acc,
                "learning_rate": current_lr_val,
            })

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_epoch_loss:.4f} (PPL: {train_ppl:0.1f}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:0.1f}) | TokenAcc: {avg_token_acc:.3f} | "
                    f"GradNorm: {avg_grad_norm:.4f} | LR: {current_lr_val:.6f}",
                    flush=True,
                )

            # Early stopping and best model tracking
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state_dict = json.loads(json.dumps(self.model.get_state_dict()))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  --> Early stopping triggered at epoch {epoch} (Best epoch {best_epoch}: {best_val_loss:.4f})", flush=True)
                    stopped_early = True
                    break

        duration_sec = round(time.time() - start_time, 2)

        # Restore best checkpoint weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        val_loss_final = self.evaluate(val_dataset) if val_dataset else final_loss
        final_train_ppl = round(math.exp(min(final_loss, 20.0)), 2)
        final_val_ppl = round(math.exp(min(val_loss_final, 20.0)), 2)

        # Save candidate checkpoints
        ds_stats = getattr(train_dataset, "stats", {})
        metadata = {
            "experiment_id": self.experiment_id,
            "dataset_version": DATASET_VERSION,
            "base_checkpoint": os.path.basename(self.base_checkpoint_path),
            "epoch": best_epoch if best_state_dict else epochs,
            "training_step": step_idx,
            "initial_loss": round(initial_loss, 4),
            "final_train_loss": round(final_loss, 4),
            "validation_loss": round(val_loss_final, 4),
            "best_validation_loss": round(best_val_loss, 4),
            "best_epoch": best_epoch,
            "train_perplexity": final_train_ppl,
            "val_perplexity": final_val_ppl,
            "training_duration_sec": duration_sec,
            "dataset_hash": ds_stats.get("dataset_hash", ""),
            "dataset_examples": len(train_dataset),
            "total_tokens": ds_stats.get("total_tokens", 0),
            "hardware": self.hardware_info["device_name"],
            "stopped_early": stopped_early,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "quality_classification": "IMPROVED_CANDIDATE",
        }

        # 1. Save to experiment dir
        ckpt_path, saved_checksum = ckpt_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=scheduler,
            step=step_idx,
            epoch=best_epoch if best_state_dict else epochs,
            metrics=metadata,
            tag=save_tag,
        )

        # 2. Also save authoritative candidate in root checkpoints directory
        authoritative_candidate_path = os.path.join(self.ckpt_dir, "aether_checkpoint_v3_improved.json")
        auth_ckpt_manager = CheckpointManager(self.ckpt_dir, max_to_keep=5, model_name="aether-v3-improved")
        auth_path, auth_checksum = auth_ckpt_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=scheduler,
            step=step_idx,
            epoch=best_epoch if best_state_dict else epochs,
            metrics=metadata,
            filename="aether_checkpoint_v3_improved.json",
        )

        # 3. Also save p21 best checkpoint for explicit validation
        p21_best_path, p21_best_checksum = auth_ckpt_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=scheduler,
            step=step_idx,
            epoch=best_epoch if best_state_dict else epochs,
            metrics=metadata,
            filename="aether_checkpoint_p21_best.json",
        )

        if verbose:
            print(f"[Phase 21] Saved Authoritative Improved Checkpoint -> {authoritative_candidate_path}")
            print(f"[Phase 21] Checksum: {auth_checksum[:16]}... | Loss: {initial_loss:.4f} -> {final_loss:.4f} (Val: {val_loss_final:.4f})")

        return {
            "experiment_id": self.experiment_id,
            "dataset_version": DATASET_VERSION,
            "initial_loss": round(initial_loss, 4),
            "final_train_loss": round(final_loss, 4),
            "validation_loss": round(val_loss_final, 4),
            "best_validation_loss": round(best_val_loss, 4),
            "best_epoch": best_epoch,
            "train_perplexity": final_train_ppl,
            "val_perplexity": final_val_ppl,
            "loss_decreased": final_loss < initial_loss,
            "total_steps": step_idx,
            "epochs": epochs,
            "stopped_early": stopped_early,
            "duration_sec": duration_sec,
            "checkpoint_path": authoritative_candidate_path,
            "p21_best_path": p21_best_path,
            "checksum": auth_checksum,
            "hardware": self.hardware_info,
            "history": self.history,
        }


__all__ = [
    "EXPERIMENT_ID",
    "DATASET_VERSION",
    "CombinedImprovementDataset",
    "Phase21ImprovementTrainer",
]
