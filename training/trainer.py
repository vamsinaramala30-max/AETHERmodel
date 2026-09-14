"""
AETHER MODEL — Native Transformer Trainer (Phase 10 Production Foundation)

Executes causal cross-entropy optimization using AdamW and analytical backpropagation.
Tracks training and validation losses, gradient norms, step metrics, and saves verified checkpoints.
Implements early stopping, learning rate warmup/cosine decay, and validation-based best-checkpoint selection.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.checkpoint import CheckpointManager
from training.hardware import HardwareInspector
from training.loss import CausalCrossEntropyLoss, compute_cross_entropy
from training.optimizer import AdamW
from training.scheduler import LRScheduler


class AetherTrainer:
    """
    Authoritative training engine for Aether Transformer models.
    """

    def __init__(
        self,
        model: Optional[AetherModel] = None,
        config: Optional[ModelConfig] = None,
        lr: float = 2e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        self.config = config or ModelConfig()
        self.model = model or AetherModel(self.config)

        # Load canonical BPE tokenizer if artifact exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bpe_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        if os.path.exists(bpe_path):
            self.tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
        else:
            self.tokenizer = AetherTokenizer()

        self.hardware_info = HardwareInspector.detect()

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.label_smoothing = float(label_smoothing)

        pad_id = self.tokenizer.token_to_id.get("<pad>", 0)
        self.criterion = CausalCrossEntropyLoss(
            pad_token_id=pad_id,
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

    @staticmethod
    def compute_cross_entropy(
        logits_matrix: Union[np.ndarray, List[List[float]]],
        target_ids: Union[np.ndarray, List[int]],
        asst_start_idx: int = 0,
        pad_token_id: int = 0,
    ) -> Tuple[float, List[List[float]], Dict[str, float]]:
        """
        Computes sequence cross-entropy loss and gradients w.r.t logits matrix.
        Masks user/system prompt tokens and padding tokens, focusing loss on assistant response tokens and EOS.
        Returns: (avg_loss, grad_logits, metrics_dict)
        """
        return compute_cross_entropy(
            logits_matrix=logits_matrix,
            target_ids=target_ids,
            asst_start_idx=asst_start_idx,
            pad_token_id=pad_token_id,
        )

    def evaluate(self, val_dataset: Any) -> float:
        """
        Computes average cross-entropy loss across validation dataset without mutating model parameters.
        """
        if val_dataset is None or len(val_dataset) == 0:
            return 0.0

        total_val_loss = 0.0
        count = 0
        pad_id = self.tokenizer.token_to_id.get("<pad>", 0)

        for item in val_dataset:
            input_ids = item["input_ids"]
            target_ids = item["target_ids"]
            asst_idx = item.get("asst_start_idx", item.get("asst_start_pos", 0))

            logits = self.model.forward_all(input_ids)
            loss, _, _metrics = self.compute_cross_entropy(logits, target_ids, asst_idx, pad_token_id=pad_id)
            if math.isfinite(loss):
                total_val_loss += loss
                count += 1

        return round(total_val_loss / float(max(1, count)), 4)

    def _compute_gradient_norm(self) -> float:
        """Computes total L2 gradient norm across all model parameters."""
        total_norm_sq = 0.0
        for name, param, grad in self.model.architecture.get_named_parameters():
            g_arr = np.asarray(grad, dtype=np.float64)
            total_norm_sq += float(np.sum(g_arr * g_arr))
        return round(math.sqrt(total_norm_sq), 6)

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        epochs: int = 5,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
        patience: int = 5,
        min_delta: float = 0.001,
        save_tag: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes full causal language model training loop with analytical backpropagation and AdamW optimization.
        Implements early stopping based on validation loss.
        Saves checkpoint at the best validation loss state.
        Supports micro-batching via gradient_accumulation_steps.
        """
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        grad_accum = max(1, int(gradient_accumulation_steps))
        effective_batch_size = grad_accum
        total_examples = len(train_dataset) * epochs
        total_opt_steps = math.ceil(total_examples / grad_accum)
        warmup_steps = max(2, total_opt_steps // 10)
        scheduler = LRScheduler(base_lr=self.lr, warmup_steps=warmup_steps, total_steps=total_opt_steps)

        initial_loss = 0.0
        final_loss = 0.0
        step_idx = 0
        opt_step_idx = 0

        # Early stopping and best model tracking
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None
        patience_counter = 0
        stopped_early = False

        ckpt_manager = CheckpointManager(checkpoint_dir) if checkpoint_dir else None

        start_time = time.time()
        if verbose:
            print(f"=== [AETHER TRAINER] Starting training over {len(train_dataset)} examples for {epochs} epochs ===", flush=True)
            print(f"[Device] {self.hardware_info['device_name']} | LR={self.lr} | Warmup={warmup_steps} steps | Total Opt Steps={total_opt_steps} | GradAccum={grad_accum} (Effective Batch Size={effective_batch_size})", flush=True)
            print(f"[Early Stopping] patience={patience}, min_delta={min_delta}", flush=True)

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
                asst_idx = item.get("asst_start_idx", item.get("asst_start_pos", 0))

                # 1. Forward pass
                logits = self.model.forward_all(input_ids)

                # 2. Compute loss, grad_logits, and token-level metrics
                pad_id = self.tokenizer.token_to_id.get("<pad>", 0)
                loss, grad_logits, step_metrics = self.compute_cross_entropy(logits, target_ids, asst_idx, pad_token_id=pad_id)
                if not math.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at step {step_idx}: {loss}")

                if step_idx == 1:
                    initial_loss = loss

                epoch_token_acc.append(step_metrics.get("token_accuracy", 0.0))
                epoch_eos_acc.append(step_metrics.get("eos_accuracy", 0.0))

                # 3. Analytical backward pass (scaled by gradient accumulation if > 1)
                if grad_accum > 1:
                    scaled_grad_logits = [[g / float(grad_accum) for g in row] for row in grad_logits]
                    self.model.backward(scaled_grad_logits)
                else:
                    self.model.backward(grad_logits)

                epoch_loss += loss
                final_loss = loss

                # 4. Optimizer update step on accumulation boundary
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_dataset):
                    opt_step_idx += 1
                    current_lr = scheduler.get_lr(opt_step_idx)

                    grad_norm = self._compute_gradient_norm()
                    epoch_grad_norms.append(grad_norm)

                    self.optimizer.step(lr_override=current_lr)
                    self.model.zero_grad()

                    if verbose and (opt_step_idx % 20 == 0 or opt_step_idx == total_opt_steps or opt_step_idx == 1):
                        print(
                            f"Epoch {epoch}/{epochs} | Opt Step {opt_step_idx}/{total_opt_steps} (Sample {step_idx}/{total_examples}) | Loss: {loss:.4f} | "
                            f"TokenAcc: {step_metrics['token_accuracy']:.3f} | LR: {current_lr:.6f} | GradNorm: {grad_norm:.4f}",
                            flush=True,
                        )

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
                "effective_batch_size": effective_batch_size,
            })

            if verbose:
                print(
                    f"--> [Epoch {epoch} Complete] Avg Train Loss: {avg_epoch_loss:.4f} (PPL: {train_ppl}) | "
                    f"Val Loss: {val_loss:.4f} (PPL: {val_ppl}) | TokenAcc: {avg_token_acc:.3f} | "
                    f"EOSAcc: {avg_eos_acc:.3f} | AvgGradNorm: {avg_grad_norm:.4f}",
                    flush=True,
                )

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state_dict = json.loads(json.dumps(self.model.get_state_dict()))
                if verbose:
                    print(f"    [Best Checkpoint] New best val loss: {val_loss:.4f} at epoch {epoch}", flush=True)
            else:
                patience_counter += 1
                if verbose:
                    print(f"    [Early Stopping] No improvement for {patience_counter}/{patience} epochs", flush=True)
                if patience_counter >= patience:
                    if verbose:
                        print(f"    [Early Stopping] TRIGGERED at epoch {epoch}. Best was epoch {best_epoch} (val_loss: {best_val_loss:.4f})", flush=True)
                    stopped_early = True
                    break

        elapsed_sec = round(time.time() - start_time, 2)

        # Restore best checkpoint weights if available
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            if verbose:
                print(f"[AETHER TRAINER] Restored best checkpoint from epoch {best_epoch} (val_loss: {best_val_loss:.4f})", flush=True)

        val_loss_final = self.evaluate(val_dataset) if val_dataset else final_loss
        final_train_ppl = round(math.exp(min(final_loss, 20.0)), 2)
        final_val_ppl = round(math.exp(min(val_loss_final, 20.0)), 2)

        # Save checkpoint via CheckpointManager
        saved_checksum = ""
        ckpt_path = ""
        if ckpt_manager:
            ds_stats = getattr(train_dataset, "stats", {})
            metadata = {
                "epoch": best_epoch if best_state_dict else epochs,
                "training_step": step_idx,
                "initial_loss": round(initial_loss, 4),
                "final_train_loss": round(final_loss, 4),
                "validation_loss": round(val_loss_final, 4),
                "best_validation_loss": round(best_val_loss, 4),
                "best_epoch": best_epoch,
                "train_perplexity": final_train_ppl,
                "val_perplexity": final_val_ppl,
                "training_duration_sec": elapsed_sec,
                "dataset_hash": ds_stats.get("dataset_hash", ""),
                "dataset_examples": len(train_dataset),
                "total_tokens": ds_stats.get("total_tokens", 0),
                "hardware": self.hardware_info["device_name"],
                "stopped_early": stopped_early,
                "early_stopping_patience": patience,
                "quality_classification": "PRODUCTION_CANDIDATE" if val_loss_final < initial_loss else "TRAINING_ONLY",
            }
            ckpt_path, saved_checksum = ckpt_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=scheduler,
                step=step_idx,
                epoch=best_epoch if best_state_dict else epochs,
                metrics=metadata,
                tag=save_tag,
            )
            if verbose:
                print(f"[AETHER TRAINER] Saved checkpoint to {ckpt_path} (Checksum: {saved_checksum[:16]}...)")

        summary = {
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
            "duration_sec": elapsed_sec,
            "checkpoint_path": ckpt_path,
            "checksum": saved_checksum,
            "quality_classification": "PRODUCTION_CANDIDATE" if val_loss_final < initial_loss else "TRAINING_ONLY",
            "history": self.history,
        }

        return summary


__all__ = ["AetherTrainer"]
