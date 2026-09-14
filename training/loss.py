"""
AETHER MODEL — Causal Cross-Entropy Loss & Analytical Gradient Engine (Phase 10)

Mathematically implements:
    Loss = - (1 / N_active) * sum_{t in active} log P(y_t | x_{0:t})
    P(y_t) = exp(z_{t, y_t} - max(z_t)) / sum_j exp(z_{t, j} - max(z_t))
    grad_logits_{t, j} = (P_{t, j} - 1_{j == y_t}) / N_active

Features:
- Numerical stability via log-sum-exp trick (subtract max logit).
- Active token masking: focuses loss on assistant response tokens and EOS.
- Automatic padding token (PAD_TOKEN_ID) exclusion.
- Analytical gradient calculation w.r.t logits matrix for backpropagation.
- Token-level and EOS-level accuracy metrics.
- Mathematically valid perplexity computation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Special token IDs
# PAD is universally 0 in both legacy and BPE modes.
# EOS varies: legacy=2, BPE=3. NEVER hardcode EOS here.
# Use tokenizer.token_to_id.get('<eos>', 2) or SpecialTokens.from_tokenizer().
PAD_TOKEN_ID: int = 0
_LEGACY_EOS_FALLBACK: int = 2  # used ONLY as a default arg sentinel below


class CausalCrossEntropyLoss:
    """
    Categorical Cross-Entropy Loss designed specifically for causal next-token language modeling.
    """

    def __init__(
        self,
        pad_token_id: int = PAD_TOKEN_ID,
        eos_token_id: int = _LEGACY_EOS_FALLBACK,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def __call__(
        self,
        logits: Union[np.ndarray, List[List[float]]],
        target_ids: Union[np.ndarray, List[int]],
        asst_start_idx: int = 0,
        mask: Optional[Union[np.ndarray, List[float], List[int]]] = None,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Computes loss, analytical gradient w.r.t logits, and accuracy metrics.

        Args:
            logits: Logits matrix of shape [seq_len, vocab_size]
            target_ids: Target token IDs of shape [seq_len]
            asst_start_idx: Position where assistant response tokens begin (prompt tokens masked before this)
            mask: Optional explicit binary mask of shape [seq_len] (1.0 = active, 0.0 = masked)

        Returns:
            (loss, grad_logits, metrics_dict)
        """
        logits_np = np.asarray(logits, dtype=np.float64)
        targets_np = np.asarray(target_ids, dtype=np.int64)

        # Handle 1D / empty edge cases
        if logits_np.ndim == 1:
            logits_np = logits_np.reshape(1, -1)
            targets_np = targets_np.reshape(1)

        seq_len, vocab_size = logits_np.shape
        if seq_len == 0 or targets_np.size == 0:
            return 0.0, np.zeros_like(logits_np), {
                "token_accuracy": 0.0,
                "eos_accuracy": 0.0,
                "perplexity": 1.0,
                "active_tokens": 0,
            }

        # Build active token mask
        if mask is not None:
            active_mask = np.asarray(mask, dtype=bool).copy()
        else:
            # Start calculating loss from assistant response tokens (masking system/user prompt)
            start_t = max(0, min(asst_start_idx - 1, seq_len - 1)) if asst_start_idx > 0 else 0
            active_mask = np.zeros(seq_len, dtype=bool)
            active_mask[start_t:] = True

        # Always mask out padding tokens
        active_mask &= (targets_np != self.pad_token_id)
        # Always mask out out-of-vocabulary target IDs
        active_mask &= (targets_np >= 0) & (targets_np < vocab_size)

        active_indices = np.where(active_mask)[0]
        active_count = len(active_indices)

        if active_count == 0:
            grad_logits = np.zeros_like(logits_np)
            return 0.0, grad_logits, {
                "token_accuracy": 0.0,
                "eos_accuracy": 0.0,
                "perplexity": 1.0,
                "active_tokens": 0,
            }

        # Extract active logits and targets
        active_logits = logits_np[active_indices]  # [active_count, vocab_size]
        active_targets = targets_np[active_indices]  # [active_count]

        # Numerically stable Softmax: subtract max per row
        max_logits = np.max(active_logits, axis=-1, keepdims=True)
        shifted_logits = active_logits - max_logits
        exps = np.exp(shifted_logits)
        sum_exps = np.sum(exps, axis=-1, keepdims=True)
        sum_exps = np.where(sum_exps == 0.0, 1e-12, sum_exps)
        probs = exps / sum_exps  # [active_count, vocab_size]

        # Categorical Cross-Entropy Loss
        target_probs = probs[np.arange(active_count), active_targets]
        target_probs = np.clip(target_probs, 1e-12, 1.0)
        token_losses = -np.log(target_probs)

        if self.label_smoothing > 0.0:
            smooth_loss = -np.mean(np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
            token_losses = (1.0 - self.label_smoothing) * token_losses + self.label_smoothing * smooth_loss

        if self.reduction == "mean":
            avg_loss = float(np.mean(token_losses))
        elif self.reduction == "sum":
            avg_loss = float(np.sum(token_losses))
        else:
            avg_loss = float(np.mean(token_losses))

        # Metrics computation
        predicted = np.argmax(active_logits, axis=-1)
        token_acc = float(np.mean(predicted == active_targets))

        eos_mask = (active_targets == self.eos_token_id)
        if np.any(eos_mask):
            eos_acc = float(np.mean(predicted[eos_mask] == self.eos_token_id))
        else:
            eos_acc = 1.0 if np.any(predicted == self.eos_token_id) else 0.0

        ppl = float(math.exp(min(avg_loss, 20.0)))

        # Analytical Gradients w.r.t Logits:
        # For active positions: grad = (prob - 1_{target}) / active_count
        # For masked positions: grad = 0.0
        grad_logits = np.zeros_like(logits_np)
        grad_active = probs.copy()

        if self.label_smoothing > 0.0:
            uniform_target = self.label_smoothing / float(vocab_size)
            one_hot_weight = 1.0 - self.label_smoothing
            grad_active -= uniform_target
            grad_active[np.arange(active_count), active_targets] -= one_hot_weight
        else:
            grad_active[np.arange(active_count), active_targets] -= 1.0

        if self.reduction == "mean":
            grad_active /= float(active_count)

        grad_logits[active_indices] = grad_active

        metrics = {
            "token_accuracy": round(token_acc, 4),
            "eos_accuracy": round(eos_acc, 4),
            "perplexity": round(ppl, 4),
            "active_tokens": active_count,
        }

        return avg_loss, grad_logits, metrics


def compute_cross_entropy(
    logits_matrix: Union[np.ndarray, List[List[float]]],
    target_ids: Union[np.ndarray, List[int]],
    asst_start_idx: int = 0,
    pad_token_id: int = PAD_TOKEN_ID,
    label_smoothing: float = 0.0,
) -> Tuple[float, List[List[float]], Dict[str, float]]:
    """
    Functional convenience wrapper for CausalCrossEntropyLoss.
    """
    criterion = CausalCrossEntropyLoss(
        pad_token_id=pad_token_id,
        label_smoothing=label_smoothing,
        reduction="mean",
    )
    loss, grad_np, metrics = criterion(
        logits=logits_matrix,
        target_ids=target_ids,
        asst_start_idx=asst_start_idx,
    )
    return loss, grad_np.tolist(), metrics


__all__ = ["CausalCrossEntropyLoss", "compute_cross_entropy", "PAD_TOKEN_ID"]
