"""
AETHER MODEL — AdamW and SGD Optimizers
Pure-Python numerical optimizers with decoupled weight decay, momentum, gradient clipping,
and learning rate scheduling.
"""

import math
import numpy as np
from typing import List, Tuple, Any, Dict, Optional

class AdamW:
    def __init__(
        self,
        named_parameters: List[Tuple[str, Any, Any]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.named_parameters = named_parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

        # Maintain first and second moments for each parameter
        self.exp_avg: Dict[str, np.ndarray] = {}
        self.exp_avg_sq: Dict[str, np.ndarray] = {}

        for name, param, _ in self.named_parameters:
            param_arr = np.asarray(param, dtype=np.float64)
            self.exp_avg[name] = np.zeros_like(param_arr)
            self.exp_avg_sq[name] = np.zeros_like(param_arr)

    def clip_gradients(self) -> float:
        """Clips global L2 norm of gradients."""
        if self.max_grad_norm is None or self.max_grad_norm <= 0.0:
            return 0.0

        total_norm_sq = 0.0
        for name, param, grad in self.named_parameters:
            g_arr = np.asarray(grad, dtype=np.float64)
            total_norm_sq += float(np.sum(g_arr * g_arr))

        total_norm = math.sqrt(total_norm_sq)
        if total_norm > self.max_grad_norm:
            scale = self.max_grad_norm / (total_norm + 1e-6)
            for name, param, grad in self.named_parameters:
                if isinstance(grad, np.ndarray):
                    grad *= scale
                elif isinstance(grad, list):
                    if isinstance(grad[0], list):
                        for r in range(len(grad)):
                            for c in range(len(grad[0])):
                                grad[r][c] *= scale
                    else:
                        for i in range(len(grad)):
                            grad[i] *= scale

        return total_norm

    def step(self, lr_override: Optional[float] = None) -> None:
        """Executes optimized AdamW parameter update step."""
        self.step_count += 1
        current_lr = lr_override if lr_override is not None else self.lr

        # 1. Gradient clipping
        self.clip_gradients()

        # 2. Bias corrections precomputation
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
        inv_bc1 = current_lr / bias_correction1
        inv_sqrt_bc2 = 1.0 / math.sqrt(bias_correction2)

        b1, one_minus_b1 = self.beta1, 1.0 - self.beta1
        b2, one_minus_b2 = self.beta2, 1.0 - self.beta2
        eps = self.eps

        # 3. Update parameters
        for name, param, grad in self.named_parameters:
            m = self.exp_avg[name]
            v = self.exp_avg_sq[name]

            decay = self.weight_decay if ("bias" not in name and "gamma" not in name and "beta" not in name) else 0.0
            decay_factor = current_lr * decay

            p_arr = np.asarray(param, dtype=np.float64)
            g_arr = np.asarray(grad, dtype=np.float64)

            m[:] = b1 * m + one_minus_b1 * g_arr
            v[:] = b2 * v + one_minus_b2 * (g_arr * g_arr)

            denom = (np.sqrt(v) * inv_sqrt_bc2) + eps
            update = (decay_factor * p_arr) + (inv_bc1 * m / denom)

            if isinstance(param, np.ndarray):
                param -= update
            elif isinstance(param, list):
                upd_list = update.tolist()
                if isinstance(param[0], list):
                    for r in range(len(param)):
                        for c in range(len(param[0])):
                            param[r][c] -= upd_list[r][c]
                else:
                    for i in range(len(param)):
                        param[i] -= upd_list[i]

    def zero_grad(self) -> None:
        for name, param, grad in self.named_parameters:
            if isinstance(grad, np.ndarray):
                grad.fill(0.0)
            elif isinstance(grad, list):
                if isinstance(grad[0], list):
                    for r in range(len(grad)):
                        for c in range(len(grad[0])):
                            grad[r][c] = 0.0
                else:
                    for i in range(len(grad)):
                        grad[i] = 0.0


    def get_state_dict(self) -> Dict[str, Any]:
        """Serializes optimizer state (moments and step count)."""
        return {
            "step_count": self.step_count,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "exp_avg": {k: v.tolist() for k, v in self.exp_avg.items()},
            "exp_avg_sq": {k: v.tolist() for k, v in self.exp_avg_sq.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores optimizer moments and step count from state dictionary."""
        self.step_count = int(state_dict.get("step_count", 0))
        if "lr" in state_dict:
            self.lr = float(state_dict["lr"])
        if "weight_decay" in state_dict:
            self.weight_decay = float(state_dict["weight_decay"])

        exp_avg_data = state_dict.get("exp_avg", {})
        for name, arr in exp_avg_data.items():
            if name in self.exp_avg:
                self.exp_avg[name] = np.array(arr, dtype=np.float64)

        exp_avg_sq_data = state_dict.get("exp_avg_sq", {})
        for name, arr in exp_avg_sq_data.items():
            if name in self.exp_avg_sq:
                self.exp_avg_sq[name] = np.array(arr, dtype=np.float64)


from training.scheduler import LRScheduler, ConstantLR

__all__ = ["AdamW", "LRScheduler", "ConstantLR"]

