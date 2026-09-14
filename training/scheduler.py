"""
AETHER MODEL — Learning Rate Schedulers (Phase 10)

Implements:
- LRScheduler: Linear warmup followed by Cosine Annealing decay.
- LinearWarmupScheduler: Linear warmup with constant or linear decay.
- CosineAnnealingLR: Pure cosine annealing scheduler.
- ConstantLR: Fixed learning rate scheduler.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


class LRScheduler:
    """
    Standard Warmup + Cosine Annealing learning rate scheduler.

    Schedule:
    - Steps [0 .. warmup_steps):
        lr = min_lr + (base_lr - min_lr) * (step / warmup_steps)
    - Steps [warmup_steps .. total_steps]:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + cos(pi * progress))
    - Steps > total_steps:
        lr = min_lr
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int = 10,
        total_steps: int = 100,
        min_lr: float = 1e-5,
    ) -> None:
        self.base_lr = float(base_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(int(total_steps), self.warmup_steps + 1)
        self.min_lr = float(min_lr)
        self.current_step = 0

    def get_lr(self, step: Optional[int] = None) -> float:
        """Computes learning rate for a specific step."""
        s = self.current_step if step is None else int(step)

        if self.warmup_steps > 0 and s < self.warmup_steps:
            # Linear Warmup phase
            alpha = float(s) / float(max(1, self.warmup_steps))
            return self.min_lr + (self.base_lr - self.min_lr) * alpha

        if s >= self.total_steps:
            return self.min_lr

        # Cosine Annealing phase
        decay_steps = self.total_steps - self.warmup_steps
        progress = float(s - self.warmup_steps) / float(max(1, decay_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def step(self) -> float:
        """Advances step by 1 and returns current learning rate."""
        self.current_step += 1
        return self.get_lr(self.current_step)

    def get_state_dict(self) -> Dict[str, Any]:
        """Returns serializable scheduler state dictionary."""
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores scheduler state from dictionary."""
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.min_lr = float(state_dict.get("min_lr", self.min_lr))
        self.current_step = int(state_dict.get("current_step", 0))


class ConstantLR(LRScheduler):
    """Fixed learning rate scheduler."""

    def __init__(self, lr: float) -> None:
        super().__init__(base_lr=lr, warmup_steps=0, total_steps=1000000, min_lr=lr)

    def get_lr(self, step: Optional[int] = None) -> float:
        return self.base_lr


__all__ = ["LRScheduler", "ConstantLR"]
