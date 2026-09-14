"""
AETHER MODEL — Feed-Forward Network
Position-wise Feed-Forward Network with GELU activation.
Supports forward pass, analytical backpropagation, and parameter state serialization.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff

        scale1 = 1.0 / math.sqrt(d_model)
        scale2 = 1.0 / math.sqrt(d_ff)

        i_m = np.arange(d_model, dtype=np.float64)[:, None]
        j_f = np.arange(d_ff, dtype=np.float64)[None, :]
        self.w1: np.ndarray = (((i_m * 7 + j_f * 13) % 100) / 100.0 - 0.5) * scale1
        self.b1: np.ndarray = 0.01 * (j_f[0] % 5)

        i_f = np.arange(d_ff, dtype=np.float64)[:, None]
        j_m = np.arange(d_model, dtype=np.float64)[None, :]
        self.w2: np.ndarray = (((i_f * 17 + j_m * 19) % 100) / 100.0 - 0.5) * scale2
        self.b2: np.ndarray = 0.01 * (j_m[0] % 3)

        self.grad_w1: np.ndarray = np.zeros((d_model, d_ff), dtype=np.float64)
        self.grad_b1: np.ndarray = np.zeros(d_ff, dtype=np.float64)
        self.grad_w2: np.ndarray = np.zeros((d_ff, d_model), dtype=np.float64)
        self.grad_b2: np.ndarray = np.zeros(d_model, dtype=np.float64)

        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (x_arr, h1_pre, h1_act)

    SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)

    @classmethod
    def _gelu_np(cls, x: np.ndarray) -> np.ndarray:
        """Vectorized Gaussian Error Linear Unit (GELU) approximation."""
        return 0.5 * x * (1.0 + np.tanh(cls.SQRT_2_OVER_PI * (x + 0.044715 * (x * x * x))))

    @classmethod
    def _d_gelu_np(cls, x: np.ndarray) -> np.ndarray:
        """Vectorized analytical derivative of GELU approximation."""
        u = cls.SQRT_2_OVER_PI * (x + 0.044715 * (x * x * x))
        tanh_u = np.tanh(u)
        du_dx = cls.SQRT_2_OVER_PI * (1.0 + 0.134145 * (x * x))
        return 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u * tanh_u) * du_dx

    def forward(self, x: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size == 0:
            return x_arr

        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        h1_pre = x_arr @ self.w1 + self.b1
        h1_act = self._gelu_np(h1_pre)
        output = h1_act @ self.w2 + self.b2
        self._cache = (x_arr, h1_pre, h1_act)

        if is_1d:
            return output.reshape(-1)
        return output

    def backward(self, grad_output: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        grad_output: [seq_len, d_model] or [d_model]
        Returns grad_input: [seq_len, d_model] or [d_model]
        """
        if self._cache is None:
            g_out = np.asarray(grad_output, dtype=np.float64)
            return np.zeros_like(g_out)

        x_arr, h1_pre, h1_act = self._cache
        g_out = np.asarray(grad_output, dtype=np.float64)
        is_1d = (g_out.ndim == 1)
        if is_1d:
            g_out = g_out.reshape(1, -1)

        self.grad_b2 += np.sum(g_out, axis=0)
        self.grad_w2 += h1_act.T @ g_out

        grad_h1_act = g_out @ self.w2.T
        grad_h1_pre = grad_h1_act * self._d_gelu_np(h1_pre)

        self.grad_b1 += np.sum(grad_h1_pre, axis=0)
        self.grad_w1 += x_arr.T @ grad_h1_pre

        grad_x = grad_h1_pre @ self.w1.T
        if is_1d:
            return grad_x.reshape(-1)
        return grad_x

    def zero_grad(self) -> None:
        self.grad_w1.fill(0.0)
        self.grad_b1.fill(0.0)
        self.grad_w2.fill(0.0)
        self.grad_b2.fill(0.0)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    def set_parameters(self, params: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"FeedForwardNetwork params must be a dict, got {type(params).__name__}")
        required_keys = {"w1", "b1", "w2", "b2"}
        missing_keys = required_keys - set(params.keys())
        if missing_keys:
            raise KeyError(f"FeedForwardNetwork parameters missing required keys: {sorted(missing_keys)}")
        if strict:
            extra_keys = set(params.keys()) - required_keys
            if extra_keys:
                raise KeyError(f"FeedForwardNetwork received unexpected parameters: {sorted(extra_keys)}")

        expected_shapes = {
            "w1": (self.d_model, self.d_ff),
            "b1": (self.d_ff,),
            "w2": (self.d_ff, self.d_model),
            "b2": (self.d_model,),
        }
        for key, exp_shape in expected_shapes.items():
            arr = np.array(params[key], dtype=np.float64)
            if arr.shape != exp_shape:
                raise ValueError(f"FeedForwardNetwork.{key} shape mismatch: expected {exp_shape}, got {arr.shape}")

        self.w1 = np.array(params["w1"], dtype=np.float64)
        self.b1 = np.array(params["b1"], dtype=np.float64)
        self.w2 = np.array(params["w2"], dtype=np.float64)
        self.b2 = np.array(params["b2"], dtype=np.float64)
        self.grad_w1 = np.zeros((self.d_model, self.d_ff), dtype=np.float64)
        self.grad_b1 = np.zeros(self.d_ff, dtype=np.float64)
        self.grad_w2 = np.zeros((self.d_ff, self.d_model), dtype=np.float64)
        self.grad_b2 = np.zeros(self.d_model, dtype=np.float64)
        self.zero_grad()
