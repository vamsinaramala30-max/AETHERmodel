"""
AETHER MODEL — Layer Normalization
Standard LayerNorm over hidden dimensions with learnable scale (gamma) and shift (beta).
Supports forward normalization, caching, analytical backpropagation, and parameter state dict.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma: np.ndarray = np.ones(d_model, dtype=np.float64)
        self.beta: np.ndarray = np.zeros(d_model, dtype=np.float64)

        self.grad_gamma: np.ndarray = np.zeros(d_model, dtype=np.float64)
        self.grad_beta: np.ndarray = np.zeros(d_model, dtype=np.float64)

        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (x_hat, mean, std)

    def forward(self, x: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size == 0:
            return x_arr

        # Handle 1D input (d_model,) -> (1, d_model)
        is_1d = (x_arr.ndim == 1)
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        mean = np.mean(x_arr, axis=-1, keepdims=True)
        var = np.var(x_arr, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_hat = (x_arr - mean) / std
        output = self.gamma * x_hat + self.beta
        self._cache = (x_hat, mean, std)

        if is_1d:
            return output.reshape(-1)
        return output

    def backward(self, grad_output: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Analytical LayerNorm backward pass.
        grad_output: [seq_len, d_model] or [d_model]
        Returns grad_input: [seq_len, d_model] or [d_model]
        """
        if self._cache is None:
            g_out = np.asarray(grad_output, dtype=np.float64)
            return np.zeros_like(g_out)

        x_hat, mean, std = self._cache
        g_out = np.asarray(grad_output, dtype=np.float64)
        is_1d = (g_out.ndim == 1)
        if is_1d:
            g_out = g_out.reshape(1, -1)

        D = float(self.d_model)

        self.grad_gamma += np.sum(g_out * x_hat, axis=0)
        self.grad_beta += np.sum(g_out, axis=0)

        sum_gout_gamma = np.sum(g_out * self.gamma, axis=-1, keepdims=True)
        sum_gout_gamma_xhat = np.sum(g_out * self.gamma * x_hat, axis=-1, keepdims=True)

        grad_input = (self.gamma * g_out - (sum_gout_gamma / D) - (x_hat * sum_gout_gamma_xhat / D)) / std
        if is_1d:
            return grad_input.reshape(-1)
        return grad_input

    def zero_grad(self) -> None:
        self.grad_gamma.fill(0.0)
        self.grad_beta.fill(0.0)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist()
        }

    def set_parameters(self, params: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"LayerNorm params must be a dict, got {type(params).__name__}")
        required_keys = {"gamma", "beta"}
        missing_keys = required_keys - set(params.keys())
        if missing_keys:
            raise KeyError(f"LayerNorm parameters missing required keys: {sorted(missing_keys)}")
        if strict:
            extra_keys = set(params.keys()) - required_keys
            if extra_keys:
                raise KeyError(f"LayerNorm received unexpected parameters: {sorted(extra_keys)}")

        expected_shape = (self.d_model,)
        for key in required_keys:
            arr = np.array(params[key], dtype=np.float64)
            if arr.shape != expected_shape:
                raise ValueError(f"LayerNorm.{key} shape mismatch: expected {expected_shape}, got {arr.shape}")

        self.gamma = np.array(params["gamma"], dtype=np.float64)
        self.beta = np.array(params["beta"], dtype=np.float64)
        self.grad_gamma = np.zeros(self.d_model, dtype=np.float64)
        self.grad_beta = np.zeros(self.d_model, dtype=np.float64)
        self.zero_grad()
