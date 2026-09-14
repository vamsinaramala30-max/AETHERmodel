"""
AETHER MODEL — LM Output Head
Projects output hidden states (d_model) onto vocabulary space (vocab_size) to yield token logits.
Supports single-token projection, sequence-wide projection, analytical backpropagation, and state dict serialization.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Union

class OutputHead:
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size

        scale = 1.0 / math.sqrt(d_model)
        i_m = np.arange(d_model, dtype=np.float64)[:, None]
        j_v = np.arange(vocab_size, dtype=np.float64)[None, :]

        self.weight: np.ndarray = (((i_m * 23 + j_v * 29) % 100) / 100.0 - 0.5) * scale
        self.bias: np.ndarray = np.zeros(vocab_size, dtype=np.float64)


        self.grad_weight: np.ndarray = np.zeros((d_model, vocab_size), dtype=np.float64)
        self.grad_bias: np.ndarray = np.zeros(vocab_size, dtype=np.float64)

        self._last_hidden_states: Optional[np.ndarray] = None

    def forward(self, hidden_state: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Projects a single token's hidden state vector [d_model] to vocab_size logits.
        """
        h_arr = np.asarray(hidden_state, dtype=np.float64)
        logits = h_arr @ self.weight + self.bias
        return logits

    def forward_all(self, hidden_states: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Projects full sequence of hidden states [seq_len, d_model] to logits [seq_len, vocab_size].
        """
        h_arr = np.asarray(hidden_states, dtype=np.float64)
        self._last_hidden_states = h_arr
        if h_arr.size == 0:
            return np.zeros((0, self.vocab_size), dtype=np.float64)
        logits = h_arr @ self.weight + self.bias
        return logits

    def backward(self, grad_logits: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Analytical OutputHead backward pass.
        grad_logits: [seq_len, vocab_size]
        Returns grad_hidden_states: [seq_len, d_model]
        """
        if self._last_hidden_states is None:
            g_z = np.asarray(grad_logits, dtype=np.float64)
            return np.zeros((g_z.shape[0], self.d_model), dtype=np.float64)

        g_z = np.asarray(grad_logits, dtype=np.float64)
        h_t = self._last_hidden_states

        self.grad_bias += np.sum(g_z, axis=0)
        self.grad_weight += h_t.T @ g_z

        grad_hidden = g_z @ self.weight.T
        return grad_hidden

    def zero_grad(self) -> None:
        self.grad_weight.fill(0.0)
        self.grad_bias.fill(0.0)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
        }

    def set_parameters(self, params: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"OutputHead params must be a dict, got {type(params).__name__}")
        required_keys = {"weight", "bias"}
        missing_keys = required_keys - set(params.keys())
        if missing_keys:
            raise KeyError(f"OutputHead parameters missing required keys: {sorted(missing_keys)}")
        if strict:
            extra_keys = set(params.keys()) - required_keys
            if extra_keys:
                raise KeyError(f"OutputHead received unexpected parameters: {sorted(extra_keys)}")

        expected_shapes = {
            "weight": (self.d_model, self.vocab_size),
            "bias": (self.vocab_size,),
        }
        for key, exp_shape in expected_shapes.items():
            arr = np.array(params[key], dtype=np.float64)
            if arr.shape != exp_shape:
                raise ValueError(f"OutputHead.{key} shape mismatch: expected {exp_shape}, got {arr.shape}")

        self.weight = np.array(params["weight"], dtype=np.float64)
        self.bias = np.array(params["bias"], dtype=np.float64)
        self.grad_weight = np.zeros((self.d_model, self.vocab_size), dtype=np.float64)
        self.grad_bias = np.zeros(self.vocab_size, dtype=np.float64)
        self.zero_grad()

