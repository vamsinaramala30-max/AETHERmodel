"""
AETHER MODEL — Token Embeddings
Maps discrete token IDs into d_model dimensional dense vector space representations.
Supports forward lookup, parameter management, and analytical gradient accumulation.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Union

class TokenEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        bound = 1.0 / math.sqrt(d_model)
        # Xavier uniform initialization
        i_arr = np.arange(vocab_size, dtype=np.float64)[:, None]
        j_arr = np.arange(d_model, dtype=np.float64)[None, :]
        self.weight: np.ndarray = (((i_arr * 31 + j_arr * 17 + 7) % 1000) / 1000.0 - 0.5) * 2.0 * bound
        self.grad_weight: np.ndarray = np.zeros((vocab_size, d_model), dtype=np.float64)
        self._last_input_ids: Optional[List[int]] = None


    UNK_TOKEN_ID: int = 1  # <unk> token ID — must match tokenizer.py

    def forward(self, input_ids: Union[int, List[int], np.ndarray, Any]) -> np.ndarray:
        if isinstance(input_ids, (int, np.integer)):
            input_arr = np.array([int(input_ids)], dtype=np.int64)
            self._last_input_ids = [int(input_ids)]
        elif isinstance(input_ids, list):
            if not input_ids:
                self._last_input_ids = []
                return np.zeros((0, self.d_model), dtype=np.float64)
            input_arr = np.asarray(input_ids, dtype=np.int64).reshape(-1)
            self._last_input_ids = input_arr.tolist()
        elif isinstance(input_ids, np.ndarray):
            if input_ids.size == 0:
                self._last_input_ids = []
                return np.zeros((0, self.d_model), dtype=np.float64)
            input_arr = input_ids.astype(np.int64).reshape(-1)
            self._last_input_ids = input_arr.tolist()
        else:
            try:
                seq_list = list(input_ids)
                if not seq_list:
                    self._last_input_ids = []
                    return np.zeros((0, self.d_model), dtype=np.float64)
                input_arr = np.asarray(seq_list, dtype=np.int64).reshape(-1)
                self._last_input_ids = input_arr.tolist()
            except Exception as ex:
                raise TypeError(f"Invalid input_ids type for TokenEmbedding: {type(input_ids).__name__}") from ex

        # Safe fallback UNK id
        unk_id = self.UNK_TOKEN_ID if (0 <= self.UNK_TOKEN_ID < self.vocab_size) else 0

        # Clamp out-of-range IDs to UNK — never silently alias via modulo
        if self.vocab_size > 0:
            out_of_range = (input_arr < 0) | (input_arr >= self.vocab_size)
            indices = np.where(out_of_range, unk_id, input_arr)
            return self.weight[indices]
        else:
            return np.zeros((input_arr.size, self.d_model), dtype=np.float64)

    def backward(self, grad_output: Union[np.ndarray, List[List[float]], List[float]]) -> None:
        """
        Accumulate gradients into grad_weight for corresponding token IDs.
        grad_output shape: [seq_len, d_model]
        """
        if self._last_input_ids is None or len(self._last_input_ids) == 0:
            return
        g_out = np.asarray(grad_output, dtype=np.float64)
        if g_out.ndim == 1:
            g_out = g_out.reshape(1, -1)
        elif g_out.ndim > 2:
            g_out = g_out.reshape(-1, self.d_model)

        unk_id = self.UNK_TOKEN_ID if (0 <= self.UNK_TOKEN_ID < self.vocab_size) else 0

        for pos, token_id in enumerate(self._last_input_ids):
            if pos >= g_out.shape[0]:
                break
            target_id = token_id if (self.vocab_size > 0 and 0 <= token_id < self.vocab_size) else unk_id
            if 0 <= target_id < self.vocab_size:
                self.grad_weight[target_id] += g_out[pos]

    def zero_grad(self) -> None:
        self.grad_weight.fill(0.0)

    def get_parameters(self) -> Dict[str, Any]:
        return {"weight": self.weight.tolist()}

    def set_parameters(self, params: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"TokenEmbedding params must be a dict, got {type(params).__name__}")
        if "weight" not in params:
            raise KeyError("TokenEmbedding parameters missing required key 'weight'")
        if strict:
            extra_keys = set(params.keys()) - {"weight"}
            if extra_keys:
                raise KeyError(f"TokenEmbedding received unexpected parameters: {extra_keys}")

        w_arr = np.array(params["weight"], dtype=np.float64)
        expected_shape = (self.vocab_size, self.d_model)
        if w_arr.shape != expected_shape:
            raise ValueError(f"TokenEmbedding shape mismatch: expected {expected_shape}, got {w_arr.shape}")

        self.weight = w_arr
        self.grad_weight = np.zeros(expected_shape, dtype=np.float64)
        self.zero_grad()

