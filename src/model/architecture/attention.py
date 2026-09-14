"""
AETHER MODEL — Multi-Head Self-Attention
Computes scaled dot-product self-attention across n_heads with causal sequence masking.
Supports forward pass, KV caching, exact analytical backpropagation, and state dict serialization.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = 1.0 / math.sqrt(d_model)
        i_m = np.arange(d_model, dtype=np.float64)[:, None]
        j_m = np.arange(d_model, dtype=np.float64)[None, :]

        self.q_proj: np.ndarray = (((i_m * 13 + j_m * 7) % 100) / 100.0 - 0.5) * scale
        self.k_proj: np.ndarray = (((i_m * 17 + j_m * 11) % 100) / 100.0 - 0.5) * scale
        self.v_proj: np.ndarray = (((i_m * 19 + j_m * 23) % 100) / 100.0 - 0.5) * scale
        self.out_proj: np.ndarray = (((i_m * 29 + j_m * 31) % 100) / 100.0 - 0.5) * scale

        self.grad_q_proj: np.ndarray = np.zeros((d_model, d_model), dtype=np.float64)
        self.grad_k_proj: np.ndarray = np.zeros((d_model, d_model), dtype=np.float64)
        self.grad_v_proj: np.ndarray = np.zeros((d_model, d_model), dtype=np.float64)
        self.grad_out_proj: np.ndarray = np.zeros((d_model, d_model), dtype=np.float64)

        self._cache: Optional[Dict[str, Any]] = None

    def forward(
        self,
        x: Union[np.ndarray, List[List[float]]],
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_kv_cache: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Executes causal multi-head self-attention forward pass.
        x: [seq_len, d_model] or [d_model] (1D single token)
        kv_cache: Optional tuple (cached_k, cached_v) of shape [cached_seq_len, d_model]
        use_kv_cache: Whether to return updated (keys, values) cache
        Returns: attn_out [seq_len, d_model] (and updated_kv_cache if requested)
        """
        x_arr = np.asarray(x, dtype=np.float64)

        # Handle 1D input [d_model] -> [1, d_model]
        if x_arr.ndim == 1:
            if x_arr.shape[0] == self.d_model:
                x_arr = x_arr.reshape(1, self.d_model)
            elif x_arr.size == 0:
                x_arr = np.zeros((0, self.d_model), dtype=np.float64)
            else:
                raise ValueError(
                    f"Input 1D array has feature dimension {x_arr.shape[0]}, expected d_model={self.d_model}"
                )
        elif x_arr.ndim == 2:
            if x_arr.shape[0] > 0 and x_arr.shape[1] != self.d_model:
                raise ValueError(
                    f"Input feature dimension {x_arr.shape[1]} does not match d_model {self.d_model}"
                )
        elif x_arr.ndim > 2:
            raise ValueError(f"Expected 2D input (seq_len, d_model), got shape {x_arr.shape}")

        seq_len = x_arr.shape[0]
        if seq_len == 0:
            empty_cache = (
                np.zeros((0, self.d_model), dtype=np.float64),
                np.zeros((0, self.d_model), dtype=np.float64)
            )
            if use_kv_cache or kv_cache is not None:
                return np.zeros((0, self.d_model), dtype=np.float64), empty_cache
            return np.zeros((0, self.d_model), dtype=np.float64)

        # 1. Project Q, K, V
        queries = x_arr @ self.q_proj  # [seq_len, d_model]
        keys_new = x_arr @ self.k_proj  # [seq_len, d_model]
        values_new = x_arr @ self.v_proj  # [seq_len, d_model]

        # 2. KV Cache Concatenation and validation
        if kv_cache is not None:
            if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
                raise ValueError(
                    f"kv_cache must be a tuple of (cached_k, cached_v), got {type(kv_cache).__name__}"
                )
            cached_k, cached_v = kv_cache
            if not isinstance(cached_k, np.ndarray):
                cached_k = np.asarray(cached_k, dtype=np.float64)
            if not isinstance(cached_v, np.ndarray):
                cached_v = np.asarray(cached_v, dtype=np.float64)

            if cached_k.ndim == 1:
                cached_k = cached_k.reshape(1, -1) if cached_k.size == self.d_model else cached_k.reshape(0, self.d_model)
            if cached_v.ndim == 1:
                cached_v = cached_v.reshape(1, -1) if cached_v.size == self.d_model else cached_v.reshape(0, self.d_model)

            if cached_k.ndim != 2 or cached_v.ndim != 2:
                raise ValueError(
                    f"cached_k and cached_v must be 2D arrays, got k={cached_k.shape}, v={cached_v.shape}"
                )
            if cached_k.shape[0] != cached_v.shape[0]:
                raise ValueError(
                    f"cached_k and cached_v sequence length mismatch: {cached_k.shape[0]} vs {cached_v.shape[0]}"
                )
            if cached_k.size > 0 and cached_k.shape[1] != self.d_model:
                raise ValueError(
                    f"cached_k feature dimension {cached_k.shape[1]} does not match d_model {self.d_model}"
                )
            if cached_v.size > 0 and cached_v.shape[1] != self.d_model:
                raise ValueError(
                    f"cached_v feature dimension {cached_v.shape[1]} does not match d_model {self.d_model}"
                )

            if cached_k.size > 0:
                keys = np.concatenate([cached_k, keys_new], axis=0)
                values = np.concatenate([cached_v, values_new], axis=0)
            else:
                keys = keys_new
                values = values_new
            new_kv_cache = (keys, values)
        else:
            keys = keys_new
            values = values_new
            new_kv_cache = (keys, values)

        total_seq_len = keys.shape[0]
        scale = 1.0 / math.sqrt(self.d_k)

        # 3. Reshape and Transpose into Multi-Head representations: [n_heads, seq_len, d_k]
        Q_heads = queries.reshape(seq_len, self.n_heads, self.d_k).swapaxes(0, 1)
        K_heads = keys.reshape(total_seq_len, self.n_heads, self.d_k).swapaxes(0, 1)
        V_heads = values.reshape(total_seq_len, self.n_heads, self.d_k).swapaxes(0, 1)

        # 4. Scaled Dot-Product Attention Scores: [n_heads, seq_len, total_seq_len]
        scores = np.matmul(Q_heads, K_heads.swapaxes(-1, -2)) * scale

        # 5. Causal Masking (query at offset + i attends only to key positions j <= offset + i)
        offset = total_seq_len - seq_len
        mask = np.tril(np.ones((seq_len, total_seq_len), dtype=bool), k=offset)
        masked_scores = np.where(mask, scores, -1e9)

        # 6. Numerically stable Softmax
        max_scores = np.max(masked_scores, axis=-1, keepdims=True)
        max_scores = np.where(np.isfinite(max_scores), max_scores, 0.0)
        exps = np.exp(masked_scores - max_scores)
        exps = np.where(mask, exps, 0.0)
        sum_exps = np.sum(exps, axis=-1, keepdims=True)
        sum_exps = np.where(sum_exps == 0.0, 1e-9, sum_exps)
        attn_weights = exps / sum_exps  # [n_heads, seq_len, total_seq_len]

        # 7. Weighted Value Aggregation: [n_heads, seq_len, d_k]
        context_heads = np.matmul(attn_weights, V_heads)

        # 8. Concatenate Heads: [seq_len, d_model] and Project
        context_vecs = context_heads.swapaxes(0, 1).reshape(seq_len, self.d_model)
        attn_out = context_vecs @ self.out_proj

        self._cache = {
            "x": x_arr,
            "queries": queries,
            "keys": keys,
            "values": values,
            "attn_weights": attn_weights,
            "context_vecs": context_vecs,
            "scale": scale,
        }

        if use_kv_cache or kv_cache is not None:
            return attn_out, new_kv_cache
        return attn_out

    def backward(self, grad_output: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Vectorized analytical causal multi-head self-attention backward pass.
        grad_output: [seq_len, d_model]
        Returns grad_input: [seq_len, d_model]
        """
        if self._cache is None:
            g_out = np.asarray(grad_output, dtype=np.float64)
            return np.zeros_like(g_out)

        x = self._cache["x"]
        queries = self._cache["queries"]
        keys = self._cache["keys"]
        values = self._cache["values"]
        attn_weights = self._cache["attn_weights"]  # [n_heads, seq_len, total_seq_len]
        context_vecs = self._cache["context_vecs"]  # [seq_len, d_model]
        scale = self._cache["scale"]

        g_out = np.asarray(grad_output, dtype=np.float64)
        if g_out.ndim == 1:
            g_out = g_out.reshape(1, -1)
        seq_len = x.shape[0]
        if seq_len == 0:
            return np.zeros_like(g_out)

        # 1. Gradients w.r.t out_proj and context_vecs
        self.grad_out_proj += context_vecs.T @ g_out
        grad_context_vecs = g_out @ self.out_proj.T  # [seq_len, d_model]

        # 2. Reshape grad_context_vecs to heads: [n_heads, seq_len, d_k]
        grad_context_heads = grad_context_vecs.reshape(seq_len, self.n_heads, self.d_k).swapaxes(0, 1)

        total_seq_len = keys.shape[0]
        Q_heads = queries.reshape(seq_len, self.n_heads, self.d_k).swapaxes(0, 1)
        K_heads = keys.reshape(total_seq_len, self.n_heads, self.d_k).swapaxes(0, 1)
        V_heads = values.reshape(total_seq_len, self.n_heads, self.d_k).swapaxes(0, 1)

        # 3. Backprop through context = attn_weights @ V_heads
        grad_V_heads = np.matmul(attn_weights.swapaxes(-1, -2), grad_context_heads)
        grad_attn_weights = np.matmul(grad_context_heads, V_heads.swapaxes(-1, -2))

        # 4. Backprop through Softmax
        sum_w_gradw = np.sum(attn_weights * grad_attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn_weights - sum_w_gradw)

        # 5. Backprop through scaled dot products
        grad_scores_scaled = grad_scores * scale

        grad_Q_heads = np.matmul(grad_scores_scaled, K_heads)  # [n_heads, seq_len, d_k]
        grad_K_heads = np.matmul(grad_scores_scaled.swapaxes(-1, -2), Q_heads)  # [n_heads, total_seq_len, d_k]

        # 6. Recombine head gradients back to [seq_len, d_model] / [total_seq_len, d_model]
        grad_queries = grad_Q_heads.swapaxes(0, 1).reshape(seq_len, self.d_model)
        grad_keys = grad_K_heads.swapaxes(0, 1).reshape(total_seq_len, self.d_model)
        grad_values = grad_V_heads.swapaxes(0, 1).reshape(total_seq_len, self.d_model)

        # 7. Gradients w.r.t Q, K, V projection matrices and input x
        self.grad_q_proj += x.T @ grad_queries
        self.grad_k_proj += x.T @ grad_keys[:seq_len]
        self.grad_v_proj += x.T @ grad_values[:seq_len]

        grad_x = (
            grad_queries @ self.q_proj.T
            + grad_keys[:seq_len] @ self.k_proj.T
            + grad_values[:seq_len] @ self.v_proj.T
        )
        return grad_x

    def zero_grad(self) -> None:
        self.grad_q_proj.fill(0.0)
        self.grad_k_proj.fill(0.0)
        self.grad_v_proj.fill(0.0)
        self.grad_out_proj.fill(0.0)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "q_proj": self.q_proj.tolist(),
            "k_proj": self.k_proj.tolist(),
            "v_proj": self.v_proj.tolist(),
            "out_proj": self.out_proj.tolist(),
        }

    def set_parameters(self, params: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"MultiHeadAttention params must be a dict, got {type(params).__name__}")
        required_keys = {"q_proj", "k_proj", "v_proj", "out_proj"}
        missing_keys = required_keys - set(params.keys())
        if missing_keys:
            raise KeyError(f"MultiHeadAttention parameters missing required keys: {sorted(missing_keys)}")
        if strict:
            extra_keys = set(params.keys()) - required_keys
            if extra_keys:
                raise KeyError(f"MultiHeadAttention received unexpected parameters: {sorted(extra_keys)}")

        expected_shape = (self.d_model, self.d_model)
        for key in required_keys:
            arr = np.array(params[key], dtype=np.float64)
            if arr.shape != expected_shape:
                raise ValueError(f"MultiHeadAttention.{key} shape mismatch: expected {expected_shape}, got {arr.shape}")

        self.q_proj = np.array(params["q_proj"], dtype=np.float64)
        self.k_proj = np.array(params["k_proj"], dtype=np.float64)
        self.v_proj = np.array(params["v_proj"], dtype=np.float64)
        self.out_proj = np.array(params["out_proj"], dtype=np.float64)
        self.zero_grad()
