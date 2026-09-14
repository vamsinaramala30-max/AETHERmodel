"""
AETHER MODEL — Transformer Block
Combines Multi-Head Attention, Feed-Forward Network, Layer Normalization, and Residual Connections.
Supports forward pass, KV-caching, analytical backward pass, and parameter state management.

Pre-LayerNorm Architecture:
    norm_x1 = LayerNorm_1(x)
    attn_out, kv_cache = MultiHeadAttention(norm_x1, kv_cache)
    res1 = x + attn_out
    norm_x2 = LayerNorm_2(res1)
    ffn_out = FeedForward(norm_x2)
    res2 = res1 + ffn_out
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork
from .normalization import LayerNorm

class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-5):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.eps = eps

        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model, eps)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln2 = LayerNorm(d_model, eps)

        self._cache: Optional[Dict[str, Any]] = None

    def forward(
        self,
        x: Union[np.ndarray, List[List[float]]],
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_kv_cache: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Executes Pre-LayerNorm Transformer block forward pass.
        x: [seq_len, d_model] or [d_model]
        kv_cache: Optional tuple of (cached_k, cached_v)
        use_kv_cache: Whether to return updated KV cache
        Returns: res2 [seq_len, d_model] (and updated_kv_cache if requested)
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
                    f"TransformerBlock input 1D array has feature dimension {x_arr.shape[0]}, expected d_model={self.d_model}"
                )
        elif x_arr.ndim == 2:
            if x_arr.shape[0] > 0 and x_arr.shape[1] != self.d_model:
                raise ValueError(
                    f"TransformerBlock input feature dimension {x_arr.shape[1]} does not match d_model {self.d_model}"
                )
        elif x_arr.ndim > 2:
            raise ValueError(f"TransformerBlock expects 2D input (seq_len, d_model), got shape {x_arr.shape}")

        if x_arr.shape[0] == 0:
            empty_cache = (
                np.zeros((0, self.d_model), dtype=np.float64),
                np.zeros((0, self.d_model), dtype=np.float64)
            )
            if use_kv_cache or kv_cache is not None:
                return np.zeros((0, self.d_model), dtype=np.float64), empty_cache
            return np.zeros((0, self.d_model), dtype=np.float64)

        # Residual 1: x + Self-Attention(LayerNorm_1(x))
        norm_x1 = self.ln1.forward(x_arr)
        if use_kv_cache or kv_cache is not None:
            attn_out, new_kv_cache = self.attn.forward(norm_x1, kv_cache=kv_cache, use_kv_cache=True)
        else:
            attn_out = self.attn.forward(norm_x1)
            new_kv_cache = None

        res1 = x_arr + attn_out

        # Residual 2: res1 + FeedForward(LayerNorm_2(res1))
        norm_x2 = self.ln2.forward(res1)
        ffn_out = self.ffn.forward(norm_x2)
        res2 = res1 + ffn_out

        self._cache = {
            "x": x_arr,
            "norm_x1": norm_x1,
            "attn_out": attn_out,
            "res1": res1,
            "norm_x2": norm_x2,
            "ffn_out": ffn_out,
        }

        if use_kv_cache or kv_cache is not None:
            return res2, new_kv_cache
        return res2

    def backward(self, grad_output: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Analytical TransformerBlock backward pass through residuals, FFN, and Self-Attention.
        grad_output: [seq_len, d_model]
        Returns grad_input: [seq_len, d_model]
        """
        if self._cache is None:
            g_out = np.asarray(grad_output, dtype=np.float64)
            return np.zeros_like(g_out)

        g_out = np.asarray(grad_output, dtype=np.float64)
        if g_out.ndim == 1:
            g_out = g_out.reshape(1, -1)

        seq_len = self._cache["x"].shape[0]
        if seq_len == 0:
            return np.zeros_like(g_out)

        # 1. Backward through Residual 2: res2 = res1 + ffn_out
        grad_ffn_out = g_out
        grad_res1_direct = g_out

        # 2. Backward through FFN & LN2
        grad_norm_x2 = self.ffn.backward(grad_ffn_out)
        grad_res1_from_ffn = self.ln2.backward(grad_norm_x2)

        # 3. Combine gradients at res1
        grad_res1 = grad_res1_direct + grad_res1_from_ffn

        # 4. Backward through Residual 1: res1 = x + attn_out
        grad_attn_out = grad_res1
        grad_x_direct = grad_res1

        # 5. Backward through Self-Attention & LN1
        grad_norm_x1 = self.attn.backward(grad_attn_out)
        grad_x_from_attn = self.ln1.backward(grad_norm_x1)

        # 6. Combine gradients at block input x
        grad_x = grad_x_direct + grad_x_from_attn

        return grad_x

    def zero_grad(self) -> None:
        self.attn.zero_grad()
        self.ln1.zero_grad()
        self.ffn.zero_grad()
        self.ln2.zero_grad()

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "attn": self.attn.get_parameters(),
            "ln1": self.ln1.get_parameters(),
            "ffn": self.ffn.get_parameters(),
            "ln2": self.ln2.get_parameters(),
        }

    def set_parameters(self, params: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"TransformerBlock params must be a dict, got {type(params).__name__}")
        required_keys = {"attn", "ln1", "ffn", "ln2"}
        missing_keys = required_keys - set(params.keys())
        if missing_keys:
            raise KeyError(f"TransformerBlock parameters missing required keys: {sorted(missing_keys)}")
        if strict:
            extra_keys = set(params.keys()) - required_keys
            if extra_keys:
                raise KeyError(f"TransformerBlock received unexpected parameters: {sorted(extra_keys)}")

        self.attn.set_parameters(params["attn"], strict=strict)
        self.ln1.set_parameters(params["ln1"], strict=strict)
        self.ffn.set_parameters(params["ffn"], strict=strict)
        self.ln2.set_parameters(params["ln2"], strict=strict)
        self.zero_grad()
