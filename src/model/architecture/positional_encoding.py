"""
AETHER MODEL — Positional Encoding
Adds sinusoidal positional information to token embeddings for sequence-order awareness.
"""

import math
import numpy as np
from typing import Union, List

class SinusoidalPositionalEncoding:
    def __init__(self, d_model: int, max_seq_len: int = 1024):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        pos_arr = np.arange(max_seq_len, dtype=np.float64)[:, None]
        i_arr = np.arange(d_model, dtype=np.float64)[None, :]
        even_mask = (i_arr % 2 == 0)
        denom_pow = np.where(even_mask, i_arr, i_arr - 1) / d_model
        angles = pos_arr / (10000.0 ** denom_pow)
        self.pe: np.ndarray = np.where(even_mask, np.sin(angles), np.cos(angles))


    def forward(self, x: Union[np.ndarray, List[List[float]], List[float]], start_pos: int = 0) -> np.ndarray:
        """
        Add positional encodings element-wise to input vector sequence starting at start_pos.
        Guarantees returned shape is strictly [sequence_length, d_model].
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size == 0:
            return np.zeros((0, self.d_model), dtype=np.float64)

        if x_arr.ndim == 1:
            if x_arr.shape[0] == self.d_model:
                x_arr = x_arr.reshape(1, self.d_model)
            else:
                raise ValueError(
                    f"PositionalEncoding 1D input dimension mismatch: expected d_model {self.d_model}, got {x_arr.shape[0]}"
                )
        elif x_arr.ndim == 2:
            if x_arr.shape[1] != self.d_model:
                raise ValueError(
                    f"PositionalEncoding feature dimension mismatch: expected d_model {self.d_model}, got {x_arr.shape[1]}"
                )
        else:
            raise ValueError(
                f"PositionalEncoding expects 2D input [seq_len, d_model], got shape {x_arr.shape}"
            )

        seq_len = x_arr.shape[0]
        pos_offset = max(0, int(start_pos))
        end_pos = pos_offset + seq_len

        if end_pos <= self.max_seq_len:
            pe_slice = self.pe[pos_offset:end_pos]
        else:
            indices = [min(pos_offset + i, self.max_seq_len - 1) for i in range(seq_len)]
            pe_slice = self.pe[indices]

        return x_arr + pe_slice

