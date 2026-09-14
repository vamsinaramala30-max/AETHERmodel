"""
AETHER MODEL — Transformer Architecture
Full Decoder-only Transformer model assembling embeddings, positional encoding, stacked transformer blocks, layer norm, and LM output head.
Supports single-step and full-sequence forward pass, analytical backpropagation, and state dict serialization.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from model.config.model_config import ModelConfig
from model.architecture.embeddings import TokenEmbedding
from model.architecture.positional_encoding import SinusoidalPositionalEncoding
from model.architecture.transformer_block import TransformerBlock
from model.architecture.normalization import LayerNorm
from model.architecture.output_head import OutputHead

class AetherTransformerArchitecture:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len)
        self.blocks = [
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.epsilon)
            for _ in range(config.n_layers)
        ]
        self.final_ln = LayerNorm(config.d_model, config.epsilon)
        self.lm_head = OutputHead(config.d_model, config.vocab_size)

        self._last_input_ids: Optional[List[int]] = None

    def forward(self, input_ids: List[int]) -> List[float]:
        """
        Executes forward pass over sequence of input_ids.
        Returns logits array of length vocab_size for predicting the next token.
        """
        if not input_ids:
            return [0.0] * self.config.vocab_size

        # 1. Embedding lookup
        x = self.token_embedding.forward(input_ids)

        # 2. Add Positional Encoding
        x = self.pos_encoding.forward(x)

        # 3. Stacked Transformer Blocks
        for block in self.blocks:
            x = block.forward(x)

        x = self.final_ln.forward(x)
        logits = self.lm_head.forward(x[-1])

        return logits.tolist() if hasattr(logits, "tolist") else list(logits)

    def forward_step(
        self,
        token_id: int,
        start_pos: int,
        layer_caches: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Tuple[Union[np.ndarray, List[float]], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Single-step forward pass for position start_pos using internal KV-caching.
        """
        x = self.token_embedding.forward([token_id])
        x = self.pos_encoding.forward(x, start_pos=start_pos)

        new_layer_caches = []
        for idx, block in enumerate(self.blocks):
            cache = layer_caches[idx] if layer_caches is not None else None
            x, new_cache = block.forward(x, kv_cache=cache, use_kv_cache=True)
            new_layer_caches.append(new_cache)

        x = self.final_ln.forward(x)
        logits = self.lm_head.forward(x[-1])
        return logits, new_layer_caches

    def forward_prompt(
        self,
        prompt_token_ids: List[int]
    ) -> Tuple[Union[np.ndarray, List[float]], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Parallel prompt forward pass over prompt_token_ids.
        Evaluates full prompt sequence in 1 step and initializes KV caches for all blocks.
        """
        if not prompt_token_ids:
            empty_caches = [(np.zeros((0, self.config.d_model), dtype=np.float64), np.zeros((0, self.config.d_model), dtype=np.float64)) for _ in self.blocks]
            return np.zeros(self.config.vocab_size, dtype=np.float64), empty_caches

        x = self.token_embedding.forward(prompt_token_ids)
        x = self.pos_encoding.forward(x, start_pos=0)

        new_layer_caches = []
        for block in self.blocks:
            x, new_cache = block.forward(x, kv_cache=None, use_kv_cache=True)
            new_layer_caches.append(new_cache)

        x = self.final_ln.forward(x)
        logits = self.lm_head.forward(x[-1])
        return logits, new_layer_caches

    def forward_all(self, input_ids: List[int]) -> List[List[float]]:
        """
        Executes full forward pass over input_ids, computing logits for EVERY position in the sequence.
        Returns logits matrix of shape [seq_len, vocab_size].
        """
        if not input_ids:
            return []

        self._last_input_ids = list(input_ids)

        # 1. Embedding lookup
        x = self.token_embedding.forward(input_ids)

        # 2. Add Positional Encoding
        x = self.pos_encoding.forward(x)

        # 3. Stacked Transformer Blocks
        for block in self.blocks:
            x = block.forward(x)

        # 4. Final Layer Normalization
        x = self.final_ln.forward(x)

        # 5. Compute LM Head logits for all sequence positions
        all_logits = self.lm_head.forward_all(x)
        return all_logits.tolist() if hasattr(all_logits, "tolist") else [list(row) for row in all_logits]

    def backward(self, grad_logits: List[List[float]]) -> None:
        """
        Executes full analytical backpropagation pass starting from grad_logits [seq_len, vocab_size].
        Accumulates gradients into all trainable layer weights.
        """
        if not grad_logits or self._last_input_ids is None:
            return

        # 1. Backward through LM Head
        grad_x = self.lm_head.backward(grad_logits)

        # 2. Backward through Final Layer Normalization
        grad_x = self.final_ln.backward(grad_x)

        # 3. Backward through Stacked Transformer Blocks (in reverse order)
        for block in reversed(self.blocks):
            grad_x = block.backward(grad_x)

        # 4. Positional encoding has no learnable parameters (dx is passed directly)

        # 5. Backward through Token Embedding
        self.token_embedding.backward(grad_x)

    def zero_grad(self) -> None:
        self.token_embedding.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.final_ln.zero_grad()
        self.lm_head.zero_grad()

    def get_state_dict(self) -> Dict[str, Any]:
        state = {
            "token_embedding": self.token_embedding.get_parameters(),
            "blocks": [block.get_parameters() for block in self.blocks],
            "final_ln": self.final_ln.get_parameters(),
            "lm_head": self.lm_head.get_parameters(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")

        required_keys = {"token_embedding", "blocks", "final_ln", "lm_head"}
        missing_keys = required_keys - set(state_dict.keys())
        if missing_keys:
            raise KeyError(f"state_dict missing required top-level keys: {sorted(missing_keys)}")

        if strict:
            extra_keys = set(state_dict.keys()) - required_keys
            if extra_keys:
                raise KeyError(f"state_dict received unexpected top-level keys: {sorted(extra_keys)}")

        blocks_data = state_dict["blocks"]
        if not isinstance(blocks_data, list):
            raise TypeError(f"state_dict['blocks'] must be a list, got {type(blocks_data).__name__}")

        if len(blocks_data) != len(self.blocks):
            raise ValueError(
                f"state_dict blocks count mismatch: model has {len(self.blocks)} layers, "
                f"checkpoint has {len(blocks_data)} layers"
            )

        self.token_embedding.set_parameters(state_dict["token_embedding"], strict=strict)
        for idx, block_state in enumerate(blocks_data):
            self.blocks[idx].set_parameters(block_state, strict=strict)
        self.final_ln.set_parameters(state_dict["final_ln"], strict=strict)
        self.lm_head.set_parameters(state_dict["lm_head"], strict=strict)
        self.zero_grad()

    def get_named_parameters(self) -> List[Tuple[str, Any, Any]]:
        """
        Returns list of (param_name, param_tensor, grad_tensor) for optimizer parameter updates.
        """
        params = []
        # Embedding
        params.append(("token_embedding.weight", self.token_embedding.weight, self.token_embedding.grad_weight))
        # Blocks
        for b_idx, block in enumerate(self.blocks):
            # Attention
            params.append((f"blocks.{b_idx}.attn.q_proj", block.attn.q_proj, block.attn.grad_q_proj))
            params.append((f"blocks.{b_idx}.attn.k_proj", block.attn.k_proj, block.attn.grad_k_proj))
            params.append((f"blocks.{b_idx}.attn.v_proj", block.attn.v_proj, block.attn.grad_v_proj))
            params.append((f"blocks.{b_idx}.attn.out_proj", block.attn.out_proj, block.attn.grad_out_proj))
            # LN1
            params.append((f"blocks.{b_idx}.ln1.gamma", block.ln1.gamma, block.ln1.grad_gamma))
            params.append((f"blocks.{b_idx}.ln1.beta", block.ln1.beta, block.ln1.grad_beta))
            # FFN
            params.append((f"blocks.{b_idx}.ffn.w1", block.ffn.w1, block.ffn.grad_w1))
            params.append((f"blocks.{b_idx}.ffn.b1", block.ffn.b1, block.ffn.grad_b1))
            params.append((f"blocks.{b_idx}.ffn.w2", block.ffn.w2, block.ffn.grad_w2))
            params.append((f"blocks.{b_idx}.ffn.b2", block.ffn.b2, block.ffn.grad_b2))
            # LN2
            params.append((f"blocks.{b_idx}.ln2.gamma", block.ln2.gamma, block.ln2.grad_gamma))
            params.append((f"blocks.{b_idx}.ln2.beta", block.ln2.beta, block.ln2.grad_beta))
        # Final LN
        params.append(("final_ln.gamma", self.final_ln.gamma, self.final_ln.grad_gamma))
        params.append(("final_ln.beta", self.final_ln.beta, self.final_ln.grad_beta))
        # LM Head
        params.append(("lm_head.weight", self.lm_head.weight, self.lm_head.grad_weight))
        params.append(("lm_head.bias", self.lm_head.bias, self.lm_head.grad_bias))

        return params

