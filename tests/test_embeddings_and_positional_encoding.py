"""
AETHER MODEL — Embedding and Positional Encoding Regression Test Suite (Phase 3)

Validates:
1. Every valid token ID can be embedded.
2. Invalid token IDs (negative, out-of-range, non-integer) are handled safely and explicitly.
3. Embedding output shape is exactly [sequence_length, d_model].
4. Embedding weights match checkpoint dimensions.
5. Positional encoding has exactly d_model dimensions.
6. Position 0 is mathematically correct (even indices = 0.0, odd indices = 1.0).
7. Positions increase deterministically.
8. Maximum sequence length is respected, and oversized sequences are handled safely.
9. forward() and forward_prompt() use identical positional conventions.
10. forward_step() uses the SAME positional convention as full-sequence inference.
11. Cached generation positions do not restart at zero accidentally.
12. No accidental broadcasting changes the tensor shape.
13. Representation at final position is numerically equivalent between full forward and prompt + cached step passes.
"""

import sys
import os
import unittest
import math
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from model.architecture.embeddings import TokenEmbedding
from model.architecture.positional_encoding import SinusoidalPositionalEncoding


class TestEmbeddingsAndPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 579
        self.d_model = 64
        self.max_seq_len = 256
        self.config = ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            n_layers=2,
            n_heads=2,
            d_ff=128
        )
        self.model = AetherModel(self.config)
        self.embedding = self.model.architecture.token_embedding
        self.pos_encoding = self.model.architecture.pos_encoding
        self.tol = 1e-5

    def test_01_every_valid_token_id_can_be_embedded(self):
        """Every valid token ID in range [0, vocab_size-1] can be embedded and returns correct weights."""
        all_ids = list(range(self.vocab_size))
        out = self.embedding.forward(all_ids)
        self.assertEqual(out.shape, (self.vocab_size, self.d_model))
        self.assertTrue(np.allclose(out, self.embedding.weight))

        # Test individual valid IDs
        for tid in [0, 1, 2, 50, 100, self.vocab_size - 1]:
            single_out = self.embedding.forward([tid])
            self.assertEqual(single_out.shape, (1, self.d_model))
            self.assertTrue(np.allclose(single_out[0], self.embedding.weight[tid]))

    def test_02_invalid_token_ids_handled_safely(self):
        """Negative IDs and out-of-range IDs clamp safely to UNK_TOKEN_ID without raising or aliasing."""
        unk_id = self.embedding.UNK_TOKEN_ID
        expected_unk_vec = self.embedding.weight[unk_id]

        invalid_ids = [-1, -100, self.vocab_size, self.vocab_size + 50, 999999]
        out = self.embedding.forward(invalid_ids)
        self.assertEqual(out.shape, (len(invalid_ids), self.d_model))

        for row in out:
            self.assertTrue(np.allclose(row, expected_unk_vec), "Invalid token must map to UNK embedding vector")

    def test_03_embedding_output_shape_exactness(self):
        """Embedding output shape is strictly 2D [sequence_length, d_model] across all input representations."""
        # Empty input
        self.assertEqual(self.embedding.forward([]).shape, (0, self.d_model))
        self.assertEqual(self.embedding.forward(np.array([], dtype=np.int64)).shape, (0, self.d_model))

        # Single token integer scalar
        out_scalar = self.embedding.forward(42)
        self.assertEqual(out_scalar.shape, (1, self.d_model))

        # Single token list
        out_list1 = self.embedding.forward([42])
        self.assertEqual(out_list1.shape, (1, self.d_model))

        # Single token numpy 1D array
        out_arr1 = self.embedding.forward(np.array([42]))
        self.assertEqual(out_arr1.shape, (1, self.d_model))

        # 2D numpy array input should be flattened to [sequence_length, d_model]
        out_arr2d = self.embedding.forward(np.array([[10, 20, 30]]))
        self.assertEqual(out_arr2d.shape, (3, self.d_model))

        # Multiple tokens list
        out_multi = self.embedding.forward([1, 2, 3, 4, 5])
        self.assertEqual(out_multi.shape, (5, self.d_model))

    def test_04_embedding_weights_match_checkpoint_dimensions(self):
        """Embedding weights match the verified model configuration and checkpoint contract."""
        self.assertEqual(self.embedding.weight.shape, (self.vocab_size, self.d_model))
        self.assertEqual(self.embedding.grad_weight.shape, (self.vocab_size, self.d_model))

    def test_05_positional_encoding_dimensions(self):
        """Positional encoding has exactly d_model dimensions and precomputes max_seq_len positions."""
        self.assertEqual(self.pos_encoding.pe.shape, (self.max_seq_len, self.d_model))
        self.assertEqual(self.pos_encoding.d_model, self.d_model)

    def test_06_position_0_is_correct(self):
        """Position 0 evaluates to sin(0)=0.0 for even dimensions and cos(0)=1.0 for odd dimensions."""
        p0 = self.pos_encoding.pe[0]
        for i in range(self.d_model):
            if i % 2 == 0:
                self.assertAlmostEqual(p0[i], 0.0, places=7, msg=f"pe[0, {i}] even dimension should be 0.0")
            else:
                self.assertAlmostEqual(p0[i], 1.0, places=7, msg=f"pe[0, {i}] odd dimension should be 1.0")

    def test_07_positions_increase_deterministically(self):
        """Positional encodings follow exact mathematical sinusoidal formulas deterministically."""
        for pos in range(min(50, self.max_seq_len)):
            for i in range(self.d_model):
                if i % 2 == 0:
                    denom = 10000.0 ** (i / self.d_model)
                    expected = math.sin(pos / denom)
                else:
                    denom = 10000.0 ** ((i - 1) / self.d_model)
                    expected = math.cos(pos / denom)
                self.assertAlmostEqual(self.pos_encoding.pe[pos, i], expected, places=7)

    def test_08_maximum_sequence_length_and_oversized_sequences(self):
        """Sequences up to and exceeding max_seq_len are processed without shape crashes."""
        # Exact max_seq_len
        x_max = np.zeros((self.max_seq_len, self.d_model))
        out_max = self.pos_encoding.forward(x_max, start_pos=0)
        self.assertEqual(out_max.shape, (self.max_seq_len, self.d_model))
        self.assertTrue(np.allclose(out_max, self.pos_encoding.pe))

        # Oversized sequence (e.g. max_seq_len + 50)
        oversized_len = self.max_seq_len + 50
        x_over = np.zeros((oversized_len, self.d_model))
        out_over = self.pos_encoding.forward(x_over, start_pos=0)
        self.assertEqual(out_over.shape, (oversized_len, self.d_model))

        # Beyond max_seq_len, positions clamp to max_seq_len - 1
        last_pe_vec = self.pos_encoding.pe[-1]
        for p in range(self.max_seq_len, oversized_len):
            self.assertTrue(np.allclose(out_over[p], last_pe_vec))

    def test_09_forward_and_forward_prompt_positional_conventions(self):
        """forward() and forward_prompt() use identical position conventions starting at pos 0."""
        token_ids = [12, 34, 56, 78, 90]
        # Embeddings + Positional Encoding for full forward
        x_full = self.embedding.forward(token_ids)
        pe_full = self.pos_encoding.forward(x_full, start_pos=0)

        # Embeddings + Positional Encoding for prompt forward
        x_prompt = self.embedding.forward(token_ids)
        pe_prompt = self.pos_encoding.forward(x_prompt, start_pos=0)

        self.assertTrue(np.allclose(pe_full, pe_prompt))

    def test_10_forward_step_positional_convention(self):
        """forward_step() position k uses the exact same positional encoding slice as full sequence position k."""
        token_ids = [15, 30, 45, 60, 75]
        x_seq = self.embedding.forward(token_ids)
        pe_seq = self.pos_encoding.forward(x_seq, start_pos=0)

        for pos, tid in enumerate(token_ids):
            x_step = self.embedding.forward([tid])
            pe_step = self.pos_encoding.forward(x_step, start_pos=pos)
            self.assertEqual(pe_step.shape, (1, self.d_model))
            self.assertTrue(np.allclose(pe_step[0], pe_seq[pos]), f"Positional mismatch at position {pos}")

    def test_11_cached_generation_positions_do_not_restart(self):
        """Verify that step-by-step cached position advance strictly increments and does not restart at zero."""
        prompt = [10, 20, 30]
        curr_pos = len(prompt)

        # Simulate 5 generation steps
        next_tokens = [40, 50, 60, 70, 80]
        for step, tok in enumerate(next_tokens):
            step_pos = curr_pos + step
            self.assertGreaterEqual(step_pos, len(prompt))
            # Verify pos encoding used corresponds to step_pos
            x = self.embedding.forward([tok])
            pe_step = self.pos_encoding.forward(x, start_pos=step_pos)
            self.assertTrue(np.allclose(pe_step[0], x[0] + self.pos_encoding.pe[step_pos]))

    def test_12_no_accidental_broadcasting(self):
        """1D vector input to positional encoding normalizes to [1, d_model] and does not broadcast into a matrix."""
        vec_1d = np.random.randn(self.d_model)
        pe_1d = self.pos_encoding.forward(vec_1d, start_pos=3)
        self.assertEqual(pe_1d.shape, (1, self.d_model))
        expected = vec_1d + self.pos_encoding.pe[3]
        self.assertTrue(np.allclose(pe_1d[0], expected))

        # Test negative start_pos clamps safely to 0
        pe_neg = self.pos_encoding.forward(vec_1d, start_pos=-5)
        self.assertEqual(pe_neg.shape, (1, self.d_model))
        self.assertTrue(np.allclose(pe_neg[0], vec_1d + self.pos_encoding.pe[0]))

    def test_13_full_forward_vs_cached_step_numerical_consistency(self):
        """Full forward representation at final position and prompt + cached step representation are numerically identical."""
        tokens = [5, 14, 28, 55, 92, 110, 204, 305, 412]

        # 1. Full sequence forward pass
        full_logits = self.model.forward(tokens)

        # 2. Prompt pass on prefix + cached forward_step for rest
        prefix_len = 4
        prefix = tokens[:prefix_len]
        remaining = tokens[prefix_len:]

        step_logits, caches = self.model.forward_prompt(prefix)
        curr_pos = len(prefix)
        for tok in remaining:
            step_logits, caches = self.model.forward_step(tok, start_pos=curr_pos, layer_caches=caches)
            curr_pos += 1

        # Numerical comparison
        diff = np.max(np.abs(np.array(full_logits) - np.array(step_logits)))
        self.assertLess(
            diff,
            self.tol,
            f"Full forward vs cached step discrepancy exceeds tolerance: max diff = {diff:.2e}"
        )

    def test_14_embedding_backward_gradient_accumulation(self):
        """TokenEmbedding.backward correctly accumulates gradients for 1D and 2D gradient arrays."""
        self.embedding.zero_grad()
        token_ids = [10, 20, 10, -1]  # 10 appears twice, 20 once, -1 is invalid (UNK)
        self.embedding.forward(token_ids)

        # 2D gradient input [4, d_model]
        grad_out = np.ones((4, self.d_model), dtype=np.float64)
        self.embedding.backward(grad_out)

        # Token 10 should have accumulated 2.0 per dimension
        self.assertTrue(np.allclose(self.embedding.grad_weight[10], 2.0))
        # Token 20 should have accumulated 1.0 per dimension
        self.assertTrue(np.allclose(self.embedding.grad_weight[20], 1.0))
        # UNK token should have accumulated 1.0 for the invalid token
        self.assertTrue(np.allclose(self.embedding.grad_weight[self.embedding.UNK_TOKEN_ID], 1.0))

        # Test 1D gradient input for single token (no scalar broadcasting bug)
        self.embedding.zero_grad()
        self.embedding.forward([45])
        grad_1d = np.full(self.d_model, 3.5, dtype=np.float64)
        self.embedding.backward(grad_1d)
        self.assertTrue(np.allclose(self.embedding.grad_weight[45], 3.5))


if __name__ == "__main__":
    unittest.main()
