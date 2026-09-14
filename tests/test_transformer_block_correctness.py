"""
AETHER MODEL — TransformerBlock Correctness & Integration Test Suite
Phase 5 Comprehensive Audit & Regression Testing.

Verifies:
1. Architecture Flow: Pre-LayerNorm (LN1 -> Attention -> Residual -> LN2 -> FFN -> Residual)
2. Numerical Equivalence: Full-sequence block forward vs step-by-step cached block forward
3. Dimension & Sequence Length Invariance
4. Input Non-Mutation & Safe Memory Handling
5. Gradient Flow & Analytical Backpropagation through both Residual Branches
6. Parameter Serialization & Strict State Management
7. Numerical Stability (No NaN/Inf on Extreme Values)
8. Edge Cases: Sequence lengths 1, 2, max_len, empty arrays, 1D vectors
"""

import sys
import os
import unittest
import math
import numpy as np

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.config.model_config import ModelConfig
from model.architecture.transformer_block import TransformerBlock
from model.architecture.normalization import LayerNorm
from model.architecture.feed_forward import FeedForwardNetwork
from model.architecture.attention import MultiHeadAttention

class TestTransformerBlockCorrectness(unittest.TestCase):
    def setUp(self):
        self.d_model = 32
        self.n_heads = 4
        self.d_ff = 64
        self.eps = 1e-5
        self.block = TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            eps=self.eps
        )
        self.tol = 1e-5

    # =========================================================================
    # 1. ARCHITECTURE FLOW & DIMENSION INVARIANCE
    # =========================================================================
    def test_pre_ln_architecture_flow_and_dimension_invariance(self):
        """Verifies Pre-LN flow and that (seq_len, d_model) shape is strictly preserved."""
        seq_len = 8
        x = np.random.randn(seq_len, self.d_model)
        out = self.block.forward(x)

        self.assertEqual(out.shape, (seq_len, self.d_model))
        self.assertTrue(np.all(np.isfinite(out)))

        # Verify manual Pre-LN step-by-step match
        norm_x1 = self.block.ln1.forward(x)
        attn_out = self.block.attn.forward(norm_x1)
        res1 = x + attn_out
        norm_x2 = self.block.ln2.forward(res1)
        ffn_out = self.block.ffn.forward(norm_x2)
        expected_res2 = res1 + ffn_out

        max_err = np.max(np.abs(out - expected_res2))
        self.assertLess(max_err, self.tol, f"Pre-LN execution mismatch: {max_err}")

    def test_residual_connections_integrity(self):
        """Verifies that residual connections preserve input signal when sub-layers are zeroed."""
        zero_block = TransformerBlock(self.d_model, self.n_heads, self.d_ff)
        zero_block.attn.out_proj.fill(0.0)
        zero_block.ffn.w2.fill(0.0)
        zero_block.ffn.b2.fill(0.0)

        x = np.random.randn(5, self.d_model)
        out = zero_block.forward(x)

        # When attention out_proj and FFN w2/b2 are 0, block forward must equal x
        max_err = np.max(np.abs(out - x))
        self.assertLess(max_err, self.tol, f"Residual bypass failure: {max_err}")

    # =========================================================================
    # 2. FULL-SEQUENCE VS CACHED BLOCK FORWARD EQUIVALENCE
    # =========================================================================
    def test_full_sequence_vs_cached_block_forward_equivalence(self):
        """
        Verifies that single-step cached block forward matches full-sequence block forward exactly.
        """
        seq_len = 10
        x_seq = np.random.randn(seq_len, self.d_model)

        # 1. Full-sequence block forward
        full_out = self.block.forward(x_seq)

        # 2. Step-by-step cached block forward
        kv_cache = None
        cached_steps = []
        for pos in range(seq_len):
            x_token = x_seq[pos:pos+1]
            step_out, kv_cache = self.block.forward(x_token, kv_cache=kv_cache, use_kv_cache=True)
            cached_steps.append(step_out)

        cached_out = np.vstack(cached_steps)

        # Check full representation equivalence across all sequence positions
        max_err = np.max(np.abs(full_out - cached_out))
        self.assertLess(
            max_err,
            self.tol,
            f"TransformerBlock full vs cached forward mismatch: {max_err:.2e}"
        )

    def test_prompt_chunk_plus_cached_steps_equivalence(self):
        """
        Verifies prompt chunk prefill (e.g. 4 tokens) + cached step-by-step forward.
        """
        seq_len = 8
        x_seq = np.random.randn(seq_len, self.d_model)

        full_out = self.block.forward(x_seq)

        # Prefill prompt of 4 tokens
        prefix_len = 4
        prompt_x = x_seq[:prefix_len]
        prompt_out, kv_cache = self.block.forward(prompt_x, use_kv_cache=True)

        cached_outputs = [prompt_out]
        for pos in range(prefix_len, seq_len):
            x_step = x_seq[pos:pos+1]
            step_out, kv_cache = self.block.forward(x_step, kv_cache=kv_cache, use_kv_cache=True)
            cached_outputs.append(step_out)

        combined_cached_out = np.vstack(cached_outputs)
        max_err = np.max(np.abs(full_out - combined_cached_out))
        self.assertLess(
            max_err,
            self.tol,
            f"Chunk prefill + cached steps discrepancy: {max_err:.2e}"
        )

    # =========================================================================
    # 3. INPUT NON-MUTATION & MEMORY SAFETY
    # =========================================================================
    def test_input_non_mutation(self):
        """Verifies that input tensors are never mutated in-place by forward or backward."""
        x_orig = np.random.randn(6, self.d_model)
        x_copy = x_orig.copy()

        # Forward pass
        out = self.block.forward(x_orig)
        self.assertTrue(np.array_equal(x_orig, x_copy), "Input x was mutated during forward pass!")

        # Backward pass
        grad_out = np.random.randn(6, self.d_model)
        grad_copy = grad_out.copy()
        grad_in = self.block.backward(grad_out)
        self.assertTrue(np.array_equal(grad_out, grad_copy), "grad_output was mutated during backward pass!")
        self.assertTrue(np.array_equal(x_orig, x_copy), "Input x was mutated during backward pass!")

    # =========================================================================
    # 4. ANALYTICAL BACKWARD PASS & FINITE-DIFFERENCE GRADIENTS
    # =========================================================================
    def test_block_gradient_verification_finite_differences(self):
        """Verifies analytical gradients of TransformerBlock against two-sided numerical finite differences."""
        eps = 1e-4
        seq_len = 4
        x = np.random.randn(seq_len, self.d_model)
        target = np.random.randn(seq_len, self.d_model)

        def loss_fn():
            out = self.block.forward(x)
            return 0.5 * np.sum((out - target) ** 2)

        # Forward & analytical backward
        self.block.zero_grad()
        out = self.block.forward(x)
        grad_out = out - target
        grad_x = self.block.backward(grad_out)

        # 1. Verify grad_x against finite differences
        for r in range(seq_len):
            for c in range(min(2, self.d_model)):
                orig_val = x[r, c]

                x[r, c] = orig_val + eps
                l_plus = loss_fn()

                x[r, c] = orig_val - eps
                l_minus = loss_fn()

                x[r, c] = orig_val
                g_num = (l_plus - l_minus) / (2.0 * eps)
                g_ana = grad_x[r, c]

                abs_err = abs(g_ana - g_num)
                rel_err = abs_err / max(abs(g_ana), abs(g_num), 1e-4)
                self.assertLess(
                    rel_err,
                    0.05,
                    f"TransformerBlock grad_x[{r},{c}] mismatch: ana={g_ana}, num={g_num}, rel_err={rel_err}"
                )

        # 2. Verify sub-layer parameter gradients
        sub_params = [
            ("attn.q_proj", self.block.attn.q_proj, self.block.attn.grad_q_proj),
            ("attn.out_proj", self.block.attn.out_proj, self.block.attn.grad_out_proj),
            ("ln1.gamma", self.block.ln1.gamma, self.block.ln1.grad_gamma),
            ("ffn.w1", self.block.ffn.w1, self.block.ffn.grad_w1),
            ("ffn.b1", self.block.ffn.b1, self.block.ffn.grad_b1),
            ("ffn.w2", self.block.ffn.w2, self.block.ffn.grad_w2),
            ("ln2.gamma", self.block.ln2.gamma, self.block.ln2.grad_gamma),
        ]

        for p_name, param_tensor, grad_tensor in sub_params:
            if param_tensor.ndim == 2:
                for r, c in [(0, 0), (1, 1)]:
                    orig_p = param_tensor[r, c]
                    param_tensor[r, c] = orig_p + eps
                    l_plus = loss_fn()
                    param_tensor[r, c] = orig_p - eps
                    l_minus = loss_fn()
                    param_tensor[r, c] = orig_p

                    g_num = (l_plus - l_minus) / (2.0 * eps)
                    g_ana = grad_tensor[r, c]
                    abs_err = abs(g_ana - g_num)
                    rel_err = abs_err / max(abs(g_ana), abs(g_num), 1e-4)
                    self.assertLess(
                        rel_err,
                        0.05,
                        f"Parameter {p_name}[{r},{c}] gradient error: ana={g_ana}, num={g_num}, rel_err={rel_err}"
                    )
            else:
                for i in [0, 1]:
                    orig_p = param_tensor[i]
                    param_tensor[i] = orig_p + eps
                    l_plus = loss_fn()
                    param_tensor[i] = orig_p - eps
                    l_minus = loss_fn()
                    param_tensor[i] = orig_p

                    g_num = (l_plus - l_minus) / (2.0 * eps)
                    g_ana = grad_tensor[i]
                    abs_err = abs(g_ana - g_num)
                    rel_err = abs_err / max(abs(g_ana), abs(g_num), 1e-4)
                    self.assertLess(
                        rel_err,
                        0.05,
                        f"Parameter {p_name}[{i}] gradient error: ana={g_ana}, num={g_num}, rel_err={rel_err}"
                    )

    def test_gradient_accumulation_and_zero_grad(self):
        """Verifies that gradients accumulate across multiple backward calls and reset on zero_grad."""
        x = np.random.randn(3, self.d_model)
        grad_out = np.ones((3, self.d_model), dtype=np.float64)

        self.block.zero_grad()
        self.block.forward(x)
        self.block.backward(grad_out)
        first_grad_q = self.block.attn.grad_q_proj.copy()

        # Second backward without zero_grad -> must double
        self.block.forward(x)
        self.block.backward(grad_out)
        second_grad_q = self.block.attn.grad_q_proj.copy()

        max_err = np.max(np.abs(second_grad_q - 2.0 * first_grad_q))
        self.assertLess(max_err, 1e-10, "Gradients failed to accumulate correctly!")

        # zero_grad
        self.block.zero_grad()
        self.assertTrue(np.all(self.block.attn.grad_q_proj == 0.0))
        self.assertTrue(np.all(self.block.ln1.grad_gamma == 0.0))
        self.assertTrue(np.all(self.block.ffn.grad_w1 == 0.0))
        self.assertTrue(np.all(self.block.ln2.grad_gamma == 0.0))

    # =========================================================================
    # 5. PARAMETER SERIALIZATION & VALIDATION
    # =========================================================================
    def test_state_dict_get_set_parameters(self):
        """Verifies state dict extraction and restoring with strict validation."""
        params = self.block.get_parameters()
        self.assertSetEqual(set(params.keys()), {"attn", "ln1", "ffn", "ln2"})

        new_block = TransformerBlock(self.d_model, self.n_heads, self.d_ff)
        new_block.set_parameters(params, strict=True)

        x = np.random.randn(4, self.d_model)
        out1 = self.block.forward(x)
        out2 = new_block.forward(x)

        max_diff = np.max(np.abs(out1 - out2))
        self.assertLess(max_diff, 1e-10)

    def test_invalid_parameters_rejection(self):
        """Verifies that missing or invalid parameter keys raise KeyError / TypeError."""
        with self.assertRaises(TypeError):
            self.block.set_parameters("invalid")

        with self.assertRaises(KeyError):
            self.block.set_parameters({"attn": {}})

        params = self.block.get_parameters()
        params["unexpected_key"] = 123
        with self.assertRaises(KeyError):
            self.block.set_parameters(params, strict=True)

    # =========================================================================
    # 6. EDGE CASES & NUMERICAL STABILITY
    # =========================================================================
    def test_sequence_length_1_and_2(self):
        """Tests sequence length 1 and 2 edge cases."""
        for seq_len in [1, 2]:
            x = np.random.randn(seq_len, self.d_model)
            out = self.block.forward(x)
            self.assertEqual(out.shape, (seq_len, self.d_model))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_maximum_sequence_length(self):
        """Tests block forward on sequence length 256."""
        max_seq_len = 256
        x = np.random.randn(max_seq_len, self.d_model)
        out = self.block.forward(x)
        self.assertEqual(out.shape, (max_seq_len, self.d_model))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_empty_input_array(self):
        """Tests empty input (0, d_model) returns (0, d_model) cleanly."""
        empty_x = np.zeros((0, self.d_model))
        out = self.block.forward(empty_x)
        self.assertEqual(out.shape, (0, self.d_model))

        out_c, new_c = self.block.forward(empty_x, use_kv_cache=True)
        self.assertEqual(out_c.shape, (0, self.d_model))
        self.assertEqual(new_c[0].shape, (0, self.d_model))
        self.assertEqual(new_c[1].shape, (0, self.d_model))

    def test_1d_input_array(self):
        """Tests 1D vector (d_model,) automatically reshaped to (1, d_model)."""
        x_1d = np.random.randn(self.d_model)
        out = self.block.forward(x_1d)
        self.assertEqual(out.shape, (1, self.d_model))

    def test_mismatched_feature_dimension_rejection(self):
        """Tests that mismatched feature dimension raises ValueError."""
        x_invalid = np.random.randn(4, self.d_model + 5)
        with self.assertRaises(ValueError):
            self.block.forward(x_invalid)

    def test_numerical_stability_extreme_inputs(self):
        """Tests that large inputs do not cause NaN or exploding values."""
        large_x = np.random.randn(8, self.d_model) * 1e3
        out = self.block.forward(large_x)
        self.assertEqual(out.shape, (8, self.d_model))
        self.assertTrue(np.all(np.isfinite(out)), "Extreme input caused NaN or Inf in TransformerBlock!")

if __name__ == "__main__":
    unittest.main()
