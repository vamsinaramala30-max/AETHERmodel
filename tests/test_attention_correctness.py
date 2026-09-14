"""
AETHER MODEL — Transformer Attention Correctness & KV-Cache Verification Suite
Phase 4 Comprehensive Audit & Regression Testing.

Verifies:
1. Multi-Head Attention equations & tensor shapes (Q, K, V, scaled scores, causal mask, softmax, context aggregation, out projection)
2. Numerical equivalence between:
   - Full sequence forward pass
   - Prompt prefill + 1 cached step
   - Prompt prefill + multiple cached steps
   - Step-by-step cached pass from token 0
3. Strict Causal Masking:
   - Token at position i receives 0 information from positions > i
   - Token at position i responds to modifications at positions <= i
4. Edge cases:
   - Sequence length 1
   - Sequence length 2
   - Maximum sequence length
   - Empty input & empty cache
   - 1D vector input
   - Invalid cache shapes & error handling
5. Numerical stability & NaN/Inf prevention
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
from model.model import AetherModel
from model.architecture.attention import MultiHeadAttention
from model.architecture.transformer_block import TransformerBlock

class TestAttentionCorrectness(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=128,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            max_seq_len=64
        )
        self.model = AetherModel(self.config, skip_checkpoint=True)
        self.tol = 1e-5

    # =========================================================================
    # 1. TENSOR SHAPES & MULTI-HEAD DIMENSIONS
    # =========================================================================
    def test_multi_head_dimensions_and_projections(self):
        """Verifies n_heads, d_k, projection shapes, and multi-head split correctness."""
        d_model = 32
        n_heads = 4
        d_k = d_model // n_heads  # 8
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.assertEqual(attn.d_model, d_model)
        self.assertEqual(attn.n_heads, n_heads)
        self.assertEqual(attn.d_k, d_k)
        self.assertEqual(attn.q_proj.shape, (d_model, d_model))
        self.assertEqual(attn.k_proj.shape, (d_model, d_model))
        self.assertEqual(attn.v_proj.shape, (d_model, d_model))
        self.assertEqual(attn.out_proj.shape, (d_model, d_model))

        seq_len = 5
        x = np.random.randn(seq_len, d_model)
        out = attn.forward(x)

        self.assertEqual(out.shape, (seq_len, d_model))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_invalid_d_model_n_heads_combination(self):
        """Verifies assertion when d_model is not divisible by n_heads."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(d_model=33, n_heads=4)

    # =========================================================================
    # 2. CRITICAL TEST: FULL SEQUENCE VS KV-CACHE EQUIVALENCE
    # =========================================================================
    def test_critical_full_sequence_vs_cached_inference_equivalence(self):
        """
        CRITICAL TEST:
        Runs the SAME token sequence through:
        A. Normal full-sequence forward pass
        B. Prompt prefill + 1-token cached forward_step
        C. Prompt prefill + multiple-token cached forward_step
        D. Step-by-step cached forward_step from position 0
        """
        token_seq = [7, 23, 45, 88, 12, 61, 99, 104, 3, 50]
        seq_len = len(token_seq)

        # A. Normal full-sequence pass
        full_logits = self.model.forward(token_seq)
        all_logits = self.model.forward_all(token_seq)

        # Verify full_logits matches last row of forward_all
        max_diff_full = np.max(np.abs(np.array(full_logits) - np.array(all_logits[-1])))
        self.assertLess(max_diff_full, self.tol, f"forward() vs forward_all()[-1] mismatch: {max_diff_full}")

        # B. Prompt + one-token cached attention
        prefix_len = seq_len - 1
        prompt_prefix = token_seq[:prefix_len]
        last_token = token_seq[-1]

        prompt_logits, prompt_caches = self.model.forward_prompt(prompt_prefix)
        one_step_logits, one_step_caches = self.model.forward_step(
            last_token,
            start_pos=prefix_len,
            layer_caches=prompt_caches
        )

        max_err_b = np.max(np.abs(np.array(full_logits) - np.array(one_step_logits)))
        self.assertLess(
            max_err_b,
            self.tol,
            f"[TEST B FAIL] Prompt + 1-step cached logits discrepancy: {max_err_b:.2e}"
        )

        # C. Prompt + multiple-token cached attention (e.g. prefix length 3, 7 steps)
        prefix_len_c = 3
        prompt_c = token_seq[:prefix_len_c]
        remaining_c = token_seq[prefix_len_c:]

        _, multi_caches = self.model.forward_prompt(prompt_c)
        curr_pos = prefix_len_c
        multi_step_logits = None
        for tok in remaining_c:
            multi_step_logits, multi_caches = self.model.forward_step(
                tok,
                start_pos=curr_pos,
                layer_caches=multi_caches
            )
            curr_pos += 1

        max_err_c = np.max(np.abs(np.array(full_logits) - np.array(multi_step_logits)))
        self.assertLess(
            max_err_c,
            self.tol,
            f"[TEST C FAIL] Prompt + multi-step cached logits discrepancy: {max_err_c:.2e}"
        )

        # D. Step-by-step from position 0 (empty cache)
        step_caches = None
        step_logits = None
        for pos, tok in enumerate(token_seq):
            step_logits, step_caches = self.model.forward_step(
                tok,
                start_pos=pos,
                layer_caches=step_caches
            )

        max_err_d = np.max(np.abs(np.array(full_logits) - np.array(step_logits)))
        self.assertLess(
            max_err_d,
            self.tol,
            f"[TEST D FAIL] Step-by-step cached logits discrepancy: {max_err_d:.2e}"
        )

        # Verify intermediate positions in step-by-step match forward_all
        step_caches = None
        for pos, tok in enumerate(token_seq):
            step_logits, step_caches = self.model.forward_step(
                tok,
                start_pos=pos,
                layer_caches=step_caches
            )
            expected_row = all_logits[pos]
            diff_pos = np.max(np.abs(np.array(step_logits) - np.array(expected_row)))
            self.assertLess(
                diff_pos,
                self.tol,
                f"Position {pos} intermediate step mismatch: {diff_pos:.2e}"
            )

    # =========================================================================
    # 3. STRICT CAUSAL MASKING VERIFICATION
    # =========================================================================
    def test_strict_causal_isolation(self):
        """
        A token at position i must NOT receive ANY information from positions > i.
        Perturbing tokens at positions > i must have exactly 0 effect on representations at position i.
        """
        base_tokens = [10, 20, 30, 40, 50, 60]
        seq_len = len(base_tokens)

        base_all_logits = np.array(self.model.forward_all(base_tokens))

        for target_pos in range(seq_len - 1):
            # Modify tokens that appear strictly AFTER target_pos
            perturbed_tokens = list(base_tokens)
            for future_pos in range(target_pos + 1, seq_len):
                perturbed_tokens[future_pos] = (perturbed_tokens[future_pos] + 17) % self.config.vocab_size

            perturbed_all_logits = np.array(self.model.forward_all(perturbed_tokens))

            # Position target_pos must be mathematically identical
            diff_at_target = np.max(np.abs(base_all_logits[target_pos] - perturbed_all_logits[target_pos]))
            self.assertEqual(
                diff_at_target,
                0.0,
                f"Causality violation! Modification at future position altered position {target_pos}, diff={diff_at_target}"
            )

            # Positions before target_pos must also be identical
            for prev_pos in range(target_pos):
                diff_prev = np.max(np.abs(base_all_logits[prev_pos] - perturbed_all_logits[prev_pos]))
                self.assertEqual(
                    diff_prev,
                    0.0,
                    f"Causality violation! Future modification altered earlier position {prev_pos}, diff={diff_prev}"
                )

    def test_causal_dependency_on_past(self):
        """Modifying input at position j <= i MUST change the representation at position i."""
        # 1. Test directly on MultiHeadAttention layer
        attn = MultiHeadAttention(d_model=self.config.d_model, n_heads=self.config.n_heads)
        base_x = np.random.randn(4, self.config.d_model)
        base_attn_out = attn.forward(base_x)

        # Perturb position 0
        perturbed_x = base_x.copy()
        perturbed_x[0] += 1.5
        perturbed_attn_out = attn.forward(perturbed_x)

        # Positions 0, 1, 2, 3 must all change since they attend to pos 0
        for pos in range(4):
            diff = np.max(np.abs(base_attn_out[pos] - perturbed_attn_out[pos]))
            self.assertGreater(
                diff,
                1e-3,
                f"Expected attention output at position {pos} to depend on past position 0, got diff={diff}"
            )

        # 2. Test perturbing position 2
        perturbed_x2 = base_x.copy()
        perturbed_x2[2] += 2.0
        perturbed_attn_out2 = attn.forward(perturbed_x2)

        # Positions 0 and 1 must NOT change (exact 0.0)
        self.assertEqual(np.max(np.abs(base_attn_out[0] - perturbed_attn_out2[0])), 0.0)
        self.assertEqual(np.max(np.abs(base_attn_out[1] - perturbed_attn_out2[1])), 0.0)

        # Positions 2 and 3 MUST change
        self.assertGreater(np.max(np.abs(base_attn_out[2] - perturbed_attn_out2[2])), 1e-3)
        self.assertGreater(np.max(np.abs(base_attn_out[3] - perturbed_attn_out2[3])), 1e-3)

    # =========================================================================
    # 4. EDGE CASES
    # =========================================================================
    def test_sequence_length_1(self):
        """Tests sequence length 1 forward, cached, and single-step."""
        tokens = [42]
        logits_full = self.model.forward(tokens)
        self.assertEqual(len(logits_full), self.config.vocab_size)
        self.assertTrue(all(math.isfinite(v) for v in logits_full))

        logits_step, caches = self.model.forward_step(42, start_pos=0, layer_caches=None)
        max_diff = np.max(np.abs(np.array(logits_full) - np.array(logits_step)))
        self.assertLess(max_diff, self.tol)

        # Single step with MultiHeadAttention directly
        attn = self.model.architecture.blocks[0].attn
        x1 = np.random.randn(1, self.config.d_model)
        out1 = attn.forward(x1)
        self.assertEqual(out1.shape, (1, self.config.d_model))
        self.assertTrue(np.all(np.isfinite(out1)))

    def test_sequence_length_2(self):
        """Tests sequence length 2 forward vs cached step."""
        tokens = [15, 73]
        logits_full = self.model.forward(tokens)

        step0_logits, caches0 = self.model.forward_step(15, start_pos=0, layer_caches=None)
        step1_logits, caches1 = self.model.forward_step(73, start_pos=1, layer_caches=caches0)

        max_diff = np.max(np.abs(np.array(logits_full) - np.array(step1_logits)))
        self.assertLess(max_diff, self.tol)

    def test_maximum_sequence_length(self):
        """Tests forward pass and step caching up to max_seq_len."""
        max_len = self.config.max_seq_len  # 64
        tokens = [(i * 7 + 3) % self.config.vocab_size for i in range(max_len)]

        logits_full = self.model.forward(tokens)
        self.assertEqual(len(logits_full), self.config.vocab_size)
        self.assertTrue(all(math.isfinite(v) for v in logits_full))

        # Prompt prefill max_len - 1 + step 1
        prompt_len = max_len - 1
        prompt_logits, caches = self.model.forward_prompt(tokens[:prompt_len])
        final_step_logits, _ = self.model.forward_step(tokens[-1], start_pos=prompt_len, layer_caches=caches)

        max_diff = np.max(np.abs(np.array(logits_full) - np.array(final_step_logits)))
        self.assertLess(max_diff, self.tol)

    def test_empty_sequence_and_empty_cache(self):
        """Tests empty input handling and empty cache edge cases."""
        attn = self.model.architecture.blocks[0].attn
        empty_x = np.zeros((0, self.config.d_model))

        out = attn.forward(empty_x)
        self.assertEqual(out.shape, (0, self.config.d_model))

        out_c, new_c = attn.forward(empty_x, use_kv_cache=True)
        self.assertEqual(out_c.shape, (0, self.config.d_model))
        self.assertEqual(new_c[0].shape, (0, self.config.d_model))
        self.assertEqual(new_c[1].shape, (0, self.config.d_model))

        # Model level empty input
        logits_empty = self.model.forward([])
        self.assertEqual(len(logits_empty), self.config.vocab_size)
        self.assertEqual(self.model.forward_all([]), [])

    def test_1d_input_handling(self):
        """Tests 1D vector input (d_model,) automatically reshaped to (1, d_model)."""
        attn = self.model.architecture.blocks[0].attn
        x_1d = np.random.randn(self.config.d_model)

        out = attn.forward(x_1d)
        self.assertEqual(out.shape, (1, self.config.d_model))

    def test_invalid_cache_validation(self):
        """Tests that invalid cache structures raise descriptive ValueErrors."""
        attn = self.model.architecture.blocks[0].attn
        x = np.random.randn(2, self.config.d_model)

        # Not a tuple/list
        with self.assertRaises(ValueError):
            attn.forward(x, kv_cache="invalid_cache")

        # Wrong tuple length
        with self.assertRaises(ValueError):
            attn.forward(x, kv_cache=(np.zeros((2, self.config.d_model)),))

        # Mismatched sequence length between K and V
        with self.assertRaises(ValueError):
            attn.forward(
                x,
                kv_cache=(
                    np.zeros((3, self.config.d_model)),
                    np.zeros((2, self.config.d_model))
                )
            )

        # Mismatched feature dimension
        with self.assertRaises(ValueError):
            attn.forward(
                x,
                kv_cache=(
                    np.zeros((2, self.config.d_model + 4)),
                    np.zeros((2, self.config.d_model + 4))
                )
            )

    # =========================================================================
    # 5. NUMERICAL STABILITY & NAN/INF PREVENTION
    # =========================================================================
    def test_numerical_stability_large_magnitude_inputs(self):
        """Tests that extremely large input values do not trigger NaN or Inf in attention."""
        attn = self.model.architecture.blocks[0].attn

        large_x = np.random.randn(8, self.config.d_model) * 1e4
        out = attn.forward(large_x)

        self.assertEqual(out.shape, (8, self.config.d_model))
        self.assertTrue(np.all(np.isfinite(out)), "Output contains NaN or Inf on large inputs!")

        # Attention weights should sum to 1.0 per query row per head
        attn_weights = attn._cache["attn_weights"]
        sum_weights = np.sum(attn_weights, axis=-1)
        max_sum_err = np.max(np.abs(sum_weights - 1.0))
        self.assertLess(max_sum_err, 1e-4, f"Softmax weights do not sum to 1: max err = {max_sum_err}")

    def test_multi_head_cross_head_independence(self):
        """Verifies that different attention heads compute independent attention patterns."""
        d_model = 32
        n_heads = 4
        d_k = 8
        attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        x = np.random.randn(6, d_model)
        attn.forward(x)
        weights = attn._cache["attn_weights"]  # [n_heads, 6, 6]

        self.assertEqual(weights.shape, (n_heads, 6, 6))
        # Ensure head 0 and head 1 are not identical
        diff_h0_h1 = np.max(np.abs(weights[0] - weights[1]))
        self.assertGreater(diff_h0_h1, 1e-4, "Expected different heads to have independent attention patterns")

    def test_intermediate_hidden_states_cached_vs_uncached(self):
        """Verifies that intermediate hidden states at each block level match between full and cached passes."""
        tokens = [12, 34, 56, 78, 90]
        # 1. Full pass representations
        x_full = self.model.architecture.token_embedding.forward(tokens)
        x_full = self.model.architecture.pos_encoding.forward(x_full)

        block_outputs_full = []
        x_curr = x_full
        for block in self.model.architecture.blocks:
            x_curr = block.forward(x_curr)
            block_outputs_full.append(x_curr)

        # 2. Step-by-step cached representations
        caches = None
        for pos, tok in enumerate(tokens):
            x_step = self.model.architecture.token_embedding.forward([tok])
            x_step = self.model.architecture.pos_encoding.forward(x_step, start_pos=pos)

            new_caches = []
            for b_idx, block in enumerate(self.model.architecture.blocks):
                c = caches[b_idx] if caches is not None else None
                x_step, new_c = block.forward(x_step, kv_cache=c, use_kv_cache=True)
                new_caches.append(new_c)

                # Check intermediate hidden state at this position
                expected_hidden = block_outputs_full[b_idx][pos:pos+1]
                diff = np.max(np.abs(x_step - expected_hidden))
                self.assertLess(
                    diff,
                    self.tol,
                    f"Block {b_idx} hidden state mismatch at pos {pos}: diff={diff:.2e}"
                )
            caches = new_caches

    def test_various_head_and_dimension_configurations(self):
        """Tests attention correctness across various n_heads and d_model configurations."""
        configs = [
            (16, 1),   # Single head (n_heads = 1)
            (32, 2),   # 2 heads
            (64, 4),   # 4 heads
            (64, 8),   # 8 heads
            (128, 16), # 16 heads
        ]
        for d_m, n_h in configs:
            cfg = ModelConfig(vocab_size=64, d_model=d_m, n_layers=2, n_heads=n_h, d_ff=d_m * 2, max_seq_len=32)
            m = AetherModel(cfg, skip_checkpoint=True)
            seq = [5, 12, 19, 26, 33]

            full_logits = m.forward(seq)

            # Step-by-step
            caches = None
            step_logits = None
            for p, t in enumerate(seq):
                step_logits, caches = m.forward_step(t, start_pos=p, layer_caches=caches)

            diff = np.max(np.abs(np.array(full_logits) - np.array(step_logits)))
            self.assertLess(
                diff,
                self.tol,
                f"Configuration (d_model={d_m}, n_heads={n_h}) equivalence failed: diff={diff:.2e}"
            )

    def test_token_generator_and_streaming_with_kv_cache(self):
        """Verifies TokenGenerator and StreamTokenGenerator deterministic outputs."""
        from tokenizer.tokenizer import AetherTokenizer
        from inference.generation import TokenGenerator
        from inference.streaming import StreamTokenGenerator

        tok = AetherTokenizer()
        generator = TokenGenerator(self.model, tok)
        streamer = StreamTokenGenerator(self.model, tok)

        prompt = [10, 20, 30]
        gen_tokens = generator.generate_tokens(prompt, max_tokens=5, deterministic=True)
        self.assertTrue(isinstance(gen_tokens, list))
        self.assertEqual(len(gen_tokens), 5)

        stream_tokens = []
        for chunk in streamer.stream_generate(prompt, max_tokens=5, deterministic=True):
            if "token_id" in chunk:
                stream_tokens.append(chunk["token_id"])

        self.assertEqual(gen_tokens, stream_tokens, "TokenGenerator and StreamTokenGenerator produced different tokens!")

if __name__ == "__main__":
    unittest.main()
