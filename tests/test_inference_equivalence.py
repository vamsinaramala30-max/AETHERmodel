"""
AETHER MODEL — Numerical Equivalence Test Suite
Verifies that the optimized vectorized Transformer architecture produces numerically equivalent outputs
to the exact mathematical reference scalar formulation within an explicit floating-point tolerance (eps <= 1e-5).
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
from model.architecture.attention import MultiHeadAttention
from model.architecture.feed_forward import FeedForwardNetwork
from model.architecture.normalization import LayerNorm
from model.architecture.output_head import OutputHead

class TestInferenceEquivalence(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(vocab_size=256, d_model=32, n_layers=2, n_heads=2, d_ff=64)
        self.model = AetherModel(self.config)
        self.input_ids = [12, 45, 67, 89, 120]
        self.tol = 1e-5

    def test_token_embedding_equivalence(self):
        emb_layer = self.model.architecture.token_embedding
        vec_out = emb_layer.forward(self.input_ids)
        self.assertIsInstance(vec_out, np.ndarray)
        self.assertEqual(vec_out.shape, (len(self.input_ids), self.config.d_model))

        # Compare against scalar lookup
        for idx, tid in enumerate(self.input_ids):
            ref_row = emb_layer.weight[tid % emb_layer.vocab_size]
            max_err = np.max(np.abs(vec_out[idx] - ref_row))
            self.assertLess(max_err, self.tol)

    def test_positional_encoding_equivalence(self):
        pe_layer = self.model.architecture.pos_encoding
        x_dummy = np.random.randn(len(self.input_ids), self.config.d_model)
        pe_out = pe_layer.forward(x_dummy)

        # Scalar reference computation
        for pos in range(len(self.input_ids)):
            for i in range(self.config.d_model):
                if i % 2 == 0:
                    ref_val = x_dummy[pos, i] + math.sin(pos / (10000.0 ** (i / self.config.d_model)))
                else:
                    ref_val = x_dummy[pos, i] + math.cos(pos / (10000.0 ** ((i - 1) / self.config.d_model)))
                self.assertAlmostEqual(pe_out[pos, i], ref_val, delta=self.tol)

    def test_layer_norm_equivalence(self):
        ln_layer = self.model.architecture.final_ln
        x_dummy = np.random.randn(len(self.input_ids), self.config.d_model)
        ln_out = ln_layer.forward(x_dummy)

        for pos in range(len(self.input_ids)):
            vec = x_dummy[pos]
            mean = np.mean(vec)
            var = np.var(vec)
            std = math.sqrt(var + ln_layer.eps)
            for i in range(self.config.d_model):
                ref_val = ln_layer.gamma[i] * ((vec[i] - mean) / std) + ln_layer.beta[i]
                self.assertAlmostEqual(ln_out[pos, i], ref_val, delta=self.tol)

    def test_feed_forward_equivalence(self):
        ffn = self.model.architecture.blocks[0].ffn
        x_dummy = np.random.randn(len(self.input_ids), self.config.d_model)
        ffn_out = ffn.forward(x_dummy)

        # Scalar reference implementation
        for pos_idx, vec in enumerate(x_dummy):
            h1_act = []
            for j in range(ffn.d_ff):
                val = ffn.b1[j]
                for i in range(ffn.d_model):
                    val += vec[i] * ffn.w1[i, j]
                gelu_val = 0.5 * val * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (val + 0.044715 * (val ** 3))))
                h1_act.append(gelu_val)

            out_vec = []
            for j in range(ffn.d_model):
                val = ffn.b2[j]
                for i in range(ffn.d_ff):
                    val += h1_act[i] * ffn.w2[i, j]
                out_vec.append(val)

            max_err = np.max(np.abs(ffn_out[pos_idx] - out_vec))
            self.assertLess(max_err, self.tol)

    def test_attention_equivalence(self):
        attn = self.model.architecture.blocks[0].attn
        x_dummy = np.random.randn(len(self.input_ids), self.config.d_model)
        attn_out = attn.forward(x_dummy)

        # Multi-Head reference attention calculation
        seq_len = len(self.input_ids)
        queries = [x_dummy[t] @ attn.q_proj for t in range(seq_len)]
        keys = [x_dummy[t] @ attn.k_proj for t in range(seq_len)]
        values = [x_dummy[t] @ attn.v_proj for t in range(seq_len)]

        scale = 1.0 / math.sqrt(attn.d_k)
        ref_attn_out = np.zeros((seq_len, self.config.d_model))

        for i in range(seq_len):
            head_contexts = []
            for h in range(attn.n_heads):
                h_start = h * attn.d_k
                h_end = (h + 1) * attn.d_k
                q_h_i = queries[i][h_start:h_end]

                scores = []
                for j in range(i + 1):
                    k_h_j = keys[j][h_start:h_end]
                    scores.append(np.dot(q_h_i, k_h_j) * scale)

                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exps) or 1e-9
                weights = [e / sum_e for e in exps]

                ctx_h = np.zeros(attn.d_k)
                for j in range(i + 1):
                    v_h_j = values[j][h_start:h_end]
                    ctx_h += weights[j] * v_h_j
                head_contexts.append(ctx_h)

            full_ctx = np.concatenate(head_contexts)
            ref_attn_out[i] = full_ctx @ attn.out_proj

        max_err = np.max(np.abs(attn_out - ref_attn_out))
        self.assertLess(max_err, self.tol)

    def test_full_model_forward_logits_equivalence(self):
        logits = self.model.forward(self.input_ids)
        self.assertEqual(len(logits), self.config.vocab_size)
        self.assertTrue(all(math.isfinite(val) for val in logits))

        logits_all = self.model.forward_all(self.input_ids)
        self.assertEqual(len(logits_all), len(self.input_ids))
        self.assertEqual(len(logits_all[-1]), self.config.vocab_size)

        max_err = np.max(np.abs(np.array(logits) - np.array(logits_all[-1])))
        self.assertLess(max_err, self.tol)

    def test_single_step_kv_cache_equivalence(self):
        """Verifies that single-step KV-cache forward pass matches full sequence forward pass exactly."""
        full_logits = self.model.forward(self.input_ids)

        layer_caches = None
        single_logits = None
        for pos, tid in enumerate(self.input_ids):
            single_logits, layer_caches = self.model.forward_step(tid, start_pos=pos, layer_caches=layer_caches)

        max_err = np.max(np.abs(np.array(full_logits) - np.array(single_logits)))
        self.assertLess(max_err, self.tol, f"KV-cache single step discrepancy: max_err={max_err}")

    def test_parallel_prompt_kv_cache_equivalence(self):
        """Verifies that parallel forward_prompt pass matches step-by-step forward_step pass exactly."""
        step_caches = None
        step_logits = None
        for pos, tid in enumerate(self.input_ids):
            step_logits, step_caches = self.model.forward_step(tid, start_pos=pos, layer_caches=step_caches)

        prompt_logits, prompt_caches = self.model.forward_prompt(self.input_ids)

        max_err_logits = np.max(np.abs(np.array(step_logits) - np.array(prompt_logits)))
        self.assertLess(max_err_logits, self.tol, f"Parallel prompt logits discrepancy: max_err={max_err_logits}")

        for l_idx in range(len(step_caches)):
            step_k, step_v = step_caches[l_idx]
            prompt_k, prompt_v = prompt_caches[l_idx]
            max_err_k = np.max(np.abs(step_k - prompt_k))
            max_err_v = np.max(np.abs(step_v - prompt_v))
            self.assertLess(max_err_k, self.tol, f"Parallel prompt KV cache K discrepancy in layer {l_idx}: max_err={max_err_k}")
            self.assertLess(max_err_v, self.tol, f"Parallel prompt KV cache V discrepancy in layer {l_idx}: max_err={max_err_v}")

if __name__ == "__main__":
    unittest.main()
