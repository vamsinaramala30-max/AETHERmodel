"""
AETHER MODEL — Phase 6: Complete Transformer Forward Path & LM Head Tests
Tests forward pipeline, full vs KV-cached equivalence, shape contracts, checkpoint fidelity,
empty input handling, max sequence length, and deterministic token generation.
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
from tokenizer.tokenizer import AetherTokenizer
from inference.engine import AetherInferenceEngine

class TestTransformerForwardAndLMHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AetherModel()
        vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
        cls.tokenizer = AetherTokenizer(vocab_file=vocab_path, frozen=True)
        cls.engine = AetherInferenceEngine(model=cls.model, tokenizer=cls.tokenizer)
        cls.config = cls.model.config

    def test_checkpoint_status_and_fidelity(self):
        """Verify that authentic trained weights are loaded and status is READY."""
        self.assertEqual(self.model.load_status, "READY")
        self.assertTrue(self.config.has_trained_weights)
        self.assertTrue(len(self.config.weights_hash) > 0)
        self.assertTrue(self.config.vocab_size >= self.tokenizer.vocab_size)
        self.assertTrue(self.config.vocab_size > 0)
        self.assertTrue(self.config.d_model > 0)
        self.assertTrue(self.config.n_layers > 0)
        self.assertTrue(self.config.n_heads > 0)
        self.assertTrue(self.config.d_ff > 0)
        self.assertTrue(self.config.max_seq_len > 0)

    def test_untrained_and_missing_checkpoint_fidelity(self):
        """Verify has_trained_weights flag correctly reflects missing / untrained state."""
        untrained = AetherModel(skip_checkpoint=True)
        self.assertFalse(untrained.config.has_trained_weights)
        self.assertEqual(untrained.load_status, "READY")

        missing = AetherModel(ModelConfig(weights_path="nonexistent.json"))
        self.assertFalse(missing.config.has_trained_weights)
        self.assertEqual(missing.load_status, "CHECKPOINT_MISSING")

    def test_architecture_tensor_shapes(self):
        """Verify all layer weights and biases conform to exact dimensional specification."""
        emb = self.model.architecture.token_embedding
        lm_head = self.model.architecture.lm_head
        final_ln = self.model.architecture.final_ln
        v_size = self.config.vocab_size
        d_m = self.config.d_model
        d_ff = self.config.d_ff

        self.assertEqual(emb.weight.shape, (v_size, d_m))
        self.assertEqual(lm_head.weight.shape, (d_m, v_size))
        self.assertEqual(lm_head.bias.shape, (v_size,))
        self.assertEqual(final_ln.gamma.shape, (d_m,))
        self.assertEqual(final_ln.beta.shape, (d_m,))

        for b_idx, block in enumerate(self.model.architecture.blocks):
            self.assertEqual(block.attn.q_proj.shape, (d_m, d_m))
            self.assertEqual(block.attn.k_proj.shape, (d_m, d_m))
            self.assertEqual(block.attn.v_proj.shape, (d_m, d_m))
            self.assertEqual(block.attn.out_proj.shape, (d_m, d_m))
            self.assertEqual(block.ln1.gamma.shape, (d_m,))
            self.assertEqual(block.ln1.beta.shape, (d_m,))
            self.assertEqual(block.ffn.w1.shape, (d_m, d_ff))
            self.assertEqual(block.ffn.b1.shape, (d_ff,))
            self.assertEqual(block.ffn.w2.shape, (d_ff, d_m))
            self.assertEqual(block.ffn.b2.shape, (d_m,))
            self.assertEqual(block.ln2.gamma.shape, (d_m,))
            self.assertEqual(block.ln2.beta.shape, (d_m,))

    def test_empty_input_handling(self):
        """Verify graceful handling of empty token sequences."""
        f_out = self.model.forward([])
        self.assertEqual(len(f_out), self.config.vocab_size)
        self.assertTrue(all(v == 0.0 for v in f_out))

        f_all = self.model.forward_all([])
        self.assertEqual(f_all, [])

        f_prompt, f_caches = self.model.forward_prompt([])
        self.assertEqual(len(f_prompt), self.config.vocab_size)
        self.assertTrue(np.all(f_prompt == 0.0))
        self.assertEqual(len(f_caches), self.config.n_layers)
        for k, v in f_caches:
            self.assertEqual(k.shape, (0, self.config.d_model))
            self.assertEqual(v.shape, (0, self.config.d_model))

    def test_forward_pipeline_step_by_step(self):
        """Verify exact step-by-step tensor transitions through the forward pipeline."""
        input_ids = [47, 108, 198, 169]  # 'What is Aether?'
        seq_len = len(input_ids)
        d_m = self.config.d_model
        v_size = self.config.vocab_size

        # 1. Token Embedding lookup
        x_emb = self.model.architecture.token_embedding.forward(input_ids)
        self.assertEqual(x_emb.shape, (seq_len, d_m))

        # 2. Positional Encoding
        x_pos = self.model.architecture.pos_encoding.forward(x_emb)
        self.assertEqual(x_pos.shape, (seq_len, d_m))

        # 3. Transformer Blocks
        h = x_pos
        for block in self.model.architecture.blocks:
            h = block.forward(h)
            self.assertEqual(h.shape, (seq_len, d_m))

        # 4. Final LayerNorm
        h_ln = self.model.architecture.final_ln.forward(h)
        self.assertEqual(h_ln.shape, (seq_len, d_m))

        # 5. LM Head
        logits = self.model.architecture.lm_head.forward(h_ln[-1])
        self.assertEqual(logits.shape, (v_size,))

        all_logits = self.model.architecture.lm_head.forward_all(h_ln)
        self.assertEqual(all_logits.shape, (seq_len, v_size))

        np.testing.assert_allclose(logits, all_logits[-1], atol=1e-12)

    def test_critical_equivalence_and_kv_caching(self):
        """
        CRITICAL TEST:
        Compare forward(full sequence), forward_prompt(prompt), and forward_step(next token).
        """
        test_prompts = [
            "Hello",
            "What is Aether?",
            "What can you help me with?",
            "Explain automation.",
            "How can I plan my week?"
        ]
        v_size = self.config.vocab_size

        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            seq_len = len(tokens)

            # A. forward(full sequence)
            logits_full = np.array(self.model.forward(tokens))
            # B. forward_all(full sequence)
            logits_all = np.array(self.model.forward_all(tokens))
            # C. forward_prompt(prompt)
            logits_prompt, caches_prompt = self.model.forward_prompt(tokens)
            logits_prompt = np.array(logits_prompt)

            self.assertEqual(logits_full.shape, (v_size,))
            self.assertEqual(logits_all.shape, (seq_len, v_size))
            self.assertEqual(logits_prompt.shape, (v_size,))
            self.assertEqual(len(caches_prompt), self.config.n_layers)

            # Exact equality (diff < 1e-12)
            np.testing.assert_allclose(logits_full, logits_all[-1], atol=1e-12)
            np.testing.assert_allclose(logits_full, logits_prompt, atol=1e-12)

            # Advance with next token
            next_token = int(np.argmax(logits_prompt))
            self.assertTrue(0 <= next_token < v_size)

            extended_tokens = tokens + [next_token]

            # Extended forward
            logits_ext_full = np.array(self.model.forward(extended_tokens))
            # Extended prompt
            logits_ext_prompt, caches_ext_prompt = self.model.forward_prompt(extended_tokens)
            logits_ext_prompt = np.array(logits_ext_prompt)
            # Step forward with KV-cache
            logits_step, caches_step = self.model.forward_step(
                next_token,
                start_pos=seq_len,
                layer_caches=caches_prompt
            )
            logits_step = np.array(logits_step)

            np.testing.assert_allclose(logits_ext_full, logits_step, atol=1e-12)
            np.testing.assert_allclose(logits_ext_prompt, logits_step, atol=1e-12)

            for l in range(self.config.n_layers):
                np.testing.assert_allclose(caches_ext_prompt[l][0], caches_step[l][0], atol=1e-12)
                np.testing.assert_allclose(caches_ext_prompt[l][1], caches_step[l][1], atol=1e-12)

    def test_deterministic_output_reproducibility(self):
        """Verify 100% deterministic output generation across 3 consecutive runs."""
        test_prompts = [
            "Hello",
            "What is Aether?",
            "What can you help me with?",
            "Explain automation.",
            "How can I plan my week?"
        ]

        for prompt in test_prompts:
            run_results = []
            for _ in range(3):
                text, meta = self.engine.generate_response(
                    prompt,
                    context={"temperature": 0.0, "deterministic": True, "max_tokens": 25}
                )
                run_results.append((text, meta["tokens_generated"]))

            self.assertEqual(run_results[0], run_results[1])
            self.assertEqual(run_results[1], run_results[2])
            self.assertTrue(len(run_results[0][0]) > 0)

    def test_max_seq_len_and_boundaries(self):
        """Verify sequence boundaries up to and exceeding max_seq_len."""
        seq_at_max = list(range(self.config.max_seq_len))
        logits_max = self.model.forward(seq_at_max)
        self.assertEqual(len(logits_max), self.config.vocab_size)
        self.assertTrue(all(math.isfinite(x) for x in logits_max))

        seq_overflow = list(range(self.config.max_seq_len + 32))
        logits_overflow = self.model.forward(seq_overflow)
        self.assertEqual(len(logits_overflow), self.config.vocab_size)
        self.assertTrue(all(math.isfinite(x) for x in logits_overflow))

    def test_token_id_range_validity(self):
        """Verify all generated token IDs remain strictly within [0, vocab_size)."""
        prompt_ids = self.tokenizer.encode("Aether platform automation")
        gen_tokens = self.engine.generator.generate_tokens(
            prompt_ids,
            max_tokens=30,
            temperature=0.7,
            deterministic=False
        )
        for tid in gen_tokens:
            self.assertTrue(0 <= tid < self.config.vocab_size)

if __name__ == "__main__":
    unittest.main()
