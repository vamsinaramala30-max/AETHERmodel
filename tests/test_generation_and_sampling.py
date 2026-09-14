"""
AETHER MODEL — Phase 7 Generation, Sampling, Streaming & Decoding Test Suite
Verifies all 16 Phase 7 requirements:
1. Generation position handling
2. Prompt-length boundary errors
3. Max_seq_len boundary handling
4. EOS handling
5. Invalid token handling
6. Repetition penalty behavior
7. No-repeat n-gram behavior
8. Top-k behavior
9. Top-p behavior
10. Deterministic generation
11. Random sampling stability
12. Tokenizer spacing
13. Streaming token boundaries
14. Timeout handling
15. Cancellation handling
16. Duplicate final SSE events
"""

import sys
import os
import time
import unittest
import numpy as np
from typing import List

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer, EOS_TOKEN_ID, UNK_TOKEN_ID
from inference.sampling import (
    apply_repetition_penalty,
    apply_no_repeat_ngram_blocker,
    sample_next_token
)
from inference.generation import TokenGenerator
from inference.streaming import StreamTokenGenerator
from inference.engine import AetherInferenceEngine
from serving.streaming import format_sse_chunk

class TestSamplingStrategies(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.dummy_logits = np.array([float(i) for i in range(self.vocab_size)], dtype=np.float64)

    def test_repetition_penalty_positive_and_negative_logits(self):
        logits = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float64)
        history = [0, 4]  # Seen tokens: index 0 (-4.0) and index 4 (4.0)
        penalized = apply_repetition_penalty(logits, history, penalty=2.0)

        # Negative logit multiplied by penalty: -4.0 * 2.0 = -8.0 (less likely)
        self.assertAlmostEqual(penalized[0], -8.0)
        # Positive logit divided by penalty: 4.0 / 2.0 = 2.0 (less likely)
        self.assertAlmostEqual(penalized[4], 2.0)
        # Unseen tokens unchanged
        self.assertAlmostEqual(penalized[1], -2.0)
        self.assertAlmostEqual(penalized[3], 2.0)

    def test_repetition_penalty_edge_cases(self):
        logits = np.array([1.0, 2.0, 3.0])
        # Penalty 1.0 has no effect
        p1 = apply_repetition_penalty(logits, [0, 1], penalty=1.0)
        np.testing.assert_array_almost_equal(p1, logits)

        # Non-positive penalty returns unchanged
        p0 = apply_repetition_penalty(logits, [0, 1], penalty=0.0)
        np.testing.assert_array_almost_equal(p0, logits)

        # Empty history returns unchanged
        p_empty = apply_repetition_penalty(logits, [], penalty=1.5)
        np.testing.assert_array_almost_equal(p_empty, logits)

    def test_no_repeat_ngram_blocker(self):
        # Sequence: A, B, C, A, B -> prefix is (A, B) -> continuation C must be blocked
        history = [10, 20, 30, 10, 20]
        logits = np.zeros(50, dtype=np.float64)
        penalized = apply_no_repeat_ngram_blocker(logits, history, ngram_size=3)

        self.assertEqual(penalized[30], -1e9, "Token 30 should be banned to prevent repeating 3-gram (10, 20, 30)")
        self.assertEqual(penalized[10], 0.0)
        self.assertEqual(penalized[20], 0.0)

    def test_no_repeat_ngram_blocker_cross_prompt_boundary(self):
        # Prompt: [5, 6, 7], Generated so far: [8, 5, 6] -> prefix is (5, 6) -> 7 should be blocked
        history = [5, 6, 7, 8, 5, 6]
        logits = np.zeros(20, dtype=np.float64)
        penalized = apply_no_repeat_ngram_blocker(logits, history, ngram_size=3)
        self.assertEqual(penalized[7], -1e9, "Token 7 should be blocked across prompt boundary")

    def test_sample_next_token_deterministic_argmax(self):
        logits = np.array([1.0, 5.0, 2.0, 0.5])
        # temperature = 0.0
        self.assertEqual(sample_next_token(logits, temperature=0.0), 1)
        # deterministic = True
        self.assertEqual(sample_next_token(logits, temperature=0.8, deterministic=True), 1)
        # top_k = 1
        self.assertEqual(sample_next_token(logits, temperature=0.8, top_k=1), 1)

    def test_sample_next_token_top_k(self):
        logits = np.zeros(100, dtype=np.float64)
        logits[10] = 10.0
        logits[20] = 9.0
        logits[30] = 8.0
        # top_k = 2 should only ever choose index 10 or 20
        chosen = set()
        for s in range(50):
            token = sample_next_token(logits, temperature=1.0, top_k=2, top_p=1.0, seed=s)
            chosen.add(token)
        self.assertTrue(chosen.issubset({10, 20}), f"Expected subset of {{10, 20}}, got {chosen}")

    def test_sample_next_token_top_p_nucleus(self):
        logits = np.array([10.0, 9.9, 0.1, 0.0, -10.0])
        # Top-p = 0.8 should only keep top candidates
        chosen = set()
        for s in range(50):
            token = sample_next_token(logits, temperature=0.7, top_k=0, top_p=0.8, seed=s)
            chosen.add(token)
        self.assertTrue(chosen.issubset({0, 1}))

    def test_sample_next_token_seed_stability(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        run1 = [sample_next_token(logits, temperature=0.9, top_k=5, seed=12345 + i) for i in range(20)]
        run2 = [sample_next_token(logits, temperature=0.9, top_k=5, seed=12345 + i) for i in range(20)]
        self.assertEqual(run1, run2, "Sampling with identical seeds must produce identical token sequences")

    def test_sample_next_token_nan_inf_safety(self):
        logits = np.array([np.nan, 2.0, np.inf, -np.inf, 1.0])
        token = sample_next_token(logits, temperature=0.7)
        self.assertIn(token, range(len(logits)))


class TestAutoregressiveGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AetherModel()
        vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
        cls.tokenizer = AetherTokenizer(vocab_file=vocab_path, frozen=True)
        cls.generator = TokenGenerator(cls.model, cls.tokenizer)
        cls.stream_generator = StreamTokenGenerator(cls.model, cls.tokenizer)

    def test_empty_prompt(self):
        gen = self.generator.generate_tokens([])
        self.assertEqual(gen, [])

        stream_chunks = list(self.stream_generator.stream_generate([]))
        self.assertEqual(len(stream_chunks), 1)
        self.assertTrue(stream_chunks[0]["done"])
        self.assertEqual(stream_chunks[0]["finish_reason"], "empty")

    def test_invalid_token_ids_in_prompt(self):
        # Prompt with negative or out-of-vocab IDs
        invalid_prompt = [-5, 99999, 118]
        gen = self.generator.generate_tokens(invalid_prompt, max_tokens=5, deterministic=True)
        self.assertIsInstance(gen, list)
        self.assertTrue(all(0 <= tid < self.model.config.vocab_size for tid in gen))

    def test_max_seq_len_boundary_and_oversized_prompt(self):
        max_len = self.model.config.max_seq_len
        # Oversized prompt: 300 tokens (greater than max_seq_len=256)
        long_prompt = [118] * 300
        gen = self.generator.generate_tokens(long_prompt, max_tokens=10, deterministic=True)
        self.assertIsInstance(gen, list)
        # Should generate at least 1 token without index error
        self.assertGreater(len(gen), 0)
        self.assertTrue(all(0 <= tid < self.model.config.vocab_size for tid in gen))

    def test_stop_tokens_handling(self):
        prompt = self.tokenizer.encode("Hello")
        # Custom stop token test
        gen = self.generator.generate_tokens(prompt, max_tokens=50, stop_token_ids=[EOS_TOKEN_ID], deterministic=True)
        self.assertNotIn(EOS_TOKEN_ID, gen, "Stop token should not be included in generated tokens")

    def test_cancellation_handling(self):
        prompt = self.tokenizer.encode("Explain automation in detail.")
        cancel_called = [0]

        def cancel_checker():
            cancel_called[0] += 1
            return cancel_called[0] >= 3

        gen = self.generator.generate_tokens(prompt, max_tokens=50, is_cancelled=cancel_checker)
        self.assertLessEqual(len(gen), 5, "Generation should abort promptly upon cancellation")

        # Stream cancellation
        cancel_called[0] = 0
        chunks = list(self.stream_generator.stream_generate(prompt, max_tokens=50, is_cancelled=cancel_checker))
        self.assertTrue(chunks[-1]["done"])
        self.assertEqual(chunks[-1]["finish_reason"], "cancelled")

    def test_timeout_handling(self):
        prompt = self.tokenizer.encode("Explain automation in detail.")
        # Very short timeout
        gen = self.generator.generate_tokens(prompt, max_tokens=100, timeout_sec=0.0001)
        self.assertLessEqual(len(gen), 5)

        stream_chunks = list(self.stream_generator.stream_generate(prompt, max_tokens=100, timeout_sec=0.0001))
        self.assertTrue(stream_chunks[-1]["done"])
        self.assertEqual(stream_chunks[-1]["finish_reason"], "timeout")

    def test_deterministic_reproducibility(self):
        prompt = self.tokenizer.encode("What is Aether?")
        runs = [
            self.generator.generate_tokens(prompt, max_tokens=25, deterministic=True)
            for _ in range(5)
        ]
        first_run = runs[0]
        for idx, r in enumerate(runs[1:], 2):
            self.assertEqual(r, first_run, f"Run {idx} differed from Run 1 in deterministic mode!")

    def test_streaming_vs_non_streaming_exact_equivalence(self):
        prompts = [
            "Hello",
            "What is Aether?",
            "Explain automation.",
            "How can I plan my week?"
        ]
        for p in prompts:
            prompt_ids = self.tokenizer.encode(p)

            # 1. Non-streaming generation
            non_stream_ids = self.generator.generate_tokens(prompt_ids, max_tokens=30, deterministic=True)
            non_stream_text = self.tokenizer.decode(non_stream_ids, skip_special_tokens=True)

            # 2. Streaming generation
            stream_chunks = list(self.stream_generator.stream_generate(prompt_ids, max_tokens=30, deterministic=True))
            stream_ids = [c["token_id"] for c in stream_chunks if "token_id" in c]
            stream_text = "".join(c.get("delta", "") for c in stream_chunks)

            self.assertEqual(non_stream_ids, stream_ids, f"Token IDs mismatched between non-stream and stream for prompt: '{p}'")
            self.assertEqual(non_stream_text, stream_text, f"Decoded text mismatched between non-stream and stream for prompt: '{p}'")

    def test_streaming_single_done_event(self):
        prompt_ids = self.tokenizer.encode("Hello")
        stream_chunks = list(self.stream_generator.stream_generate(prompt_ids, max_tokens=10, deterministic=True))

        done_chunks = [c for c in stream_chunks if c.get("done") is True]
        self.assertEqual(len(done_chunks), 1, "There must be exactly ONE terminal done chunk in the stream!")
        self.assertIn("finish_reason", done_chunks[0])

    def test_sse_chunk_formatter(self):
        chunk = {"delta": " world", "done": False, "confidence": "HIGH_CONFIDENCE", "has_trained_weights": True}
        sse_str = format_sse_chunk(chunk)
        self.assertTrue(sse_str.startswith("data: "))
        self.assertTrue(sse_str.endswith("\n\n"))
        self.assertIn('"delta": " world"', sse_str)
        self.assertIn('"done": false', sse_str)


class TestAetherInferenceEngineGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AetherInferenceEngine()

    def test_engine_generate_response_deterministic(self):
        prompt = "What is Aether?"
        resp1, meta1 = self.engine.generate_response(prompt, context={"deterministic": True, "max_tokens": 30})
        resp2, meta2 = self.engine.generate_response(prompt, context={"deterministic": True, "max_tokens": 30})

        self.assertEqual(resp1, resp2, "Engine deterministic responses must be identical across runs")
        self.assertEqual(meta1["tokens_generated"], meta2["tokens_generated"])
        self.assertGreater(len(resp1.strip()), 0)

    def test_engine_streaming_matches_non_streaming(self):
        prompt = "What is Aether?"
        context = {"deterministic": True, "max_tokens": 30}

        resp_non_stream, meta_ns = self.engine.generate_response(prompt, context)
        stream = self.engine.stream_generate(prompt, context)
        chunks = list(stream)

        resp_stream = "".join(c.get("delta", "") for c in chunks)
        self.assertEqual(resp_non_stream, resp_stream, "Engine non-streaming and streaming responses must match exactly")

    def test_quality_criteria(self):
        test_prompts = [
            "Hello",
            "What is Aether?",
            "Explain Aether automation."
        ]
        for p in test_prompts:
            text, meta = self.engine.generate_response(p, context={"temperature": 0.0, "max_tokens": 40})
            # 1. Non-empty
            self.assertGreater(len(text.strip()), 0, f"Generated text was empty for prompt: {p}")
            # 2. Not dominated by repeated tokens or loops
            words = text.split()
            if len(words) > 6:
                unique_words = set(words)
                diversity = len(unique_words) / len(words)
                self.assertGreater(diversity, 0.4, f"Text diversity too low ({diversity}) for prompt: {p}")
            # 3. Not corrupted with special tokens
            self.assertNotIn("<eos>", text)
            self.assertNotIn("<unk>", text)
            self.assertNotIn("<pad>", text)

if __name__ == "__main__":
    unittest.main()
