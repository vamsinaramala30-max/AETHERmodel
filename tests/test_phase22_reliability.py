"""
AETHER MODEL — Phase 22 Production Inference & Reliability Unit Test Suite
"""

import os
import sys
import json
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from model.config.model_config import ModelConfig
from tokenizer.tokenizer import AetherTokenizer
from inference.engine import AetherInferenceEngine
from inference.context import ContextManager
from serving.schemas import GenerateRequest, GenerateResponse, HealthResponse
from serving.health import get_health_status


class TestPhase22Reliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_p21 = os.path.join(base_dir, "checkpoints", "aether_checkpoint_p21_best.json")
        cls.tokenizer_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        cls.tokenizer = AetherTokenizer(vocab_file=cls.tokenizer_path, frozen=True)
        cls.model_config = ModelConfig.authoritative(weights_path=cls.ckpt_p21)
        cls.model = AetherModel(cls.model_config)
        cls.engine = AetherInferenceEngine(model=cls.model, tokenizer=cls.tokenizer)

    def test_01_phase21_checkpoint_loaded_and_verified(self):
        """Phase 21 checkpoint must load cleanly with READY status."""
        self.assertTrue(self.model.config.has_trained_weights)
        self.assertEqual(self.model.load_status, "READY")
        self.assertEqual(self.model.config.vocab_size, 1024)
        self.assertEqual(len(self.model.last_validation_errors), 0)

    def test_02_context_manager_role_ordering(self):
        """ContextManager formats tokens in standard sequence ending with Assistant token."""
        from tokenizer.tokenizer import ASSISTANT_TOKEN_ID
        ctx_mgr = ContextManager(self.tokenizer, max_seq_len=256)
        token_ids = ctx_mgr.format_prompt(
            "Plan my week",
            {
                "system_prompt": "System instructions",
                "memory_context": "User memory",
                "rag_context": "Retrieved docs",
            }
        )
        self.assertGreater(len(token_ids), 0)
        self.assertEqual(token_ids[-1], ASSISTANT_TOKEN_ID)

    def test_03_intelligent_context_trimming(self):
        """ContextManager trims excess turns while preserving user request and assistant prefix."""
        from tokenizer.tokenizer import ASSISTANT_TOKEN_ID
        ctx_mgr = ContextManager(self.tokenizer, max_seq_len=64)
        long_history = [{"role": "user", "content": f"Turn message {i} with lots of tokens and details"} for i in range(20)]
        token_ids = ctx_mgr.format_prompt("Current prompt", {"conversation_history": long_history})
        self.assertLess(len(token_ids), 64)
        self.assertEqual(token_ids[-1], ASSISTANT_TOKEN_ID)

    def test_04_deterministic_generation(self):
        """Identical seed produces identical generated tokens."""
        text1, meta1 = self.engine.generate_response("Explain testing.", {"max_tokens": 12, "deterministic": True, "seed": 42})
        text2, meta2 = self.engine.generate_response("Explain testing.", {"max_tokens": 12, "deterministic": True, "seed": 42})
        self.assertEqual(text1, text2)

    def test_05_streaming_lifecycle(self):
        """Streaming yields chunks and ends with done=True."""
        chunks = list(self.engine.stream_generate("Short test", {"max_tokens": 10}))
        self.assertGreater(len(chunks), 0)
        self.assertTrue(chunks[-1].get("done"))
        self.assertIn("finish_reason", chunks[-1])

    def test_06_cancellation(self):
        """Stream generation terminates immediately upon cancellation callback."""
        called = False
        def cancel_cb():
            nonlocal called
            called = True
            return True

        chunks = list(self.engine.stream_generate("Cancellation test", {"max_tokens": 50, "is_cancelled": cancel_cb}))
        self.assertTrue(called)
        self.assertTrue(chunks[-1].get("done"))
        self.assertEqual(chunks[-1].get("finish_reason"), "cancelled")

    def test_07_thread_safe_concurrent_generation(self):
        """Concurrent inference calls execute safely without crashing or corrupting state."""
        def run_inference(i):
            return self.engine.generate_response(f"Concurrent prompt {i}", {"max_tokens": 10, "seed": i})

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_inference, i) for i in range(4)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 4)
        for text, meta in results:
            self.assertEqual(meta["lifecycle_state"], "COMPLETED")
            self.assertGreater(meta["tokens_generated"], 0)

    def test_08_serving_health_check(self):
        """Health check returns READY with authentic model information."""
        health = get_health_status(self.engine)
        self.assertEqual(health["status"], "READY")
        self.assertTrue(health["loaded"])
        self.assertTrue(health["has_trained_weights"])
        self.assertEqual(health["vocab_size"], 1024)

    def test_09_schemas_serialization(self):
        """GenerateRequest and GenerateResponse serialize and deserialize accurately."""
        req_data = {
            "prompt": "Test prompt",
            "temperature": 0.5,
            "max_tokens": 64,
            "context": {"system_prompt": "Sys"}
        }
        req = GenerateRequest.from_dict(req_data)
        self.assertEqual(req.prompt, "Test prompt")
        self.assertEqual(req.temperature, 0.5)
        self.assertEqual(req.max_tokens, 64)

        resp = GenerateResponse(
            id="gen_123",
            object="text_completion",
            content="Hello world",
            confidence="HIGH_CONFIDENCE",
            evidence_used=True,
            has_trained_weights=True,
            model="aether-v2-scaled",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )
        resp_dict = resp.to_dict()
        self.assertEqual(resp_dict["content"], "Hello world")
        self.assertEqual(resp_dict["usage"]["total_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
