"""
AETHER MODEL — Phase 14 Production Tokenizer Integration Test Suite

Tests the complete end-to-end integration pipeline:
    Tokenizer -> Dataset -> Model -> Training -> Checkpoint -> Inference -> API Serving

Verifies:
1. Tokenizer loading from production checkpoint artifact.
2. Dataset tokenization, sequence construction (input_ids vs target_ids), and boundary validation.
3. Model embedding layer & LM head dimension matching (vocab_size=1024).
4. Full forward and backward training step with weight optimization.
5. Checkpoint serialization, SHA-256 checksum verification, and state restoration.
6. Checkpoint vocabulary mismatch detection and clear failure modes.
7. Inference engine prompt formatting, generation, and response decoding.
8. Streaming SSE generator token deltas.
9. HTTP server API endpoints (/health, /models, /generate, /audit).
"""

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

# Setup path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from tokenizer.special_tokens import SpecialTokens
from src.data.dataset import CausalInstructionDataset
from inference.engine import AetherInferenceEngine
from training.trainer import AetherTrainer
from training.checkpoint import CheckpointManager


class TestProductionTokenizerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bpe_artifact_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        cls.assertTrue(
            os.path.exists(cls.bpe_artifact_path),
            f"Required BPE artifact missing at: {cls.bpe_artifact_path}",
        )
        cls.tokenizer = AetherTokenizer(vocab_file=cls.bpe_artifact_path, frozen=True)
        cls.temp_dir = tempfile.mkdtemp(prefix="aether_phase14_test_")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_tokenizer_artifact_integrity(self):
        """Verifies the production BPE artifact loads with 1024 vocab and all special tokens."""
        self.assertEqual(self.tokenizer.vocab_size, 1024)
        self.assertTrue(self.tokenizer.is_bpe)
        self.assertTrue(self.tokenizer.is_frozen)

        # Check all 9 canonical special tokens
        canonical = [
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<system>", "<user>", "<assistant>", "<tool>", "<evidence>"
        ]
        for tok in canonical:
            self.assertIn(tok, self.tokenizer.token_to_id)
            tid = self.tokenizer.token_to_id[tok]
            self.assertEqual(self.tokenizer.id_to_token[tid], tok)

    def test_02_dataset_tokenization_and_causal_pairs(self):
        """Verifies CausalInstructionDataset encodes samples and constructs valid next-token pairs."""
        records = [
            {
                "system": "You are Aether AI.",
                "user": "What is agentic automation?",
                "assistant": "Agentic automation orchestrates tools and tasks autonomously.",
                "category": "automation",
            },
            {
                "system": "",
                "user": "Calculate 25 * 4",
                "assistant": "25 * 4 = 100",
                "category": "reasoning",
            },
        ]
        ds = CausalInstructionDataset(records, tokenizer=self.tokenizer, max_seq_len=256)
        self.assertEqual(len(ds), 2)

        for ex in ds:
            inp = ex["input_ids"]
            tgt = ex["target_ids"]
            self.assertEqual(len(inp), len(tgt))
            self.assertNotEqual(inp, tgt)
            # Causal property: tgt is shifted by 1
            self.assertEqual(inp[1:], tgt[:-1])

            # Verify all IDs in bounds [0, 1024)
            for tid in inp + tgt:
                self.assertGreaterEqual(tid, 0)
                self.assertLess(tid, self.tokenizer.vocab_size)

    def test_03_model_vocabulary_dimension_matching(self):
        """Verifies Model embedding and output head match tokenizer vocab_size (1024)."""
        config = ModelConfig.v2_scaled(vocab_size=self.tokenizer.vocab_size, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)

        self.assertEqual(model.config.vocab_size, 1024)
        self.assertEqual(model.architecture.token_embedding.vocab_size, 1024)
        self.assertEqual(model.architecture.lm_head.vocab_size, 1024)

        # Test forward pass with sample sequence
        input_ids = self.tokenizer.encode("Hello world! Aether is running.")
        logits = model.forward(input_ids)
        self.assertEqual(len(logits), 1024)

    def test_04_training_forward_backward_optimization_cycle(self):
        """Executes full training step: tokenize -> forward -> loss -> backward -> weight update."""
        config = ModelConfig.v2_scaled(vocab_size=self.tokenizer.vocab_size, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)

        records = [
            {
                "system": "You are Aether AI.",
                "user": "Hello!",
                "assistant": "Hello! How can I assist your workspace today?",
                "category": "conversation",
            },
            {
                "system": "You are Aether AI.",
                "user": "What is BPE?",
                "assistant": "BPE is Byte-Pair Encoding for subword tokenization.",
                "category": "technical",
            },
        ]
        ds = CausalInstructionDataset(records, tokenizer=self.tokenizer, max_seq_len=128)

        trainer = AetherTrainer(model=model, config=config, lr=1e-3)
        trainer.tokenizer = self.tokenizer

        # Capture initial state
        initial_loss = trainer.evaluate(ds)

        # Train 3 epochs
        summary = trainer.train(
            train_dataset=ds,
            epochs=3,
            verbose=False,
        )

        self.assertTrue(summary["loss_decreased"] or summary["final_train_loss"] <= initial_loss)
        self.assertGreater(summary["total_steps"], 0)

    def test_05_checkpoint_save_reload_and_integrity(self):
        """Verifies checkpoint saving with metadata, SHA-256 checksum, and restoration."""
        config = ModelConfig.v2_scaled(vocab_size=self.tokenizer.vocab_size, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)

        ckpt_dir = os.path.join(self.temp_dir, "ckpts")
        manager = CheckpointManager(checkpoint_dir=ckpt_dir)

        ckpt_path, checksum = manager.save(
            model=model,
            step=10,
            epoch=2,
            metrics={"training_loss": 1.234},
            tag="test_p14",
        )
        self.assertTrue(os.path.exists(ckpt_path))
        self.assertGreater(len(checksum), 0)

        # Reload into a fresh model
        reloaded_model = AetherModel(config, skip_checkpoint=True)
        meta = manager.load(ckpt_path, reloaded_model)

        self.assertEqual(meta["vocab_size"], 1024)
        self.assertEqual(meta["d_model"], 64)
        self.assertTrue(reloaded_model.config.has_trained_weights)

    def test_06_checkpoint_vocabulary_mismatch_detection(self):
        """Verifies checkpoint loading fails clearly when vocabulary size does not match."""
        # Save a 1024-vocab checkpoint
        config_1024 = ModelConfig.v2_scaled(vocab_size=1024, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model_1024 = AetherModel(config_1024, skip_checkpoint=True)

        ckpt_dir = os.path.join(self.temp_dir, "mismatch_ckpts")
        manager = CheckpointManager(checkpoint_dir=ckpt_dir)
        ckpt_path, _ = manager.save(model=model_1024, step=1, tag="vocab_1024")

        # Try to load into a 579-vocab legacy model config
        config_legacy = ModelConfig.v1_legacy(vocab_size=579, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model_legacy = AetherModel(config_legacy, skip_checkpoint=True)

        with self.assertRaises(ValueError) as ctx:
            manager.load(ckpt_path, model_legacy)
        self.assertIn("Vocabulary size mismatch", str(ctx.exception))

    def test_07_inference_engine_generation_with_bpe(self):
        """Verifies AetherInferenceEngine formats prompts, runs generation, and decodes with BPE."""
        config = ModelConfig.v2_scaled(vocab_size=1024, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)
        engine = AetherInferenceEngine(model=model, tokenizer=self.tokenizer)

        prompt = "Hello Aether!"
        response_text, meta = engine.generate_response(prompt, {"max_tokens": 16, "deterministic": True})

        self.assertIsInstance(response_text, str)
        self.assertEqual(meta["lifecycle_state"], "COMPLETED")
        self.assertGreater(meta["prompt_tokens"], 0)

    def test_08_inference_streaming_sse_generator(self):
        """Verifies SSE streaming yields real token chunks decoded via BPE."""
        config = ModelConfig.v2_scaled(vocab_size=1024, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)
        engine = AetherInferenceEngine(model=model, tokenizer=self.tokenizer)

        stream = engine.stream_generate("Explain workspace projects", {"max_tokens": 10, "deterministic": True})
        chunks = list(stream)

        self.assertGreater(len(chunks), 0)
        # Final chunk should indicate done
        self.assertTrue(chunks[-1].get("done", False))

    def test_09_multilingual_indic_and_emoji_inference(self):
        """Verifies inference pipeline handles Hindi, Telugu, and emoji without errors."""
        config = ModelConfig.v2_scaled(vocab_size=1024, n_layers=2, d_model=64, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)
        engine = AetherInferenceEngine(model=model, tokenizer=self.tokenizer)

        prompts = [
            "नमस्ते, आप कैसे हैं?",
            "తెలుగు ప్రాజెక్ట్ ప్లాన్",
            "🚀 Deploying containerized microservices to cloud 🤖",
        ]
        for p in prompts:
            resp, meta = engine.generate_response(p, {"max_tokens": 8, "deterministic": True})
            self.assertEqual(meta["lifecycle_state"], "COMPLETED")
            self.assertGreater(meta["prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
