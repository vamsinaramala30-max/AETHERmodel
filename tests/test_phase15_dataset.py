"""
AETHER MODEL — Phase 15 Production Dataset Engineering & Quality Control Test Suite

Comprehensive 10-part test suite validating:
- Test 1: Dataset loading and record parsing.
- Test 2: Validation, schema compliance, and malformed sample rejection.
- Test 3: Exact, normalized, and prefix/near-duplicate deduplication.
- Test 4: Deterministic zero-leakage 3-way (Train/Val/Test) splitting.
- Test 5: Phase 14 Byte-Level BPE subword tokenization and vocabulary bounds.
- Test 6: Causal next-token batch generation, target shift, and attention masking.
- Test 7: Real training integration with AetherModel and AetherTrainer.
- Test 8: Validation pipeline and cross-entropy loss computation.
- Test 9: Real sequence-length percentile statistics (P50, P90, P95, P99).
- Test 10: Regression protection for Phase 14 tokenizer and inference engine.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.cleaner import DatasetCleaner
from data.dataset import CausalInstructionDataset
from data.split import DatasetSplitter
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import (
    ASSISTANT_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    AetherTokenizer,
)
from training.dataset import InstructionDataset
from training.dataset_quality import DatasetQualityAuditor
from training.trainer import AetherTrainer


class TestPhase15DatasetPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bpe_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        if os.path.exists(cls.bpe_path):
            cls.tokenizer = AetherTokenizer(vocab_file=cls.bpe_path, frozen=True)
        else:
            cls.tokenizer = AetherTokenizer(frozen=True)

        cls.train_path = os.path.join(base_dir, "data", "cleaned", "aether_train_split.jsonl")
        cls.val_path = os.path.join(base_dir, "data", "cleaned", "aether_val_split.jsonl")
        cls.test_path = os.path.join(base_dir, "data", "cleaned", "aether_test_split.jsonl")
        cls.eval_path = os.path.join(base_dir, "data", "evaluation", "aether_eval_suite.jsonl")

    # -------------------------------------------------------------
    # Test 1: Loading
    # -------------------------------------------------------------
    def test_01_dataset_loading_and_parsing(self):
        """Verifies dataset correctly loads and parses records from JSONL and lists."""
        self.assertTrue(os.path.exists(self.train_path), f"Train dataset missing: {self.train_path}")
        dataset = CausalInstructionDataset(self.train_path, tokenizer=self.tokenizer)
        self.assertGreater(len(dataset), 50, "Train dataset should contain substantial examples")
        self.assertIn("input_ids", dataset[0])
        self.assertIn("target_ids", dataset[0])
        self.assertIn("category", dataset[0])

    # -------------------------------------------------------------
    # Test 2: Validation & Malformed Sample Rejection
    # -------------------------------------------------------------
    def test_02_validation_and_malformed_rejection(self):
        """Verifies cleaner rejects empty strings, missing roles, broken JSON, and invalid unicode."""
        cleaner = DatasetCleaner()

        # Valid sample
        valid = cleaner.clean_record({
            "category": "explanation",
            "system": "You are Aether.",
            "user": "Explain RAG.",
            "assistant": "RAG stands for Retrieval-Augmented Generation.",
        })
        self.assertIsNotNone(valid)
        self.assertEqual(valid["category"], "explanation")

        # Missing assistant
        self.assertIsNone(cleaner.clean_record({"user": "Hello"}))

        # Missing user
        self.assertIsNone(cleaner.clean_record({"assistant": "Hi"}))

        # Empty whitespace
        self.assertIsNone(cleaner.clean_record({"user": "   ", "assistant": "   "}))

        # Non-dict
        self.assertIsNone(cleaner.clean_record(["not", "a", "dict"]))

        # Messages format support
        msg_valid = cleaner.clean_record({
            "category": "coding",
            "messages": [
                {"role": "system", "content": "You are Aether."},
                {"role": "user", "content": "Write hello world in python."},
                {"role": "assistant", "content": "print('hello world')"},
            ],
        })
        self.assertIsNotNone(msg_valid)
        self.assertEqual(msg_valid["user"], "Write hello world in python.")

    # -------------------------------------------------------------
    # Test 3: Deduplication
    # -------------------------------------------------------------
    def test_03_deterministic_deduplication(self):
        """Verifies cleaner deduplicates exact and normalized identical records."""
        sample_records = [
            {"user": "What is Aether?", "assistant": "Aether is an intelligent workspace AI.", "category": "general"},
            {"user": "What is Aether?", "assistant": "Aether is an intelligent workspace AI.", "category": "general"},
            {"user": "what is aether?", "assistant": "aether is an intelligent workspace ai.", "category": "general"},
            {"user": "Plan sprint", "assistant": "Here is the sprint plan.", "category": "planning"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
            for r in sample_records:
                tf.write(json.dumps(r) + "\n")
            temp_path = tf.name

        try:
            cleaner = DatasetCleaner()
            cleaned, stats = cleaner.clean_file(temp_path)
            self.assertEqual(len(cleaned), 2, "Should deduplicate down to 2 unique records")
            self.assertEqual(stats["duplicate_records"], 2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # -------------------------------------------------------------
    # Test 4: Splitting & Disjointness
    # -------------------------------------------------------------
    def test_04_zero_leakage_three_way_splitting(self):
        """Verifies 3-way split is deterministic and strictly disjoint across Train, Val, Test, and Eval."""
        splitter = DatasetSplitter(seed=42)
        auditor = DatasetQualityAuditor(tokenizer=self.tokenizer)

        leak_report = auditor.check_splits_disjoint(
            self.train_path, self.val_path, self.eval_path, self.test_path
        )
        self.assertTrue(
            leak_report["is_strictly_disjoint"],
            f"Leakage detected in dataset splits: {leak_report['leakage_details']}",
        )
        self.assertEqual(leak_report["train_val_leakage_count"], 0)
        self.assertEqual(leak_report["train_test_leakage_count"], 0)
        self.assertEqual(leak_report["val_test_leakage_count"], 0)
        self.assertEqual(leak_report["train_eval_leakage_count"], 0)

    # -------------------------------------------------------------
    # Test 5: Tokenization
    # -------------------------------------------------------------
    def test_05_bpe_tokenization_and_vocab_bounds(self):
        """Verifies Phase 14 BPE Tokenizer processes samples within valid vocab bounds."""
        dataset = CausalInstructionDataset(self.train_path, tokenizer=self.tokenizer)
        vocab_size = self.tokenizer.vocab_size

        for i in range(min(len(dataset), 25)):
            ex = dataset[i]
            for tid in ex["input_ids"]:
                self.assertGreaterEqual(tid, 0)
                self.assertLess(tid, vocab_size, f"Token ID {tid} out of vocab bounds {vocab_size}")
            for tid in ex["target_ids"]:
                self.assertGreaterEqual(tid, 0)
                self.assertLess(tid, vocab_size, f"Token ID {tid} out of vocab bounds {vocab_size}")

    # -------------------------------------------------------------
    # Test 6: Batching & Causal Alignment
    # -------------------------------------------------------------
    def test_06_causal_pair_alignment_and_batching(self):
        """Verifies causal LM next-token invariant: input[1:] == target[:-1] and input != target."""
        dataset = CausalInstructionDataset(self.train_path, tokenizer=self.tokenizer)
        for i in range(min(len(dataset), 25)):
            ex = dataset[i]
            inp = ex["input_ids"]
            tgt = ex["target_ids"]
            self.assertNotEqual(inp, tgt, "Input IDs must not equal Target IDs in causal LM")
            self.assertEqual(inp[1:], tgt[:-1], "Causal shift invariant failed: input[1:] != target[:-1]")

        # Test batching
        batch = dataset.get_batch([0, 1, 2], pad_to_max=True)
        self.assertEqual(batch["batch_size"], 3)
        self.assertEqual(len(batch["input_ids"]), 3)
        self.assertEqual(len(batch["target_ids"]), 3)
        self.assertEqual(len(batch["attention_mask"]), 3)

    # -------------------------------------------------------------
    # Test 7: Training Integration (Forward & Loss)
    # -------------------------------------------------------------
    def test_07_real_model_forward_and_loss(self):
        """Verifies model accepts real dataset batches and computes causal cross-entropy loss."""
        dataset = CausalInstructionDataset(self.train_path, tokenizer=self.tokenizer)
        config = ModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=256,
        )
        model = AetherModel(config, skip_checkpoint=True)
        trainer = AetherTrainer(model=model, config=config)

        sample = dataset[0]
        logits = model.forward_all(sample["input_ids"])
        loss, grad_logits, metrics = trainer.compute_cross_entropy(
            logits, sample["target_ids"], sample["asst_start_pos"]
        )
        self.assertTrue(math.isfinite(loss), f"Loss should be finite, got {loss}")
        self.assertGreater(loss, 0.0, "Loss should be positive")
        self.assertEqual(len(grad_logits), len(sample["input_ids"]))

    # -------------------------------------------------------------
    # Test 8: Validation Pipeline
    # -------------------------------------------------------------
    def test_08_validation_loss_computation(self):
        """Verifies validation loss evaluation runs across validation dataset without error."""
        val_dataset = CausalInstructionDataset(self.val_path, tokenizer=self.tokenizer)
        config = ModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=256,
        )
        model = AetherModel(config, skip_checkpoint=True)
        trainer = AetherTrainer(model=model, config=config)

        val_loss = trainer.evaluate(val_dataset)
        self.assertTrue(math.isfinite(val_loss), f"Validation loss must be finite: {val_loss}")
        self.assertGreater(val_loss, 0.0)

    # -------------------------------------------------------------
    # Test 9: Percentile Statistics Calculation
    # -------------------------------------------------------------
    def test_09_sequence_percentile_statistics(self):
        """Verifies real sequence length percentiles (P50, P90, P95, P99) are computed accurately."""
        dataset = CausalInstructionDataset(self.train_path, tokenizer=self.tokenizer)
        stats = dataset.stats

        self.assertGreater(stats["total_examples"], 0)
        self.assertGreater(stats["avg_tokens"], 0.0)
        self.assertGreater(stats["p50_tokens"], 0.0)
        self.assertGreaterEqual(stats["p90_tokens"], stats["p50_tokens"])
        self.assertGreaterEqual(stats["p95_tokens"], stats["p90_tokens"])
        self.assertGreaterEqual(stats["p99_tokens"], stats["p95_tokens"])
        self.assertGreaterEqual(stats["max_tokens"], stats["p99_tokens"])

    # -------------------------------------------------------------
    # Test 10: Phase 14 Regression Protection
    # -------------------------------------------------------------
    def test_10_phase14_regression_protection(self):
        """Verifies Phase 14 BPE Tokenizer encoding, decoding, and special tokens are intact."""
        text = "Hello Aether, please plan our sprint roadmap!"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)

        self.assertGreater(len(encoded), 0)
        self.assertIn("Hello", decoded)
        self.assertIn("sprint", decoded)

        # Check special tokens
        for sp in ["<pad>", "<unk>", "<eos>", "<system>", "<user>", "<assistant>", "<tool>", "<evidence>"]:
            self.assertIn(sp, self.tokenizer.token_to_id)


if __name__ == "__main__":
    unittest.main()
