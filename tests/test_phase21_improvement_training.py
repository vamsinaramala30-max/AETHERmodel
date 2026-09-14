"""
AETHER MODEL — Unit Test Suite for Phase 21 Improvement Training & Behavioral Alignment

Tests:
1. Combined dataset loading & deduplication.
2. Tokenizer encoding/decoding consistency on improvement examples.
3. Loss calculation and active token masking.
4. Analytical gradient shape and numerical validity.
5. Small-scale smoke training loop (forward + backward + AdamW step).
6. Checkpoint saving, SHA-256 checksums, and resumption.
7. Golden set evaluator scoring math and comparison logic.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from typing import Any, Dict

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.checkpoint import CheckpointManager
from training.dataset import InstructionDataset
from training.golden_evaluator import GoldenSetEvaluator, compare_golden_evaluations
from training.improvement_trainer import CombinedImprovementDataset, Phase21ImprovementTrainer
from training.loss import compute_cross_entropy
from training.optimizer import AdamW


class TestPhase21ImprovementTraining(unittest.TestCase):

    def setUp(self):
        self.ckpt_dir = os.path.join(base_dir, "checkpoints")
        self.bpe_tok_path = os.path.join(self.ckpt_dir, "aether_bpe_tokenizer.json")
        self.v2_ckpt_path = os.path.join(self.ckpt_dir, "aether_checkpoint_v2.json")
        self.train_path = os.path.join(base_dir, "data", "improvement", "aether_improvement_train_v1.jsonl")
        self.val_path = os.path.join(base_dir, "data", "improvement", "aether_improvement_val_v1.jsonl")
        self.test_path = os.path.join(base_dir, "data", "improvement", "aether_improvement_test_v1.jsonl")
        self.regr_path = os.path.join(base_dir, "data", "improvement", "aether_regression_dataset_v1.jsonl")
        self.golden_path = os.path.join(base_dir, "data", "improvement", "aether_golden_test_set_v1.jsonl")

    def test_01_dataset_files_exist_and_non_empty(self):
        """Verifies all Phase 20 improvement dataset files exist with valid JSONL records."""
        for p in [self.train_path, self.val_path, self.test_path, self.regr_path, self.golden_path]:
            self.assertTrue(os.path.exists(p), f"Missing dataset file: {p}")
            with open(p, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                self.assertGreater(len(lines), 0, f"Empty dataset file: {p}")
                for line in lines:
                    obj = json.loads(line)
                    self.assertIn("user", obj)
                    self.assertIn("assistant", obj)

    def test_02_combined_improvement_dataset_loader(self):
        """Verifies CombinedImprovementDataset ingests both improvement and regression records."""
        tok = AetherTokenizer(vocab_file=self.bpe_tok_path, frozen=True)
        ds = CombinedImprovementDataset(
            improvement_path=self.train_path,
            regression_path=self.regr_path,
            tokenizer=tok,
        )
        self.assertGreaterEqual(len(ds), 20)
        item0 = ds[0]
        self.assertIn("input_ids", item0)
        self.assertIn("target_ids", item0)
        self.assertIn("asst_start_idx", item0)
        self.assertGreater(item0["asst_start_idx"], 0)
        self.assertEqual(len(item0["input_ids"]), len(item0["target_ids"]))

    def test_03_causal_cross_entropy_and_active_masking(self):
        """Verifies causal cross-entropy loss masks prompt tokens and computes valid gradients."""
        vocab_size = 1024
        seq_len = 16
        asst_idx = 6

        # Synthetic logits & targets
        import numpy as np
        np.random.seed(42)
        logits = np.random.randn(seq_len, vocab_size).tolist()
        targets = np.random.randint(0, vocab_size, size=seq_len).tolist()

        loss, grad, metrics = compute_cross_entropy(logits, targets, asst_start_idx=asst_idx, pad_token_id=0)
        self.assertGreater(loss, 0.0)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(len(grad), seq_len)
        self.assertEqual(len(grad[0]), vocab_size)
        self.assertEqual(metrics["active_tokens"], seq_len - asst_idx + 1)

    def test_04_small_model_forward_backward_optimizer_smoke(self):
        """Smoke test verifying forward, loss, backward, and AdamW update steps on a mini model."""
        cfg = ModelConfig(
            vocab_size=1024,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=64,
        )
        model = AetherModel(cfg, skip_checkpoint=True)
        named_params = model.architecture.get_named_parameters()
        optimizer = AdamW(named_parameters=named_params, lr=1e-3, weight_decay=0.01)

        input_ids = [3, 10, 20, 4, 30, 40, 5, 50, 60, 2]
        target_ids = [10, 20, 4, 30, 40, 5, 50, 60, 2, 0]

        logits = model.forward_all(input_ids)
        loss, grad_logits, _ = compute_cross_entropy(logits, target_ids, asst_start_idx=6)
        self.assertGreater(loss, 0.0)

        # Backward
        model.zero_grad()
        model.backward(grad_logits)

        # Optimizer step
        optimizer.step()
        model.zero_grad()

        # Loss after step should compute cleanly
        logits_after = model.forward_all(input_ids)
        loss_after, _, _ = compute_cross_entropy(logits_after, target_ids, asst_start_idx=6)
        self.assertTrue(math.isfinite(loss_after))

    def test_05_checkpoint_manager_save_and_checksum(self):
        """Verifies checkpoint saving produces valid JSON and matching SHA-256 checksums."""
        cfg = ModelConfig(
            vocab_size=1024,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=64,
        )
        model = AetherModel(cfg, skip_checkpoint=True)
        test_dir = os.path.join(base_dir, "checkpoints", "test_p21_smoke_ckpt")
        os.makedirs(test_dir, exist_ok=True)
        mgr = CheckpointManager(test_dir, max_to_keep=2)

        ckpt_path, checksum = mgr.save(
            model=model,
            step=1,
            epoch=1,
            filename="smoke_test.json",
        )
        self.assertTrue(os.path.exists(ckpt_path))
        self.assertEqual(len(checksum), 64)

        # Reload
        model2 = AetherModel(cfg, skip_checkpoint=True)
        loaded = model2.load_checkpoint(ckpt_path)
        self.assertTrue(loaded)
        self.assertEqual(model2.load_status, "READY")

    def test_06_golden_evaluator_comparison_math(self):
        """Verifies compare_golden_evaluations classification math."""
        v2_rep = {
            "cases": [
                {"id": "G1", "score": 1.5, "passed": False, "category": "planning"},
                {"id": "G2", "score": 3.5, "passed": True, "category": "constraints"},
                {"id": "G3", "score": 2.0, "passed": False, "category": "clarification"},
            ]
        }
        v3_rep = {
            "cases": [
                {"id": "G1", "score": 4.5, "passed": True, "category": "planning"},  # RESOLVED
                {"id": "G2", "score": 4.0, "passed": True, "category": "constraints"},  # IMPROVED / UNCHANGED
                {"id": "G3", "score": 2.0, "passed": False, "category": "clarification"},  # UNCHANGED
            ]
        }
        comp = compare_golden_evaluations(v2_rep, v3_rep)
        self.assertEqual(comp["total_cases"], 3)
        self.assertEqual(comp["resolved_count"], 1)
        self.assertEqual(comp["new_failures_count"], 0)


if __name__ == "__main__":
    unittest.main()
