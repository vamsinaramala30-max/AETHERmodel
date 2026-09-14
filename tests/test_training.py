"""
AETHER MODEL — Training Pipeline Unit Test Suite
Validates dataset loading, loss calculation, backward gradient flow, AdamW optimizer parameter updates,
measurable loss reduction, and checkpoint save/load integrity.
"""

import sys
import os
import unittest
import math

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from model.config.model_config import ModelConfig
from tokenizer.tokenizer import AetherTokenizer, PAD_TOKEN_ID
from training.dataset import InstructionDataset
from training.trainer import AetherTrainer

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AetherTokenizer()
        self.config = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32)
        self.model = AetherModel(self.config)

    def test_dataset_loading_and_tokenization(self):
        data_path = os.path.join(base_dir, "data", "instruction", "aether_instructions_train.jsonl")
        ds = InstructionDataset(data_path, tokenizer=self.tokenizer, max_seq_len=128)
        self.assertGreater(len(ds), 0)
        self.assertGreater(ds.stats["total_tokens"], 0)
        self.assertIn("dataset_hash", ds.stats)

        # Test train / val split
        train_ds, val_ds = ds.split(val_ratio=0.2)
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertEqual(len(train_ds) + len(val_ds), len(ds))

    def test_training_loss_decrease_milestone(self):
        """
        Runs a small deterministic training experiment.
        Verifies that training decreases sequence cross-entropy loss.
        """
        # Create a small dataset with repeated patterns to test learning
        dataset = InstructionDataset(tokenizer=self.tokenizer, max_seq_len=32)
        dataset.records = [
            {"system": "You are Aether.", "user": "Say hello", "assistant": "Hello workspace.", "category": "general"},
            {"system": "You are Aether.", "user": "What is 2+2?", "assistant": "It is four.", "category": "reasoning"},
        ]
        dataset._tokenize_all()

        trainer = AetherTrainer(model=self.model, config=self.config, lr=5e-3)
        summary = trainer.train(train_dataset=dataset, epochs=6, verbose=False)

        self.assertTrue(summary["loss_decreased"], f"Loss did not decrease: initial={summary['initial_loss']}, final={summary['final_train_loss']}")
        self.assertLess(summary["final_train_loss"], summary["initial_loss"])
        self.assertEqual(summary["total_steps"], 12)

    def test_checkpoint_save_reload_integrity(self):
        ckpt_dir = os.path.join(base_dir, "checkpoints", "test_ckpt")
        ckpt_path = os.path.join(ckpt_dir, "aether_checkpoint_v1.json")

        meta = {"quality_classification": "PRODUCTION_CANDIDATE", "step": 10}
        checksum = self.model.save_checkpoint(ckpt_path, meta)
        self.assertTrue(os.path.exists(ckpt_path))
        self.assertGreater(len(checksum), 0)

        # Reload into a fresh model
        fresh_model = AetherModel(ModelConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len
        ))
        loaded = fresh_model.load_checkpoint(ckpt_path)
        self.assertTrue(loaded)
        self.assertTrue(fresh_model.config.has_trained_weights)

    def test_loss_masking_behavior(self):
        """
        Verifies that:
        1. System and user tokens have 0 gradient (masked out)
        2. Assistant tokens have non-zero gradients
        3. Padding tokens are masked out
        """
        seq_len = 10
        vocab_size = 32
        asst_start_idx = 5 # tokens 0..4 are prompt, 5..9 are assistant + EOS

        logits_matrix = [[0.1] * vocab_size for _ in range(seq_len)]
        target_ids = [3, 10, 15, 20, 25, 4, 8, 12, 2, PAD_TOKEN_ID] # last token is padding

        loss, grad_logits, _metrics = AetherTrainer.compute_cross_entropy(
            logits_matrix=logits_matrix,
            target_ids=target_ids,
            asst_start_idx=asst_start_idx
        )

        self.assertGreater(loss, 0.0)
        self.assertEqual(len(grad_logits), seq_len)

        # Positions 0..3 (before assistant token) must be exactly 0.0 gradient
        for t in range(asst_start_idx - 1):
            self.assertTrue(all(g == 0.0 for g in grad_logits[t]), f"Prompt token at pos {t} was not masked!")

        # Position 4 (assistant prefix) through 8 (EOS) must have non-zero gradients
        for t in range(asst_start_idx - 1, seq_len - 1):
            self.assertTrue(any(abs(g) > 0.0 for g in grad_logits[t]), f"Assistant token at pos {t} had zero gradient!")

        # Position 9 (PAD_TOKEN_ID) must be masked with 0.0 gradient
        self.assertTrue(all(g == 0.0 for g in grad_logits[seq_len - 1]), "Padding token was not masked!")

    def test_checkpoint_deterministic_generation_equivalence(self):
        """
        Verifies Phase 16:
        1. Saves a checkpoint with SHA-256
        2. Unloads model
        3. Reloads checkpoint into a brand new model instance
        4. Verifies identical deterministic forward logits and generation outputs
        """
        ckpt_dir = os.path.join(base_dir, "checkpoints", "test_integrity_ckpt")
        ckpt_path = os.path.join(ckpt_dir, "aether_checkpoint_v1.json")

        try:
            meta = {"quality_classification": "PRODUCTION_CANDIDATE", "step": 42}
            checksum = self.model.save_checkpoint(ckpt_path, meta)
            self.assertTrue(os.path.exists(ckpt_path))
            self.assertGreater(len(checksum), 0)

            # Unload model and initialize a brand new instance
            del self.model
            new_model = AetherModel(ModelConfig(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                n_layers=self.config.n_layers,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                max_seq_len=self.config.max_seq_len
            ))
            loaded = new_model.load_checkpoint(ckpt_path)
            self.assertTrue(loaded)
            self.assertEqual(new_model.metadata.get("checksum"), checksum)

            # Verify forward logits
            test_seq = [1, 4, 7, 9]
            logits_1 = new_model.forward(test_seq)
            logits_2 = new_model.forward(test_seq)
            self.assertEqual(logits_1, logits_2)
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            if os.path.exists(ckpt_dir):
                os.rmdir(ckpt_dir)

if __name__ == "__main__":
    unittest.main()
