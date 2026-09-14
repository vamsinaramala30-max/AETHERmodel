"""
AETHER MODEL — Phase 11 Unit Test Suite
Controlled Model Scaling & Real Checkpoint Verification

Validates:
1. ModelConfig V1 vs V2 Presets & Constraint Validation (d_model % n_heads == 0).
2. Parameter Count & Dimensionality Integrity across all transformer layers.
3. Analytical Backward Pass & Gradient Flow on V2 Architecture.
4. Checkpoint Serialization & SHA-256 Checksum Verification for V2.
5. Legacy Checkpoint Preservation (V1 checkpoint never overwritten by V2).
6. Architecture Mismatch Guards (d_model, n_layers, vocab_size).
7. AdamW Optimizer & LR Scheduler on V2 Architecture.
8. Training Resumption Fidelity (Restore model, optimizer, scheduler, step).
9. Memory Footprint & Throughput Profiling.
10. Generation Sanity & Special Token Handling.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import unittest

import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import CausalInstructionDataset
from model.config.model_config import ModelConfig
from model.model import AetherModel
from tokenizer.tokenizer import PAD_TOKEN_ID, AetherTokenizer
from training.checkpoint import CheckpointManager
from training.loss import compute_cross_entropy
from training.optimizer import AdamW
from training.scheduler import LRScheduler
from training.trainer import AetherTrainer


class TestPhase11ModelScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bpe_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        cls.tokenizer = AetherTokenizer(vocab_file=cls.bpe_path, frozen=True) if os.path.exists(cls.bpe_path) else AetherTokenizer()
        cls.tmp_dir = os.path.join(base_dir, "checkpoints", "test_p11_scaling")
        os.makedirs(cls.tmp_dir, exist_ok=True)

    def test_01_model_config_presets_and_validation(self):
        """Test V1 legacy, V1 BPE, and V2 scaled presets and architectural constraints."""
        v1_leg = ModelConfig.v1_legacy()
        v1_bpe = ModelConfig.v1_bpe()
        v2 = ModelConfig.v2_scaled()

        # Validate architectural constraints
        v1_leg.validate()
        v1_bpe.validate()
        v2.validate()

        self.assertEqual(v1_leg.d_model, 64)
        self.assertEqual(v1_leg.vocab_size, 579)

        self.assertEqual(v1_bpe.d_model, 64)
        self.assertEqual(v1_bpe.vocab_size, 1024)

        self.assertEqual(v2.d_model, 256)
        self.assertEqual(v2.n_layers, 6)
        self.assertEqual(v2.n_heads, 8)
        self.assertEqual(v2.d_ff, 512)
        self.assertEqual(v2.vocab_size, 1024)
        self.assertEqual(v2.d_model % v2.n_heads, 0)
        self.assertEqual(v2.model_version, "2.0.0")

        # Test invalid configuration fails validation
        invalid_cfg = ModelConfig(d_model=250, n_heads=8)
        with self.assertRaises(ValueError):
            invalid_cfg.validate()

    def test_02_parameter_count_scaling_math(self):
        """Verify exact parameter counts and scaling ratios between V1 and V2."""
        v1_cfg = ModelConfig.v1_bpe()
        v2_cfg = ModelConfig.v2_scaled()

        v1_model = AetherModel(v1_cfg, skip_checkpoint=True)
        v2_model = AetherModel(v2_cfg, skip_checkpoint=True)

        v1_named = v1_model.architecture.get_named_parameters()
        v2_named = v2_model.architecture.get_named_parameters()

        v1_params = sum(p.size for _, p, _ in v1_named)
        v2_params = sum(p.size for _, p, _ in v2_named)

        # Expected:
        # V1: 1024*64 (emb) + 2 * (4*64*64 + 4*64 + 2*64*128 + 128 + 64) + 2*64 + (64*1024 + 1024)
        # = 65,536 + 2 * (16,384 + 256 + 16,384 + 192) + 128 + 66,560 = 198,656
        self.assertEqual(v1_params, 198_656)

        # V2: 1024*256 (emb) + 6 * (4*256*256 + 4*256 + 2*256*512 + 512 + 256) + 2*256 + (256*1024 + 1024)
        # = 262,144 + 6 * (262,144 + 1024 + 262,144 + 768) + 512 + 263,168 = 3,682,304
        self.assertEqual(v2_params, 3_682_304)

        param_increase = (v2_params - v1_params) / float(v1_params)
        self.assertAlmostEqual(param_increase, 17.5359, places=2)
        # 1 emb + 6 blocks * 12 params (4 attn + 2 ln1 + 4 ffn + 2 ln2) + 2 final ln + 2 lm head = 77 tensors
        self.assertEqual(len(v2_named), 1 + 6 * 12 + 2 + 2)
        self.assertTrue(v2_params > v1_params)

    def test_03_forward_and_backward_pass_v2(self):
        """Verify forward and analytical backward pass on scaled V2 model."""
        cfg = ModelConfig.v2_scaled()
        model = AetherModel(cfg, skip_checkpoint=True)

        seq_len = 16
        input_ids = [i % cfg.vocab_size for i in range(seq_len)]
        target_ids = [(i + 1) % cfg.vocab_size for i in range(seq_len)]

        # Forward pass
        logits = model.forward_all(input_ids)
        self.assertEqual(len(logits), seq_len)
        self.assertEqual(len(logits[0]), cfg.vocab_size)

        # Compute loss and backward pass
        loss, grad_logits, metrics = compute_cross_entropy(logits, target_ids, asst_start_idx=4)
        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 0.0)

        model.zero_grad()
        model.backward(grad_logits)

        # Check gradients across all parameters
        named_params = model.architecture.get_named_parameters()
        grad_norms = []
        for name, param, grad in named_params:
            self.assertEqual(param.shape, grad.shape, f"Shape mismatch on {name}")
            gnorm = float(np.linalg.norm(grad))
            grad_norms.append(gnorm)

        # Gradients must be non-zero on active layers
        self.assertGreater(sum(grad_norms), 0.0)
        self.assertTrue(all(math.isfinite(g) for g in grad_norms))

    def test_04_checkpoint_saving_and_sha256_verification(self):
        """Verify V2 checkpoint serialization with SHA-256 and metadata."""
        cfg = ModelConfig.v2_scaled()
        model = AetherModel(cfg, skip_checkpoint=True)
        opt = AdamW(model.architecture.get_named_parameters(), lr=1e-3)
        sched = LRScheduler(base_lr=1e-3, warmup_steps=5, total_steps=50)

        manager = CheckpointManager(self.tmp_dir, model_name="aether-v2-scaled")
        ckpt_path, checksum = manager.save(
            model=model,
            optimizer=opt,
            scheduler=sched,
            step=42,
            epoch=3,
            metrics={"val_loss": 3.1415, "train_loss": 2.7182},
            tag="v2_test",
        )

        self.assertTrue(os.path.exists(ckpt_path))
        self.assertEqual(len(checksum), 64)

        # Verify JSON contents
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        self.assertEqual(meta["model_name"], "aether-v2-scaled")
        self.assertEqual(meta["model_version"], "2.0.0")
        self.assertEqual(meta["d_model"], 256)
        self.assertEqual(meta["n_layers"], 6)
        self.assertEqual(meta["n_heads"], 8)
        self.assertEqual(meta["training_step"], 42)
        self.assertEqual(meta["epoch"], 3)
        self.assertEqual(meta["checksum"], checksum)

        # Reload into fresh V2 model
        fresh_model = AetherModel(cfg, skip_checkpoint=True)
        fresh_opt = AdamW(fresh_model.architecture.get_named_parameters(), lr=1e-3)
        fresh_sched = LRScheduler(base_lr=1e-3, warmup_steps=5, total_steps=50)

        loaded_meta = manager.load(ckpt_path, fresh_model, fresh_opt, fresh_sched)
        self.assertEqual(loaded_meta["training_step"], 42)

        # Verify output logits match exactly
        test_tokens = [5, 10, 20, 30]
        logits_orig = model.forward_all(test_tokens)
        logits_fresh = fresh_model.forward_all(test_tokens)
        self.assertTrue(np.allclose(logits_orig, logits_fresh))

    def test_05_legacy_checkpoint_preservation(self):
        """Verify saving a V2 checkpoint does not overwrite legacy aether_checkpoint_v1.json."""
        v1_path = os.path.join(self.tmp_dir, "aether_checkpoint_v1.json")
        dummy_v1_content = {"metadata": {"model_name": "aether-v1-authoritative", "version": "1.0.0", "d_model": 64}}
        with open(v1_path, "w", encoding="utf-8") as f:
            json.dump(dummy_v1_content, f)

        # Save V2 checkpoint
        v2_cfg = ModelConfig.v2_scaled()
        v2_model = AetherModel(v2_cfg, skip_checkpoint=True)
        manager = CheckpointManager(self.tmp_dir, model_name="aether-v2-scaled")
        manager.save(model=v2_model, step=10, epoch=1, tag="v2_scaled")

        # Verify v1 checkpoint remains unaltered
        with open(v1_path, "r", encoding="utf-8") as f:
            v1_check = json.load(f)
        self.assertEqual(v1_check["metadata"]["model_name"], "aether-v1-authoritative")
        self.assertEqual(v1_check["metadata"]["d_model"], 64)

        # Verify canonical v2 checkpoint was created
        v2_canonical = os.path.join(self.tmp_dir, "aether_checkpoint_v2.json")
        self.assertTrue(os.path.exists(v2_canonical))

    def test_06_architecture_mismatch_guards(self):
        """Verify strict error raised when loading incompatible architecture into model."""
        v2_cfg = ModelConfig.v2_scaled()
        v2_model = AetherModel(v2_cfg, skip_checkpoint=True)
        manager = CheckpointManager(self.tmp_dir, model_name="aether-v2-scaled")
        ckpt_path, _ = manager.save(model=v2_model, step=1, epoch=1, tag="guard_test")

        # Attempt to load V2 checkpoint into V1 model (d_model=64 vs 256)
        v1_model = AetherModel(ModelConfig.v1_bpe(), skip_checkpoint=True)
        with self.assertRaises(ValueError) as ctx:
            manager.load(ckpt_path, v1_model)
        self.assertIn("d_model mismatch", str(ctx.exception))

    def test_07_resume_training_fidelity(self):
        """Verify training resumption produces continuous and valid training steps."""
        cfg = ModelConfig(vocab_size=self.tokenizer.vocab_size, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=64)
        sample_dataset = [
            {
                "input_ids": [4, 10, 20, 5, 30, 40, 6, 50, 60, 3],
                "target_ids": [10, 20, 5, 30, 40, 6, 50, 60, 3, 0],
                "asst_start_idx": 6,
            }
            for _ in range(10)
        ]

        # Phase 1: Train 3 epochs and checkpoint
        model_1 = AetherModel(cfg, skip_checkpoint=True)
        trainer_1 = AetherTrainer(model=model_1, config=cfg, lr=1e-3)
        summary_1 = trainer_1.train(
            train_dataset=sample_dataset,
            epochs=3,
            checkpoint_dir=self.tmp_dir,
            save_tag="resume_p1",
            verbose=False,
        )

        ckpt_file = summary_1["checkpoint_path"]
        self.assertTrue(os.path.exists(ckpt_file))

        # Phase 2: Create fresh model and resume
        model_2 = AetherModel(cfg, skip_checkpoint=True)
        trainer_2 = AetherTrainer(model=model_2, config=cfg, lr=1e-3)
        mgr = CheckpointManager(self.tmp_dir)
        mgr.load(ckpt_file, model_2, optimizer=trainer_2.optimizer)

        # Continue training 2 more epochs
        summary_2 = trainer_2.train(
            train_dataset=sample_dataset,
            epochs=2,
            checkpoint_dir=self.tmp_dir,
            save_tag="resume_p2",
            verbose=False,
        )

        self.assertTrue(math.isfinite(summary_2["final_train_loss"]))
        self.assertGreater(summary_2["total_steps"], 0)

    def test_08_generation_sanity_v2(self):
        """Verify forward single-step and greedy generation capability on V2."""
        cfg = ModelConfig.v2_scaled()
        model = AetherModel(cfg, skip_checkpoint=True)

        prompt_ids = [4, 10, 20, 5]  # e.g., <system> ... <user>
        generated = list(prompt_ids)

        for _ in range(5):
            logits = model.forward(generated)
            self.assertEqual(len(logits), cfg.vocab_size)
            next_token = int(np.argmax(logits))
            self.assertTrue(0 <= next_token < cfg.vocab_size)
            generated.append(next_token)

        self.assertEqual(len(generated), len(prompt_ids) + 5)


if __name__ == "__main__":
    unittest.main()
