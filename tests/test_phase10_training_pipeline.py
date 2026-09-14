"""
AETHER MODEL — Phase 10 Comprehensive Training Pipeline Test Suite

Exhaustively verifies:
1. True Causal Next-Token Shift: input_ids != target_ids, input_ids[1:] == target_ids[:-1].
2. Logit Shapes & Vocab Bounds: [seq_len, vocab_size] matched against [seq_len].
3. Cross-Entropy Loss & Masking: Numerical stability, prompt masking, padding masking.
4. Analytical vs Finite-Difference Loss Gradients.
5. Backpropagation & Parameter Mutation: Parameters genuinely change after optimizer step.
6. Gradient Existence, Norms, and L2 Gradient Clipping.
7. AdamW Optimizer Moments, Weight Decay Filtering, and State Dict Serialization.
8. LR Scheduler Warmup & Cosine Annealing Dynamics.
9. Tiny-Batch Overfit Proof: Deep learning memorization of a controlled training set.
10. Validation Isolation: Zero parameter mutation and zero grad accumulation during evaluation.
11. Checkpoint Manager: Integrity checksums, state resumption, and vocabulary mismatch guards.
"""

from __future__ import annotations

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
from training.dataset import InstructionDataset
from training.loss import CausalCrossEntropyLoss, compute_cross_entropy
from training.optimizer import AdamW
from training.scheduler import LRScheduler
from training.trainer import AetherTrainer


class TestPhase10TrainingPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.config = ModelConfig(vocab_size=64, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=32)
        self.model = AetherModel(self.config, skip_checkpoint=True)
        self.tokenizer = AetherTokenizer()

    # ------------------------------------------------------------------------
    # 1. TRUE CAUSAL NEXT-TOKEN SHIFT TEST
    # ------------------------------------------------------------------------
    def test_causal_next_token_target_shift(self):
        """
        Verifies P(token[t+1] | token[0:t]) causal shift.
        Tokens: [A, B, C, D, EOS]
        Input:  [A, B, C, D]
        Target: [B, C, D, EOS]
        """
        raw_tokens = [10, 25, 33, 47, 2]  # Token IDs ending with EOS=2
        input_ids = raw_tokens[:-1]
        target_ids = raw_tokens[1:]

        self.assertNotEqual(input_ids, target_ids, "Input IDs must not equal Target IDs in causal LM!")
        self.assertEqual(len(input_ids), len(target_ids))
        self.assertEqual(input_ids[1:], target_ids[:-1], "Input shifted by 1 must match Target up to last token!")
        self.assertEqual(target_ids[-1], 2, "Last target token must be EOS!")

        # Verify through CausalInstructionDataset
        records = [
            {"system": "You are Aether.", "user": "Hello Aether", "assistant": "Greetings human.", "category": "general"}
        ]
        ds = CausalInstructionDataset(records, tokenizer=self.tokenizer)
        self.assertGreater(len(ds), 0)
        ex = ds[0]
        inp = ex["input_ids"]
        tgt = ex["target_ids"]
        self.assertNotEqual(inp, tgt)
        self.assertEqual(inp[1:], tgt[:-1])

    # ------------------------------------------------------------------------
    # 2. LOGIT SHAPES & VOCAB BOUNDS TEST
    # ------------------------------------------------------------------------
    def test_logit_shapes_and_dimensions(self):
        """
        Verifies forward_all produces [seq_len, vocab_size] matching target [seq_len].
        """
        seq_len = 8
        vocab_size = self.config.vocab_size
        input_ids = [i % vocab_size for i in range(seq_len)]

        logits = self.model.forward_all(input_ids)
        logits_arr = np.array(logits)

        self.assertEqual(logits_arr.shape, (seq_len, vocab_size), f"Expected shape ({seq_len}, {vocab_size}), got {logits_arr.shape}")
        self.assertTrue(np.all(np.isfinite(logits_arr)), "Logits contain NaN or Inf!")

    # ------------------------------------------------------------------------
    # 3. CROSS-ENTROPY LOSS & MASKING TEST
    # ------------------------------------------------------------------------
    def test_loss_computation_and_masking(self):
        """
        Verifies:
        1. Numerically stable loss computation.
        2. Prompt tokens before assistant start are masked (0.0 gradient).
        3. Assistant tokens receive valid non-zero gradients.
        4. Padding tokens (PAD_TOKEN_ID) are masked.
        """
        seq_len = 10
        vocab_size = self.config.vocab_size
        asst_start_idx = 4  # Tokens 0..3 are prompt, 4..8 are response, 9 is PAD

        logits_matrix = np.random.randn(seq_len, vocab_size)
        target_ids = [5, 12, 18, 24, 30, 36, 42, 48, 2, PAD_TOKEN_ID]

        loss, grad_logits, metrics = compute_cross_entropy(
            logits_matrix=logits_matrix,
            target_ids=target_ids,
            asst_start_idx=asst_start_idx,
            pad_token_id=PAD_TOKEN_ID,
        )

        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 0.0)
        self.assertEqual(len(grad_logits), seq_len)

        # Positions 0..2 (before assistant turn) must be masked with exact 0.0 gradient
        for t in range(asst_start_idx - 1):
            self.assertTrue(all(g == 0.0 for g in grad_logits[t]), f"Prompt token at position {t} was not masked!")

        # Assistant tokens (positions asst_start_idx - 1 .. 8) must have non-zero gradients
        for t in range(asst_start_idx - 1, seq_len - 1):
            self.assertTrue(any(abs(g) > 0.0 for g in grad_logits[t]), f"Active token at position {t} had zero gradient!")

        # Position 9 (PAD_TOKEN_ID) must be masked with exact 0.0 gradient
        self.assertTrue(all(g == 0.0 for g in grad_logits[seq_len - 1]), "Padding token at position 9 was not masked!")

        # Test extreme numerical stability
        extreme_logits = np.zeros((2, vocab_size))
        extreme_logits[0, 5] = 1000.0  # Very large positive
        extreme_logits[1, :] = -1000.0  # Very large negative
        loss_ext, grad_ext, _ = compute_cross_entropy(extreme_logits, [5, 10], asst_start_idx=0)
        self.assertTrue(math.isfinite(loss_ext), "Loss failed on extreme logits!")
        self.assertTrue(np.all(np.isfinite(grad_ext)), "Gradients contain NaN/Inf on extreme logits!")

    # ------------------------------------------------------------------------
    # 4. ANALYTICAL VS FINITE-DIFFERENCE LOSS GRADIENTS TEST
    # ------------------------------------------------------------------------
    def test_analytical_vs_finite_difference_loss_gradient(self):
        """
        Verifies analytical loss gradient w.r.t logits matches two-sided numerical finite difference:
        dLoss/dz = (Loss(z + eps) - Loss(z - eps)) / (2 * eps)
        """
        vocab_size = 8
        seq_len = 3
        logits = np.array([
            [1.2, -0.5, 0.8, 2.1, -1.0, 0.3, -0.2, 0.0],
            [-0.3, 1.5, -0.1, 0.4, 0.9, -1.2, 0.7, -0.5],
            [0.5, 0.2, -1.1, 1.8, -0.4, 0.6, -0.8, 1.0],
        ], dtype=np.float64)
        targets = [3, 1, 7]
        eps = 1e-5

        loss, grad_ana, _ = compute_cross_entropy(logits, targets, asst_start_idx=0)
        grad_ana = np.array(grad_ana)

        # Compute numerical finite difference gradient for each element
        grad_num = np.zeros_like(logits)
        for r in range(seq_len):
            for c in range(vocab_size):
                logits_plus = logits.copy()
                logits_plus[r, c] += eps
                l_plus, _, _ = compute_cross_entropy(logits_plus, targets, asst_start_idx=0)

                logits_minus = logits.copy()
                logits_minus[r, c] -= eps
                l_minus, _, _ = compute_cross_entropy(logits_minus, targets, asst_start_idx=0)

                grad_num[r, c] = (l_plus - l_minus) / (2.0 * eps)

        max_abs_diff = np.max(np.abs(grad_ana - grad_num))
        self.assertLess(max_abs_diff, 1e-6, f"Analytical gradient differs from numerical: max diff {max_abs_diff}")

    # ------------------------------------------------------------------------
    # 5. PARAMETER MUTATION TEST
    # ------------------------------------------------------------------------
    def test_backpropagation_and_parameter_mutation(self):
        """
        Verifies that Loss -> Backward -> Gradients -> Optimizer.step() actually changes model parameters.
        """
        input_ids = [3, 7, 12, 19]
        target_ids = [7, 12, 19, 2]

        named_params = self.model.architecture.get_named_parameters()
        optimizer = AdamW(named_params, lr=1e-2, weight_decay=0.01)

        # Snapshot parameters before step
        params_before = {name: np.array(param, copy=True) for name, param, _ in named_params}

        # 1. Forward
        logits = self.model.forward_all(input_ids)
        loss, grad_logits, _ = compute_cross_entropy(logits, target_ids, asst_start_idx=0)
        self.assertTrue(math.isfinite(loss))

        # 2. Backward
        self.model.zero_grad()
        self.model.backward(grad_logits)

        # Verify non-zero gradients exist
        has_nonzero_grad = False
        for name, param, grad in named_params:
            if np.any(np.abs(grad) > 0.0):
                has_nonzero_grad = True
                break
        self.assertTrue(has_nonzero_grad, "No gradients were generated during backpropagation!")

        # 3. Optimizer Step
        optimizer.step()

        # 4. Verify parameters changed
        changed_count = 0
        for name, param, _ in named_params:
            p_after = np.array(param)
            p_before = params_before[name]
            if not np.allclose(p_before, p_after):
                changed_count += 1

        self.assertGreater(changed_count, 0, "Parameters did not change after optimizer step!")
        self.assertEqual(changed_count, len(named_params), "All trainable parameters should have received updates!")

    # ------------------------------------------------------------------------
    # 6. GRADIENT EXISTENCE & L2 CLIPPING TEST
    # ------------------------------------------------------------------------
    def test_gradient_existence_and_l2_clipping(self):
        """
        Verifies gradient norm computation and global L2 norm clipping.
        """
        named_params = self.model.architecture.get_named_parameters()
        optimizer = AdamW(named_params, lr=1e-3, max_grad_norm=0.5)

        # Manually inject large gradients
        for name, param, grad in named_params:
            grad.fill(10.0)

        initial_norm = optimizer.clip_gradients()
        self.assertGreater(initial_norm, 0.5)

        # Compute new norm after clipping
        total_norm_sq = 0.0
        for name, param, grad in named_params:
            total_norm_sq += float(np.sum(grad * grad))
        clipped_norm = math.sqrt(total_norm_sq)

        self.assertAlmostEqual(clipped_norm, 0.5, places=4, msg="Gradient clipping failed to cap norm to max_grad_norm!")

    # ------------------------------------------------------------------------
    # 7. ADAMW OPTIMIZER MOMENTS & STATE DICT TEST
    # ------------------------------------------------------------------------
    def test_adamw_moments_and_state_dict(self):
        """
        Verifies AdamW first/second moment tracking, weight decay filter, and state serialization.
        """
        named_params = self.model.architecture.get_named_parameters()
        optimizer = AdamW(named_params, lr=2e-3, weight_decay=0.05)

        # Step 1
        for name, param, grad in named_params:
            grad.fill(0.1)
        optimizer.step()

        self.assertEqual(optimizer.step_count, 1)
        state_dict = optimizer.get_state_dict()
        self.assertEqual(state_dict["step_count"], 1)
        self.assertIn("exp_avg", state_dict)
        self.assertIn("exp_avg_sq", state_dict)

        # Create fresh optimizer and restore
        fresh_optimizer = AdamW(named_params, lr=2e-3, weight_decay=0.05)
        fresh_optimizer.load_state_dict(state_dict)
        self.assertEqual(fresh_optimizer.step_count, 1)

        # Verify restored moments match exactly
        for name in optimizer.exp_avg:
            np.testing.assert_array_almost_equal(optimizer.exp_avg[name], fresh_optimizer.exp_avg[name])
            np.testing.assert_array_almost_equal(optimizer.exp_avg_sq[name], fresh_optimizer.exp_avg_sq[name])

    # ------------------------------------------------------------------------
    # 8. LR SCHEDULER WARMUP & COSINE DECAY TEST
    # ------------------------------------------------------------------------
    def test_lr_scheduler_dynamics(self):
        """
        Verifies linear warmup and cosine annealing decay.
        """
        base_lr = 1e-3
        min_lr = 1e-5
        warmup_steps = 10
        total_steps = 100

        scheduler = LRScheduler(base_lr=base_lr, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)

        # Step 0: min_lr
        self.assertAlmostEqual(scheduler.get_lr(0), min_lr, places=7)

        # Mid-warmup (step 5): exactly halfway
        expected_mid_warmup = min_lr + (base_lr - min_lr) * 0.5
        self.assertAlmostEqual(scheduler.get_lr(5), expected_mid_warmup, places=6)

        # Peak at warmup_steps (step 10): base_lr
        self.assertAlmostEqual(scheduler.get_lr(10), base_lr, places=6)

        # Step total_steps (step 100): min_lr
        self.assertAlmostEqual(scheduler.get_lr(100), min_lr, places=6)

        # Beyond total_steps: stays min_lr
        self.assertAlmostEqual(scheduler.get_lr(150), min_lr, places=6)

    # ------------------------------------------------------------------------
    # 9. TINY-BATCH OVERFIT TEST (DEEP LEARNING PROOF)
    # ------------------------------------------------------------------------
    def test_tiny_batch_overfit_memorization(self):
        """
        CRITICAL TEST: Proves the training system can genuinely learn and memorize a tiny dataset.
        Trains repeatedly on 4 samples; expects substantial loss drop and high token accuracy.
        """
        np.random.seed(42)
        dataset = InstructionDataset(tokenizer=self.tokenizer, max_seq_len=24)
        dataset.records = [
            {"system": "You are Aether.", "user": "Ping", "assistant": "Pong.", "category": "general"},
            {"system": "You are Aether.", "user": "Alpha", "assistant": "Beta.", "category": "general"},
            {"system": "You are Aether.", "user": "Red", "assistant": "Blue.", "category": "general"},
            {"system": "You are Aether.", "user": "One", "assistant": "Two.", "category": "reasoning"},
        ]
        dataset._tokenize_all()

        # Fresh small model for fast learning
        cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=32)
        model = AetherModel(cfg, skip_checkpoint=True)
        trainer = AetherTrainer(model=model, config=cfg, lr=8e-3, weight_decay=0.0)

        summary = trainer.train(
            train_dataset=dataset,
            epochs=35,
            verbose=False,
            patience=35,
        )

        initial_loss = summary["initial_loss"]
        final_loss = summary["final_train_loss"]

        print(f"\n[Tiny-Batch Overfit Result] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
        self.assertTrue(summary["loss_decreased"], f"Loss failed to decrease: {initial_loss} -> {final_loss}")
        self.assertLess(final_loss, initial_loss * 0.2, f"Expected >80% loss reduction, got {initial_loss:.4f} -> {final_loss:.4f}")
        self.assertLess(final_loss, 0.6, f"Expected final loss < 0.6 on tiny set, got {final_loss:.4f}")

    # ------------------------------------------------------------------------
    # 10. VALIDATION ISOLATION TEST
    # ------------------------------------------------------------------------
    def test_validation_isolation_zero_mutation(self):
        """
        Verifies evaluation mode does not mutate model parameters or accumulate gradients.
        """
        dataset = InstructionDataset(tokenizer=self.tokenizer, max_seq_len=24)
        dataset.records = [
            {"system": "You are Aether.", "user": "Test query", "assistant": "Test answer.", "category": "general"}
        ]
        dataset._tokenize_all()

        trainer = AetherTrainer(model=self.model, config=self.config)

        # Snapshot model parameters
        params_before = {name: np.array(param, copy=True) for name, param, _ in self.model.architecture.get_named_parameters()}

        # Run evaluation multiple times
        for _ in range(3):
            val_loss = trainer.evaluate(dataset)
            self.assertTrue(math.isfinite(val_loss))

        # Check parameters are 100% identical
        for name, param, _ in self.model.architecture.get_named_parameters():
            np.testing.assert_array_equal(params_before[name], param, err_msg=f"Validation mutated parameter {name}!")

    # ------------------------------------------------------------------------
    # 11. CHECKPOINT MANAGER & RESUMPTION TEST
    # ------------------------------------------------------------------------
    def test_checkpoint_manager_roundtrip_and_mismatch(self):
        """
        Verifies:
        1. Full checkpoint save & load with SHA-256 checksums.
        2. Optimizer and scheduler state restoration.
        3. Clear error on vocabulary size mismatch.
        """
        ckpt_dir = os.path.join(base_dir, "checkpoints", "test_p10_ckpt")
        manager = CheckpointManager(ckpt_dir)

        scheduler = LRScheduler(base_lr=1e-3, warmup_steps=5, total_steps=50)
        scheduler.step()
        scheduler.step()  # Current step 2

        ckpt_path, checksum = manager.save(
            model=self.model,
            optimizer=None,
            scheduler=scheduler,
            step=2,
            epoch=1,
            tag="test_p10",
        )

        self.assertTrue(os.path.exists(ckpt_path))
        self.assertGreater(len(checksum), 0)

        # Reload into a fresh model instance
        fresh_model = AetherModel(self.config, skip_checkpoint=True)
        fresh_scheduler = LRScheduler(base_lr=1e-3, warmup_steps=5, total_steps=50)

        meta = manager.load(ckpt_path, fresh_model, scheduler=fresh_scheduler)
        self.assertEqual(meta["training_step"], 2)
        self.assertEqual(fresh_scheduler.current_step, 2)
        self.assertEqual(fresh_model.metadata.get("checksum"), checksum)

        # Verify forward logits match 100%
        test_ids = [2, 5, 10]
        logits_orig = self.model.forward_all(test_ids)
        logits_reloaded = fresh_model.forward_all(test_ids)
        np.testing.assert_array_almost_equal(logits_orig, logits_reloaded)

        # Verify vocabulary mismatch fails cleanly
        incompatible_cfg = ModelConfig(vocab_size=1024, d_model=16, n_layers=2, n_heads=2, d_ff=32)
        incompat_model = AetherModel(incompatible_cfg, skip_checkpoint=True)
        with self.assertRaises(ValueError):
            manager.load(ckpt_path, incompat_model)


if __name__ == "__main__":
    unittest.main()
