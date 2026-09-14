"""
AETHER MODEL — Analytical vs Finite-Difference Numerical Gradient Verification
Compares analytical backpropagation gradients against numerical two-sided finite difference approximations:
    grad_num = [ Loss(w + eps) - Loss(w - eps) ] / (2 * eps)
Tests Token Embedding, Multi-Head Attention, Feed-Forward Network, LayerNorm, and Output Head.
"""

import sys
import os
import unittest
import math
import numpy as np


src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.config.model_config import ModelConfig
from model.architecture.transformer import AetherTransformerArchitecture

def compute_cross_entropy_loss(logits_seq, target_ids):
    """
    logits_seq: [seq_len, vocab_size]
    target_ids: [seq_len]
    Returns scalar average cross-entropy loss.
    """
    total_loss = 0.0
    for t in range(len(target_ids)):
        logits = logits_seq[t]
        target = target_ids[t]
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_e = sum(exps) or 1e-9
        prob = max(exps[target] / sum_e, 1e-9)
        total_loss += -math.log(prob)
    return total_loss / float(len(target_ids))

def compute_loss_grad_wrt_logits(logits_seq, target_ids):
    """
    Returns grad_logits [seq_len, vocab_size] for average cross-entropy loss.
    """
    N = float(len(target_ids))
    grad_logits = []
    for t in range(len(target_ids)):
        logits = logits_seq[t]
        target = target_ids[t]
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_e = sum(exps) or 1e-9
        probs = [e / sum_e for e in exps]
        g_row = [probs[j] / N for j in range(len(probs))]
        g_row[target] -= 1.0 / N
        grad_logits.append(g_row)
    return grad_logits

class TestGradientIntegrity(unittest.TestCase):
    def setUp(self):
        # Small model configuration for high-precision finite-difference verification
        self.config = ModelConfig(vocab_size=16, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=16)
        self.model = AetherTransformerArchitecture(self.config)
        self.input_ids = [2, 5, 8, 3]
        self.target_ids = [5, 8, 3, 2]
        self.eps = 1e-4

    def test_gradient_verification_all_layers(self):
        print("\n=== STARTING NUMERICAL GRADIENT VERIFICATION ===")

        # 1. Forward pass
        logits = self.model.forward_all(self.input_ids)
        initial_loss = compute_cross_entropy_loss(logits, self.target_ids)
        self.assertTrue(math.isfinite(initial_loss))

        # 2. Analytical backward pass
        self.model.zero_grad()
        grad_logits = compute_loss_grad_wrt_logits(logits, self.target_ids)
        self.model.backward(grad_logits)

        # 3. Check selected parameter gradients against finite differences
        named_params = self.model.get_named_parameters()
        checked_count = 0

        for param_name, param_tensor, grad_tensor in named_params:
            # Check 1D or 2D parameter tensors
            if (isinstance(param_tensor, np.ndarray) and param_tensor.ndim == 2) or isinstance(param_tensor[0], list): # 2D matrix
                rows = len(param_tensor)
                cols = len(param_tensor[0])

                # Check 2 representative indices per matrix
                check_indices = [(0, 0), (min(1, rows - 1), min(1, cols - 1))]
                for r, c in check_indices:
                    orig_val = param_tensor[r][c]
                    g_ana = grad_tensor[r][c]

                    # Loss(w + eps)
                    param_tensor[r][c] = orig_val + self.eps
                    l_plus = compute_cross_entropy_loss(self.model.forward_all(self.input_ids), self.target_ids)

                    # Loss(w - eps)
                    param_tensor[r][c] = orig_val - self.eps
                    l_minus = compute_cross_entropy_loss(self.model.forward_all(self.input_ids), self.target_ids)

                    param_tensor[r][c] = orig_val # Restore

                    g_num = (l_plus - l_minus) / (2.0 * self.eps)
                    abs_err = abs(g_ana - g_num)
                    rel_err = abs_err / max(abs(g_ana), abs(g_num), 1e-4)

                    print(f"[{param_name}[{r},{c}]] Analytical: {g_ana:+.6f} | Numerical: {g_num:+.6f} | RelErr: {rel_err:.6f}")
                    self.assertLess(rel_err, 0.05, f"Gradient check failed for {param_name}[{r},{c}]")
                    checked_count += 1
            else: # 1D vector
                length = len(param_tensor)
                check_indices = [0, min(1, length - 1)]
                for i in check_indices:
                    orig_val = param_tensor[i]
                    g_ana = grad_tensor[i]

                    param_tensor[i] = orig_val + self.eps
                    l_plus = compute_cross_entropy_loss(self.model.forward_all(self.input_ids), self.target_ids)

                    param_tensor[i] = orig_val - self.eps
                    l_minus = compute_cross_entropy_loss(self.model.forward_all(self.input_ids), self.target_ids)

                    param_tensor[i] = orig_val

                    g_num = (l_plus - l_minus) / (2.0 * self.eps)
                    abs_err = abs(g_ana - g_num)
                    rel_err = abs_err / max(abs(g_ana), abs(g_num), 1e-4)

                    print(f"[{param_name}[{i}]] Analytical: {g_ana:+.6f} | Numerical: {g_num:+.6f} | RelErr: {rel_err:.6f}")
                    self.assertLess(rel_err, 0.05, f"Gradient check failed for {param_name}[{i}]")
                    checked_count += 1

        print(f"\n[PASS] Verified {checked_count} analytical gradients against finite differences successfully.")

if __name__ == "__main__":
    unittest.main()
