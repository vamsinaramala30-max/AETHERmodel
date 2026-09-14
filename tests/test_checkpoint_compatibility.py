"""
AETHER MODEL — Checkpoint Compatibility & Integrity Hardening Test Suite
Validates all 8 Phase 2 requirements:
1. Missing required parameters are detected.
2. Unexpected parameters are reported.
3. Shape mismatches are reported.
4. Vocabulary mismatch is reported.
5. Architecture mismatch is reported.
6. Partial loading cannot incorrectly report has_trained_weights=True.
7. A fully loaded checkpoint is clearly distinguishable from random initialization.
8. Loading the same checkpoint twice produces the exact same model state.
"""

import sys
import os
import json
import tempfile
import unittest
import copy
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from model.config.model_config import ModelConfig
from tokenizer.tokenizer import AetherTokenizer


class TestCheckpointCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = os.path.join(base_dir, "checkpoints", "aether_checkpoint_v1.json")
        cls.vocab_path = os.path.join(base_dir, "checkpoints", "aether_vocab.json")
        with open(cls.ckpt_path, "r", encoding="utf-8") as f:
            cls.valid_payload = json.load(f)

    def _create_temp_checkpoint(self, payload: dict, update_checksum: bool = True) -> str:
        import hashlib
        p = copy.deepcopy(payload)
        if update_checksum and "metadata" in p and "state_dict" in p:
            data_str = json.dumps(p["state_dict"], sort_keys=True, separators=(',', ':'))
            cs = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
            if "checksum" in p["metadata"]:
                p["metadata"]["checksum"] = cs
            if "checkpoint_sha256" in p["metadata"]:
                p["metadata"]["checkpoint_sha256"] = cs
        tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tf.close()
        with open(tf.name, "w", encoding="utf-8") as f:
            json.dump(p, f)
        self.addCleanup(lambda: os.remove(tf.name) if os.path.exists(tf.name) else None)
        return tf.name

    def test_01_valid_authoritative_checkpoint_loads_cleanly(self):
        """Authoritative checkpoint loads with exact shapes, params, and READY status."""
        config = ModelConfig.v1_legacy()
        model = AetherModel(config)
        self.assertTrue(model.config.has_trained_weights)
        self.assertEqual(model.load_status, "READY")
        self.assertTrue(model.config.weights_hash.startswith("sha256_"))
        self.assertEqual(len(model.last_validation_errors), 0)

        # Check all tensor shapes
        sd = model.get_state_dict()
        self.assertEqual(np.array(sd["token_embedding"]["weight"]).shape, (579, 64))
        self.assertEqual(len(sd["blocks"]), 2)
        for i in range(2):
            self.assertEqual(np.array(sd["blocks"][i]["attn"]["q_proj"]).shape, (64, 64))
            self.assertEqual(np.array(sd["blocks"][i]["ffn"]["w1"]).shape, (64, 128))
            self.assertEqual(np.array(sd["blocks"][i]["ffn"]["w2"]).shape, (128, 64))
        self.assertEqual(np.array(sd["lm_head"]["weight"]).shape, (64, 579))
        self.assertEqual(np.array(sd["lm_head"]["bias"]).shape, (579,))

    def test_02_missing_required_parameter_is_detected_and_rejected(self):
        """Missing parameter in state dict causes load to fail and not set has_trained_weights."""
        corrupted = copy.deepcopy(self.valid_payload)
        # Delete lm_head.bias
        del corrupted["state_dict"]["lm_head"]["bias"]
        corrupted_path = self._create_temp_checkpoint(corrupted)

        model = AetherModel(ModelConfig.v1_legacy(weights_path=corrupted_path) if hasattr(ModelConfig, "v1_legacy") else ModelConfig(vocab_size=579, d_model=64, n_layers=2, n_heads=2, d_ff=128, weights_path=corrupted_path))
        self.assertFalse(model.config.has_trained_weights)
        self.assertNotEqual(model.load_status, "READY")
        self.assertTrue(any("missing" in err.lower() for err in model.last_validation_errors))

    def test_03_unexpected_parameter_is_detected_and_rejected(self):
        """Unexpected parameter in state dict causes load to fail and not set has_trained_weights."""
        corrupted = copy.deepcopy(self.valid_payload)
        # Add extraneous parameter
        corrupted["state_dict"]["blocks"][0]["attn"]["extra_bias_tensor"] = [0.0] * 64
        corrupted_path = self._create_temp_checkpoint(corrupted)

        model = AetherModel(ModelConfig(vocab_size=579, d_model=64, n_layers=2, n_heads=2, d_ff=128, weights_path=corrupted_path))
        self.assertFalse(model.config.has_trained_weights)
        self.assertNotEqual(model.load_status, "READY")
        self.assertTrue(any("unexpected" in err.lower() for err in model.last_validation_errors))

    def test_04_shape_mismatch_is_detected_and_rejected(self):
        """Tensors with invalid shapes are rejected without silently mutating layers."""
        corrupted = copy.deepcopy(self.valid_payload)
        # Change FFN w1 from (64, 128) to (64, 100)
        corrupted["state_dict"]["blocks"][0]["ffn"]["w1"] = [[0.0] * 100 for _ in range(64)]
        corrupted_path = self._create_temp_checkpoint(corrupted)

        model = AetherModel(ModelConfig(vocab_size=579, d_model=64, n_layers=2, n_heads=2, d_ff=128, weights_path=corrupted_path))
        self.assertFalse(model.config.has_trained_weights)
        self.assertEqual(model.load_status, "ARCHITECTURE_MISMATCH")
        self.assertTrue(any("shape mismatch" in err.lower() for err in model.last_validation_errors))

    def test_05_vocab_size_mismatch_is_detected_and_rejected(self):
        """ModelConfig with vocab_size != 579 rejects 579-token checkpoint."""
        config = ModelConfig(vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)
        result = model.load_checkpoint(self.ckpt_path)
        self.assertFalse(result)
        self.assertFalse(model.config.has_trained_weights)
        self.assertEqual(model.load_status, "ARCHITECTURE_MISMATCH")
        self.assertTrue(any("vocab_size" in err.lower() for err in model.last_validation_errors))

    def test_06_architecture_layer_count_mismatch_is_rejected(self):
        """ModelConfig with n_layers=4 rejects 2-layer checkpoint."""
        config = ModelConfig(vocab_size=579, d_model=64, n_layers=4, n_heads=2, d_ff=128)
        model = AetherModel(config, skip_checkpoint=True)
        result = model.load_checkpoint(self.ckpt_path)
        self.assertFalse(result)
        self.assertFalse(model.config.has_trained_weights)
        self.assertEqual(model.load_status, "ARCHITECTURE_MISMATCH")
        self.assertTrue(any("n_layers" in err.lower() or "layers" in err.lower() for err in model.last_validation_errors))

    def test_07_partial_loading_cannot_report_trained_weights(self):
        """If a checkpoint fails validation midway, has_trained_weights MUST remain False."""
        corrupted = copy.deepcopy(self.valid_payload)
        # Block 0 is valid, but Block 1 has a broken tensor
        corrupted["state_dict"]["blocks"][1]["ffn"]["w2"] = [[0.0] * 32 for _ in range(128)]
        corrupted_path = self._create_temp_checkpoint(corrupted)

        model = AetherModel(ModelConfig(vocab_size=579, d_model=64, n_layers=2, n_heads=2, d_ff=128, weights_path=corrupted_path))
        self.assertFalse(model.config.has_trained_weights)
        self.assertNotEqual(model.load_status, "READY")
        self.assertNotEqual(model.config.weights_hash, self.valid_payload["metadata"]["weights_hash"])

    def test_08_fully_loaded_distinguishable_from_random_initialization(self):
        """Unweighted model vs loaded checkpoint model have distinct flags and states."""
        unweighted = AetherModel(ModelConfig.v1_legacy(), skip_checkpoint=True)
        self.assertFalse(unweighted.config.has_trained_weights)
        self.assertEqual(unweighted.config.weights_hash, "missing_weights_sha256")

        loaded = AetherModel(ModelConfig.v1_legacy())
        self.assertTrue(loaded.config.has_trained_weights)
        self.assertTrue(loaded.config.weights_hash.startswith("sha256_"))
        self.assertNotEqual(loaded.config.weights_hash, "missing_weights_sha256")

        # Analytical forward pass produces different outputs
        test_ids = [10, 20, 30]
        unweighted_logits = unweighted.forward(test_ids)
        loaded_logits = loaded.forward(test_ids)
        self.assertFalse(np.allclose(unweighted_logits, loaded_logits, atol=1e-3))

    def test_09_loading_same_checkpoint_twice_is_deterministic_and_idempotent(self):
        """Loading the same checkpoint repeatedly produces exact identical state dict and logits."""
        model1 = AetherModel(ModelConfig.v1_legacy())
        model2 = AetherModel(ModelConfig.v1_legacy())

        sd1 = model1.get_state_dict()
        sd2 = model2.get_state_dict()

        # Check state dict tensor identity
        self.assertTrue(np.array_equal(sd1["token_embedding"]["weight"], sd2["token_embedding"]["weight"]))
        self.assertTrue(np.array_equal(sd1["lm_head"]["weight"], sd2["lm_head"]["weight"]))

        test_ids = [5, 15, 25, 35]
        logits1 = model1.forward(test_ids)
        logits2 = model2.forward(test_ids)
        self.assertTrue(np.array_equal(logits1, logits2))

    def test_10_tokenizer_checkpoint_contract_alignment(self):
        """Authoritative tokenizer vocabulary size matches checkpoint vocab size exactly."""
        tokenizer = AetherTokenizer(vocab_file=self.vocab_path, frozen=True)
        model = AetherModel(ModelConfig.v1_legacy())

        # Should pass with 0 errors
        tokenizer.validate_against_vocab_size(model.vocab_size)
        self.assertEqual(tokenizer.vocab_size, model.vocab_size)
        self.assertEqual(tokenizer.vocab_size, 579)

        # Mismatched check should raise ValueError
        with self.assertRaises(ValueError):
            tokenizer.validate_against_vocab_size(4096)


if __name__ == "__main__":
    unittest.main()
