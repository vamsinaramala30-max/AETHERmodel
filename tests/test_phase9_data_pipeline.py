"""
AETHER MODEL — Phase 9 Data Pipeline Test Suite

Validates:
1. Data cleaning and schema normalization.
2. Deduplication using content hashes.
3. Strict zero-leakage Train/Validation/Evaluation splitting.
4. Causal LM training pair generation (Input != Target, Input[1:] == Target[:-1]).
5. Vocabulary bounds for all sequence token IDs.
6. Batch padding and attention mask generation.
"""

import json
import os
import sys
import tempfile
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data.cleaner import DatasetCleaner
from data.dataset import CausalInstructionDataset
from data.split import DatasetSplitter
from tokenizer.tokenizer import AetherTokenizer


class TestPhase9DataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bpe_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        if os.path.exists(bpe_path):
            cls.tokenizer = AetherTokenizer(vocab_file=bpe_path, frozen=True)
        else:
            cls.tokenizer = AetherTokenizer(frozen=True)

    def test_01_cleaner_handles_malformed_and_empty_records(self):
        """Verifies cleaner filters empty lines, broken JSON, and missing fields."""
        cleaner = DatasetCleaner()

        # 1. Valid record
        valid = cleaner.clean_record({
            "system": "You are Aether.",
            "user": "What is Python?",
            "assistant": "Python is a high-level programming language.",
            "category": "coding",
        })
        self.assertIsNotNone(valid)
        self.assertEqual(valid["user"], "What is Python?")
        self.assertEqual(valid["category"], "coding")

        # 2. Missing assistant
        missing_asst = cleaner.clean_record({"user": "Hello"})
        self.assertIsNone(missing_asst)

        # 3. Missing user
        missing_user = cleaner.clean_record({"assistant": "Hi"})
        self.assertIsNone(missing_user)

        # 4. Empty strings
        empty_str = cleaner.clean_record({"user": "   ", "assistant": "   "})
        self.assertIsNone(empty_str)

    def test_02_cleaner_deduplication(self):
        """Verifies cleaner removes duplicate records."""
        sample_records = [
            {"user": "Explain RAG", "assistant": "RAG is Retrieval-Augmented Generation.", "category": "rag"},
            {"user": "Explain RAG", "assistant": "RAG is Retrieval-Augmented Generation.", "category": "rag"},
            {"user": "explain rag", "assistant": "rag is retrieval-augmented generation.", "category": "rag"},
            {"user": "What is Aether?", "assistant": "Aether is an AI workspace.", "category": "aether"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
            for r in sample_records:
                tf.write(json.dumps(r) + "\n")
            temp_path = tf.name

        try:
            cleaner = DatasetCleaner()
            cleaned, stats = cleaner.clean_file(temp_path)
            self.assertEqual(len(cleaned), 2)
            self.assertEqual(stats["duplicate_records"], 2)
            self.assertEqual(stats["valid_records"], 2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_03_strict_zero_leakage_split(self):
        """Verifies DatasetSplitter enforces strict zero-leakage across Train, Val, and Eval."""
        eval_suite = [
            {"prompt": "eval benchmark prompt 1", "category": "coding"},
            {"prompt": "eval benchmark prompt 2", "category": "planning"},
        ]
        corpus = [
            {"user": "eval benchmark prompt 1", "assistant": "response 1", "category": "coding"},  # Should be filtered
            {"user": "train prompt A", "assistant": "response A", "category": "coding"},
            {"user": "train prompt B", "assistant": "response B", "category": "coding"},
            {"user": "train prompt C", "assistant": "response C", "category": "planning"},
            {"user": "train prompt D", "assistant": "response D", "category": "planning"},
            {"user": "train prompt E", "assistant": "response E", "category": "planning"},
        ]
        splitter = DatasetSplitter(seed=42)
        train_recs, val_recs, report = splitter.split(
            corpus,
            val_ratio=0.33,
            eval_records=eval_suite,
            filter_eval_leakage=True,
        )
        self.assertTrue(report["is_strictly_disjoint"])
        self.assertEqual(report["train_val_overlap_count"], 0)
        self.assertEqual(report["train_eval_overlap_count"], 0)
        self.assertEqual(report["val_eval_overlap_count"], 0)
        self.assertEqual(report["eval_held_out_records"], 1)

    def test_04_causal_lm_pair_alignment_and_shift(self):
        """Verifies causal language modeling pair alignment: Input[1:] == Target[:-1] and Input != Target."""
        sample_records = [
            {
                "system": "You are Aether.",
                "user": "What is 25 x 4?",
                "assistant": "25 x 4 is 100.",
                "category": "reasoning",
            },
            {
                "system": "",
                "user": "Write a python function.",
                "assistant": "def add(a, b):\n    return a + b",
                "category": "coding",
            },
        ]
        ds = CausalInstructionDataset(sample_records, tokenizer=self.tokenizer)
        self.assertEqual(len(ds), 2)

        for i in range(len(ds)):
            ex = ds[i]
            inp = ex["input_ids"]
            tgt = ex["target_ids"]

            # Causal shift invariant
            self.assertNotEqual(inp, tgt, "Input and Target must never be identical in causal LM!")
            self.assertEqual(len(inp), len(tgt))
            self.assertEqual(inp[1:], tgt[:-1], "Causal shift invariant failed!")

            # Target ends with EOS token
            eos_id = self.tokenizer.token_to_id.get("<eos>", 3)
            self.assertEqual(tgt[-1], eos_id, "Target sequence must terminate with EOS token!")

    def test_05_token_ids_within_bounds_in_dataset(self):
        """Verifies all token IDs in tokenized dataset are within [0, vocab_size - 1]."""
        clean_train_path = os.path.join(base_dir, "data", "cleaned", "aether_train_split.jsonl")
        if os.path.exists(clean_train_path):
            ds = CausalInstructionDataset(clean_train_path, tokenizer=self.tokenizer)
            vsize = self.tokenizer.vocab_size
            for ex in ds:
                for tid in ex["input_ids"]:
                    self.assertGreaterEqual(tid, 0)
                    self.assertLess(tid, vsize)
                for tid in ex["target_ids"]:
                    self.assertGreaterEqual(tid, 0)
                    self.assertLess(tid, vsize)

    def test_06_batch_generation_with_padding_and_masks(self):
        """Verifies batch construction with proper padding and attention masks."""
        sample_records = [
            {"user": "Short query", "assistant": "Short reply.", "category": "general"},
            {"user": "Much longer query with multiple words", "assistant": "A considerably longer response containing several words and details.", "category": "general"},
        ]
        ds = CausalInstructionDataset(sample_records, tokenizer=self.tokenizer)
        batch = ds.get_batch([0, 1], pad_to_max=True)

        self.assertEqual(batch["batch_size"], 2)
        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(len(batch["target_ids"]), 2)
        self.assertEqual(len(batch["attention_mask"]), 2)

        max_l = batch["seq_len"]
        self.assertEqual(len(batch["input_ids"][0]), max_l)
        self.assertEqual(len(batch["input_ids"][1]), max_l)
        self.assertEqual(len(batch["attention_mask"][0]), max_l)
        self.assertEqual(len(batch["attention_mask"][1]), max_l)


if __name__ == "__main__":
    unittest.main()
