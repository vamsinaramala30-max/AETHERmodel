"""
AETHER MODEL — Tokenizer Test Suite
Validates encoding, decoding, round-trip determinism, special tokens, padding, and truncation.
"""

import sys
import os
import unittest

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer.tokenizer import AetherTokenizer, SPECIAL_TOKENS, SYSTEM_TOKEN_ID, USER_TOKEN_ID, ASSISTANT_TOKEN_ID, EOS_TOKEN_ID

class TestTokenizerEngine(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AetherTokenizer()

    def test_special_tokens_registered(self):
        for tok in SPECIAL_TOKENS:
            self.assertIn(tok, self.tokenizer.token_to_id)
            tid = self.tokenizer.token_to_id[tok]
            self.assertEqual(self.tokenizer.id_to_token[tid], tok)

    def test_encode_decode_round_trip(self):
        text = "aether artificial intelligence system"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        for word in text.split():
            self.assertIn(word, decoded.lower())

    def test_batch_encode_padding_truncation(self):
        batch = self.tokenizer.batch_encode(
            ["Hello world", "Aether automation system pipeline execution test"],
            max_length=8,
            pad_to_max=True
        )
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), 8)
        self.assertEqual(len(batch[1]), 8)

    def test_special_tokens_preservation(self):
        prompt = "<system> You are Aether. <user> Explain RAG. <assistant>"
        ids = self.tokenizer.encode(prompt)
        self.assertIn(SYSTEM_TOKEN_ID, ids)
        self.assertIn(USER_TOKEN_ID, ids)
        self.assertIn(ASSISTANT_TOKEN_ID, ids)

if __name__ == "__main__":
    unittest.main()
