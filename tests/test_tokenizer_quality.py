"""
AETHER MODEL — Tokenizer Quality & Hardening Test Suite
Validates encoding/decoding determinism, special tokens, punctuation, whitespace,
vocabulary stability, and out-of-vocabulary handling.
"""

import sys
import os
import unittest
import tempfile

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.tokenizer import (
    AetherTokenizer,
    SPECIAL_TOKENS,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ASSISTANT_TOKEN_ID,
    TOOL_TOKEN_ID,
    EVIDENCE_TOKEN_ID,
)
from tokenizer.special_tokens import SpecialTokens

class TestTokenizerQuality(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AetherTokenizer()

    def test_special_tokens_integrity(self):
        """Verifies special tokens have fixed IDs and bidirectional mapping.

        Note: EOS is NOT compared against a hard-coded integer here because it
        differs between legacy (EOS=2) and BPE (EOS=3) modes.  Instead we
        verify that the live tokenizer is internally self-consistent via
        SpecialTokens.validate().
        """
        st = SpecialTokens.from_tokenizer(self.tokenizer)
        # PAD and UNK are universally 0 and 1
        self.assertEqual(self.tokenizer.token_to_id["<pad>"], PAD_TOKEN_ID)
        self.assertEqual(self.tokenizer.token_to_id["<unk>"], UNK_TOKEN_ID)
        # EOS must be present and internally consistent
        self.assertIn("<eos>", self.tokenizer.token_to_id)
        self.assertEqual(self.tokenizer.token_to_id["<eos>"], st.eos)
        # SpecialTokens must agree with the tokenizer on every token it knows
        st.validate(self.tokenizer)
        # Legacy-mode specific: SYSTEM, USER, ASSISTANT, TOOL, EVIDENCE
        self.assertEqual(self.tokenizer.token_to_id["<system>"], st.system)
        self.assertEqual(self.tokenizer.token_to_id["<user>"], st.user)
        self.assertEqual(self.tokenizer.token_to_id["<assistant>"], st.assistant)
        self.assertEqual(self.tokenizer.token_to_id["<tool>"], st.tool)
        self.assertEqual(self.tokenizer.token_to_id["<evidence>"], st.evidence)

        for tok, tid in self.tokenizer.token_to_id.items():
            if tok in SPECIAL_TOKENS:
                self.assertEqual(self.tokenizer.id_to_token[tid], tok)

    def test_encoding_decoding_determinism(self):
        """Verifies that encoding and decoding identical strings produces identical outputs."""
        texts = [
            "Hello",
            "What is Aether?",
            "Explain automation.",
            "Create a project plan.",
            "def calculate_total(items: list) -> float:\n    return sum(items)",
            "Aether AI Core orchestrates intent classification, contextual memory, and verified tools. Each step is logged and confirmed."
        ]
        for t in texts:
            ids1 = self.tokenizer.encode(t)
            ids2 = self.tokenizer.encode(t)
            self.assertEqual(ids1, ids2, f"Encoding not deterministic for: {t}")

            dec1 = self.tokenizer.decode(ids1)
            dec2 = self.tokenizer.decode(ids1)
            self.assertEqual(dec1, dec2, f"Decoding not deterministic for: {t}")
            self.assertGreater(len(dec1), 0)

    def test_unknown_text_frozen_handling(self):
        """Verifies frozen tokenizer maps unseen words to <unk> without mutating vocab."""
        self.tokenizer.freeze()
        initial_vocab_size = self.tokenizer.vocab_size

        unseen_text = "xyzzy999qwx supercalifragilisticexpialidocious"
        ids = self.tokenizer.encode(unseen_text)
        self.assertEqual(self.tokenizer.vocab_size, initial_vocab_size, "Vocab size changed while frozen!")
        if not self.tokenizer.is_bpe:
            self.assertTrue(all(tid == UNK_TOKEN_ID for tid in ids), f"Expected UNK_TOKEN_ID, got: {ids}")
        else:
            self.assertGreater(len(ids), 0)

    def test_punctuation_and_whitespace_consistency(self):
        """Verifies punctuation attachments and structured decoding."""
        raw = "Hello, world! How are you? Aether: (v1.0) [ready]."
        ids = self.tokenizer.encode(raw)
        decoded = self.tokenizer.decode(ids)
        dec_lower = decoded.lower()
        self.assertIn("hello,", dec_lower)
        self.assertIn("world!", dec_lower)
        self.assertIn("you?", dec_lower)
        self.assertIn("aether:", dec_lower)
        self.assertIn("[ready].", dec_lower)

    def test_vocabulary_save_and_reload(self):
        """Verifies vocabulary persistence and exact ID preservation across reloads."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            self.tokenizer.encode("new_custom_term_12345")
            custom_id = self.tokenizer.token_to_id["new_custom_term_12345"]
            self.tokenizer.save_vocab(temp_path)

            fresh_tokenizer = AetherTokenizer(vocab_file=temp_path)
            self.assertIn("new_custom_term_12345", fresh_tokenizer.token_to_id)
            self.assertEqual(fresh_tokenizer.token_to_id["new_custom_term_12345"], custom_id)
            self.assertEqual(fresh_tokenizer.id_to_token[custom_id], "new_custom_term_12345")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_frozen_initialization_retains_default_vocabulary(self):
        """Verifies AetherTokenizer(frozen=True) retains the default vocabulary and stays frozen."""
        tok_frozen = AetherTokenizer(frozen=True)
        self.assertTrue(tok_frozen.frozen)
        self.assertGreater(tok_frozen.vocab_size, 100)
        self.assertIn("aether", tok_frozen.token_to_id)
        self.assertIn("hello", tok_frozen.token_to_id)
        
        # Verify it cannot be mutated
        initial_size = tok_frozen.vocab_size
        tok_frozen.encode("unseen_word_xyz_123")
        self.assertEqual(tok_frozen.vocab_size, initial_size)

    def test_unicode_and_accented_words(self):
        """Verifies that non-ASCII Unicode words and symbols are preserved in tokenization."""
        # Unfrozen mode: words get added
        tok_unfrozen = AetherTokenizer(frozen=False)
        tokens = tok_unfrozen.tokenize("café résumé naïve 世界 🚀")
        self.assertIn("café", tokens)
        self.assertIn("résumé", tokens)
        self.assertIn("naïve", tokens)
        self.assertIn("世界", tokens)
        self.assertIn("🚀", tokens)

        ids = tok_unfrozen.encode("café résumé naïve")
        decoded = tok_unfrozen.decode(ids)
        self.assertEqual(decoded, "café résumé naïve")

        # Frozen mode: tokens are matched and mapped to UNK instead of dropped
        tok_frozen = AetherTokenizer(frozen=True)
        frozen_ids = tok_frozen.encode("café")
        self.assertEqual(frozen_ids, [UNK_TOKEN_ID])

    def test_numpy_and_integer_decode_compatibility(self):
        """Verifies decode handles standard ints, numpy scalar ints, and numpy arrays."""
        import numpy as np
        # Standard ints
        dec_list = self.tokenizer.decode([118, 108])
        self.assertIsInstance(dec_list, str)

        # NumPy integer scalars
        np_ids = [np.int64(118), np.int32(108)]
        dec_np_scalars = self.tokenizer.decode(np_ids)
        self.assertEqual(dec_np_scalars, dec_list)

        # NumPy 1D array
        dec_np_arr = self.tokenizer.decode(np.array([118, 108], dtype=np.int64))
        self.assertEqual(dec_np_arr, dec_list)

        # Invalid types rejected cleanly
        with self.assertRaises(TypeError):
            self.tokenizer.decode([3.14])

        with self.assertRaises(TypeError):
            self.tokenizer.decode(["not_an_int"])

        with self.assertRaises(TypeError):
            self.tokenizer.decode([True])

    def test_empty_and_whitespace_inputs(self):
        """Verifies empty and whitespace-only inputs."""
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.tokenizer.decode(None), "")
        self.assertEqual(self.tokenizer.encode("     "), [])
        self.assertEqual(self.tokenizer.encode("\n\t  \r\n"), [])

    def test_token_ids_within_vocab_bounds(self):
        """Verifies that all encoded token IDs are strictly within [0, vocab_size - 1]."""
        test_strings = [
            "Hello",
            "What is Aether?",
            "",
            "Hello, world! [test]: (100%).",
            "hello     world    test",
            "1 2 3 42 100",
            "xyzzy_unknown_test_token_999",
            "How does Aether orchestrate agentic automations in workspaces?",
            "café résumé naïve",
            "世界 🚀",
        ]
        tok_frozen = AetherTokenizer(frozen=True)
        for s in test_strings:
            ids = tok_frozen.encode(s)
            for tid in ids:
                self.assertGreaterEqual(tid, 0)
                self.assertLess(tid, tok_frozen.vocab_size)

    def test_required_prompt_benchmark_phrases(self):
        """Verifies encoding and decoding across all required Prompt 15 audit phrases."""
        phrases = [
            "Hello",
            "What is Aether?",
            "Explain automation.",
            "Create a project plan.",
            "Python code.",
            "A multi-sentence paragraph. It contains multiple thoughts. It finishes cleanly."
        ]
        for p in phrases:
            ids = self.tokenizer.encode(p)
            self.assertGreater(len(ids), 0)
            decoded = self.tokenizer.decode(ids)
            # Verify primary tokens exist in decoded text
            first_word = p.split()[0].replace(".", "").replace(",", "").lower()
            self.assertIn(first_word, decoded.lower())

if __name__ == "__main__":
    unittest.main()

