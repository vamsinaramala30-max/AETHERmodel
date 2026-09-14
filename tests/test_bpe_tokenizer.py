"""
AETHER MODEL — Phase 9 Production Subword BPE Tokenizer Test Suite

Validates:
1. Subword decomposition on novel and composite words.
2. Natural language round-trip fidelity.
3. Numbers and arithmetic symbols (e.g. 25 × 4 = 100).
4. Punctuation and whitespace formatting.
5. Programming code syntax, indentation, and quotes.
6. Long text multi-sentence consistency.
7. Multilingual Unicode and emoji handling without crashing.
8. Empty and whitespace-only text handling.
9. Token ID bounds: 0 <= token_id < vocab_size.
10. Deterministic serialization and deserialization.
"""

import json
import os
import sys
import unittest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer.bpe import (
    ASSISTANT_TOKEN_ID,
    BOS_TOKEN_ID,
    BYTE_DECODER,
    BYTE_ENCODER,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN_ID,
    UNK_TOKEN_ID,
    USER_TOKEN_ID,
    ByteLevelBPEEngine,
)
from tokenizer.tokenizer import AetherTokenizer
from tokenizer.trainer import ByteLevelBPETrainer


class TestProductionBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bpe_artifact_path = os.path.join(base_dir, "checkpoints", "aether_bpe_tokenizer.json")
        cls.bpe_vocab_path = os.path.join(base_dir, "checkpoints", "aether_bpe_vocab.json")

        if os.path.exists(cls.bpe_artifact_path):
            cls.tokenizer = AetherTokenizer(vocab_file=cls.bpe_artifact_path, frozen=True)
        else:
            # Train on sample corpus if artifact not on disk
            sample_texts = [
                "You are Aether AI, an intelligent agentic workspace assistant.",
                "def calculate_total(items: list) -> float:\n    return sum(items)",
                "25 × 4 = 100",
                "Hello! What's happening?",
                "café résumé naïve 世界 🚀",
            ]
            trainer = ByteLevelBPETrainer(target_vocab_size=1024, min_frequency=1)
            engine, _ = trainer.train_from_texts(sample_texts)
            cls.tokenizer = AetherTokenizer(bpe_engine=engine, frozen=True)

    def test_01_bpe_mode_active_and_special_tokens(self):
        """Verifies tokenizer is running in BPE subword mode with canonical special tokens."""
        self.assertTrue(self.tokenizer.is_bpe)
        self.assertEqual(self.tokenizer.vocab_size, 1024)

        for tok in ["<pad>", "<unk>", "<eos>", "<system>", "<user>", "<assistant>"]:
            self.assertIn(tok, self.tokenizer.token_to_id)
            tid = self.tokenizer.token_to_id[tok]
            self.assertEqual(self.tokenizer.id_to_token[tid], tok)

    def test_02_natural_language_round_trip(self):
        """Verifies exact round-trip for natural language text."""
        phrases = [
            "Hello, how are you?",
            "What is Aether?",
            "Explain automation.",
            "Create a project plan.",
            "Aether AI Core orchestrates intent classification, contextual memory, and verified tools.",
        ]
        for phrase in phrases:
            ids = self.tokenizer.encode(phrase)
            self.assertGreater(len(ids), 0)
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, phrase)

    def test_03_numbers_and_arithmetic(self):
        """Verifies exact round-trip for numbers, formulas, and math symbols."""
        test_cases = [
            "25 × 4 = 100",
            "100 / 5 = 20.0",
            "3.1415926535",
            "Equation: x^2 + y^2 = r^2",
            "-42.50 USD",
        ]
        for tc in test_cases:
            ids = self.tokenizer.encode(tc)
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, tc)

    def test_04_punctuation_and_formatting(self):
        """Verifies punctuation, contractions, and symbols are preserved without mutation."""
        test_cases = [
            "Hello! What's happening?",
            "Parentheses (test), brackets [1, 2, 3], braces {a: 'b'}.",
            "Punctuation: ; , . ! ? / \\ | ~ ` @ # $ % ^ & * - _ + =",
        ]
        for tc in test_cases:
            ids = self.tokenizer.encode(tc)
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, tc)

    def test_05_code_syntax_and_indentation(self):
        """Verifies programming syntax, indentation, and string literals."""
        code = (
            "def hello():\n"
            "    return \"Aether\"\n\n"
            "class AgentRunner:\n"
            "    async def execute(self, task: Task) -> bool:\n"
            "        return True\n"
        )
        ids = self.tokenizer.encode(code)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, code)

    def test_06_long_text_multi_sentence(self):
        """Verifies long multi-paragraph text preserves 100% round-trip fidelity."""
        long_text = (
            "A multi-sentence paragraph. It contains multiple thoughts. It finishes cleanly.\n\n"
            "Aether is an intelligent workspace platform designed for agentic task automation, "
            "contextual retrieval-augmented generation (RAG), and deterministic multi-step planning. "
            "Every step of execution is verified, logged, and audited for safety and reliability."
        )
        ids = self.tokenizer.encode(long_text)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, long_text)

    def test_07_multilingual_unicode_and_emojis(self):
        """Verifies non-ASCII Unicode characters and emojis without crashing or dropping bytes."""
        unicode_text = "café résumé naïve 世界 🚀 🤖 ⚡"
        ids = self.tokenizer.encode(unicode_text)
        self.assertGreater(len(ids), 0)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, unicode_text)

    def test_08_empty_and_whitespace_inputs(self):
        """Verifies empty string and whitespace-only inputs."""
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.tokenizer.decode(None), "")

        # Whitespace strings preserve their exact spaces
        spaces = "     \n\t  "
        ids = self.tokenizer.encode(spaces)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, spaces)

    def test_09_subword_decomposition_on_unseen_words(self):
        """Verifies that unseen rare words are decomposed into subword pieces without UNK."""
        rare_words = [
            "hyperparameterization",
            "antigravitational",
            "microarchitectural",
            "deterministic_execution_pipeline",
        ]
        for w in rare_words:
            ids = self.tokenizer.encode(w)
            self.assertGreater(len(ids), 0)
            # Unseen words decompose into subwords, never failing
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, w)

    def test_10_token_ids_strictly_within_vocabulary_bounds(self):
        """Verifies all encoded token IDs satisfy 0 <= token_id < vocab_size."""
        sample_corpus = [
            "Hello, world!",
            "25 × 4 = 100",
            "def test(): pass",
            "café résumé naïve 世界 🚀",
            "<system> System prompt <user> User prompt <assistant> Response <eos>",
        ]
        vsize = self.tokenizer.vocab_size
        for s in sample_corpus:
            ids = self.tokenizer.encode(s)
            for tid in ids:
                self.assertGreaterEqual(tid, 0, f"Token ID {tid} is negative!")
                self.assertLess(tid, vsize, f"Token ID {tid} >= vocab_size ({vsize})!")

    def test_11_vocabulary_contiguity_and_determinism(self):
        """Verifies vocabulary IDs are contiguous integers from 0 to vocab_size-1."""
        self.tokenizer.validate()
        ids = sorted(self.tokenizer.token_to_id.values())
        self.assertEqual(ids, list(range(len(ids))))

    def test_12_special_tokens_preservation_in_formatted_prompts(self):
        """Verifies special turn tokens are matched and extracted with exact special IDs."""
        prompt = "<system> You are Aether. <user> Explain BPE. <assistant> BPE is Byte-Pair Encoding. <eos>"
        ids = self.tokenizer.encode(prompt)

        sys_id = self.tokenizer.token_to_id["<system>"]
        usr_id = self.tokenizer.token_to_id["<user>"]
        asst_id = self.tokenizer.token_to_id["<assistant>"]
        eos_id = self.tokenizer.token_to_id["<eos>"]

        self.assertIn(sys_id, ids)
        self.assertIn(usr_id, ids)
        self.assertIn(asst_id, ids)
        self.assertIn(eos_id, ids)

        decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.assertIn("<system>", decoded)
        self.assertIn("<user>", decoded)
        self.assertIn("<assistant>", decoded)
        self.assertIn("<eos>", decoded)

    def test_13_save_and_load_round_trip(self):
        """Verifies that saving and reloading produces exact identical tokenizer state and hash."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            self.tokenizer.save_vocab(temp_path)
            loaded = AetherTokenizer(vocab_file=temp_path, frozen=True)
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded.get_vocab_hash(), self.tokenizer.get_vocab_hash())
            self.assertEqual(loaded.algorithm, self.tokenizer.algorithm)

            # Check encoding equivalence
            text = "Testing save/load determinism on Aether BPE tokenizer: 123 + 456 = 579."
            self.assertEqual(loaded.encode(text), self.tokenizer.encode(text))
            self.assertEqual(loaded.decode(loaded.encode(text)), text)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_14_add_special_tokens_bos_eos(self):
        """Verifies add_special_tokens prepends BOS and appends EOS."""
        text = "Hello Aether"
        ids_with_special = self.tokenizer.encode(text, add_special_tokens=True)
        bos_id = self.tokenizer.token_to_id.get("<bos>", 2)
        eos_id = self.tokenizer.token_to_id.get("<eos>", 3)

        self.assertEqual(ids_with_special[0], bos_id)
        self.assertEqual(ids_with_special[-1], eos_id)

        # Decoding with skip_special_tokens=True strips them
        decoded_clean = self.tokenizer.decode(ids_with_special, skip_special_tokens=True)
        self.assertEqual(decoded_clean, text)

    def test_15_batch_encoding_padding_truncation(self):
        """Verifies batch encoding correctly pads and truncates sequences."""
        texts = ["Short", "This is a significantly longer sentence to test truncation."]
        batch = self.tokenizer.batch_encode(texts, max_length=6, pad_to_max=True)
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), 6)
        self.assertEqual(len(batch[1]), 6)

        pad_id = self.tokenizer.token_to_id.get("<pad>", 0)
        self.assertIn(pad_id, batch[0])

    def test_16_special_tokens_resolver(self):
        """Verifies SpecialTokens dataclass resolves correctly from the loaded tokenizer."""
        from tokenizer.special_tokens import SpecialTokens, resolve_eos_id
        st = SpecialTokens.from_tokenizer(self.tokenizer)
        st.validate(self.tokenizer)

        self.assertEqual(st.pad, self.tokenizer.token_to_id["<pad>"])
        self.assertEqual(st.eos, self.tokenizer.token_to_id["<eos>"])
        self.assertEqual(st.bos, self.tokenizer.token_to_id["<bos>"])
        self.assertEqual(st.system, self.tokenizer.token_to_id["<system>"])
        self.assertEqual(st.user, self.tokenizer.token_to_id["<user>"])
        self.assertEqual(st.assistant, self.tokenizer.token_to_id["<assistant>"])
        self.assertEqual(resolve_eos_id(self.tokenizer), self.tokenizer.token_to_id["<eos>"])

    def test_17_vocabulary_mismatch_detection(self):
        """Verifies validate_against_vocab_size raises error on dimension mismatch."""
        with self.assertRaises(ValueError):
            self.tokenizer.validate_against_vocab_size(579)

        # Correct size passes
        self.tokenizer.validate_against_vocab_size(self.tokenizer.vocab_size)

    def test_18_trainer_validation_errors(self):
        """Verifies ByteLevelBPETrainer rejects invalid arguments and empty corpora."""
        with self.assertRaises(ValueError):
            # Target vocab smaller than 9 special tokens + 256 byte tokens = 265
            ByteLevelBPETrainer(target_vocab_size=100)

        trainer = ByteLevelBPETrainer(target_vocab_size=512)
        with self.assertRaises(ValueError):
            trainer.train_from_texts([])

        with self.assertRaises(ValueError):
            trainer.train_from_texts(["", "   ", "\n\t"])

    def test_19_multilingual_indic_scripts(self):
        """Verifies round-trip fidelity for Hindi and Telugu Indic scripts."""
        indic_samples = [
            "नमस्ते",
            "తెలుగు",
            "Aether supports हिन्दी (Hindi) and తెలుగు (Telugu) multilingual text processing.",
            "₹1000 - भारतीय रुपया",
        ]
        for sample in indic_samples:
            ids = self.tokenizer.encode(sample)
            self.assertGreater(len(ids), 0)
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, sample)

    def test_20_deterministic_hash_consistency(self):
        """Verifies hash generation is stable and deterministic."""
        hash1 = self.tokenizer.get_vocab_hash()
        hash2 = self.tokenizer.get_vocab_hash()
        self.assertEqual(hash1, hash2)
        self.assertTrue(hash1.startswith("sha256_") or len(hash1) == 64)


if __name__ == "__main__":
    unittest.main()
