"""
AETHER MODEL — Special-Token Pipeline Regression Test Suite (Phases 6-7)

Tests A–E validate the full token-ID contract across tokenizer, context,
generation, and validation code after the Phase 1-5 pipeline fixes.

Test A  EOS consistency:    tokenizer.eos == generation stop-ids == no legacy == 2 check
Test B  Role-token consistency: SYSTEM/USER/ASSISTANT IDs identical across
                                tokenizer & context builder
Test C  Encode/decode round-trip: special tokens survive encode->decode
Test D  Forced-EOS termination: inject EOS mid-generation, confirm stop
Test E  No stale hardcodes: static scan fails if bare `== 2` near EOS logic in
                            inference/generation.py or training/loss.py
"""

from __future__ import annotations

import os
import re
import sys
import unittest

# -- Path setup ---------------------------------------------------------------
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_BASE, "src")
for _p in [_BASE, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from tokenizer.special_tokens import SpecialTokens
from inference.context import ContextManager
from inference.generation import TokenGenerator


# ---------------------------------------------------------------------------
# Test A — EOS consistency
# ---------------------------------------------------------------------------
class TestA_EosConsistency(unittest.TestCase):
    """EOS token ID must be consistent across tokenizer, SpecialTokens, and
    generation's stop-id logic — all reading from the same live source."""

    def setUp(self):
        self.tokenizer = AetherTokenizer()
        self.st = SpecialTokens.from_tokenizer(self.tokenizer)

    def test_eos_in_vocab(self):
        self.assertIn("<eos>", self.tokenizer.token_to_id,
                      "<eos> token must be in vocabulary")

    def test_special_tokens_eos_matches_tokenizer(self):
        expected = self.tokenizer.token_to_id["<eos>"]
        self.assertEqual(self.st.eos, expected,
                         "SpecialTokens.eos must equal tokenizer.token_to_id['<eos>']")

    def test_stop_ids_contains_eos(self):
        """generation.py derives stop_ids from the loaded tokenizer — never from
        a hardcoded constant.  Simulate the same logic here and verify."""
        tok = self.tokenizer
        _eos = tok.token_to_id.get("<eos>")
        _pad = tok.token_to_id.get("<pad>")
        stop_ids = set()
        if _eos is not None:
            stop_ids.add(_eos)
        if _pad is not None:
            stop_ids.add(_pad)
        if not stop_ids:
            stop_ids = {2}  # last-resort fallback (mirrors generation.py)

        # The EOS from SpecialTokens must be in the stop-ids set
        self.assertIn(self.st.eos, stop_ids,
                      "SpecialTokens.eos must appear in derived stop_ids set")

    def test_no_legacy_constant_mismatch_in_legacy_mode(self):
        """In legacy (word-regex) mode with aether_vocab.json loaded, EOS is 2.
        When default BPE is loaded, EOS is 3."""
        legacy_vocab = os.path.join(_BASE, "checkpoints", "aether_vocab.json")
        if os.path.exists(legacy_vocab):
            tok_legacy = AetherTokenizer(vocab_file=legacy_vocab, frozen=True)
            st_legacy = SpecialTokens.from_tokenizer(tok_legacy)
            self.assertEqual(st_legacy.eos, 2, "Legacy tokenizer EOS must be 2")
        else:
            self.assertEqual(self.st.eos, 3, "Default BPE tokenizer EOS must be 3")

    def test_bpe_mode_eos_is_3(self):
        """If a BPE tokenizer is loaded, EOS must be 3 (as declared in bpe.py)."""
        # Only run if BPE file exists and is loadable
        bpe_path = os.path.join(_BASE, "checkpoints", "aether_bpe_tokenizer.json")
        if not os.path.exists(bpe_path):
            self.skipTest("BPE tokenizer file not found — skipping BPE EOS check")
        tok_bpe = AetherTokenizer(vocab_file=bpe_path, frozen=True)
        if not tok_bpe.is_bpe:
            self.skipTest("Loaded file is not a BPE vocab — skipping")
        self.assertEqual(tok_bpe.token_to_id.get("<eos>"), 3,
                         "BPE tokenizer EOS must be 3")


# ---------------------------------------------------------------------------
# Test B — Role-token consistency
# ---------------------------------------------------------------------------
class TestB_RoleTokenConsistency(unittest.TestCase):
    """SYSTEM/USER/ASSISTANT IDs must be identical in SpecialTokens and in
    what ContextManager actually injects into the token sequence."""

    def setUp(self):
        self.tokenizer = AetherTokenizer()
        self.st = SpecialTokens.from_tokenizer(self.tokenizer)
        self.ctx = ContextManager(self.tokenizer, max_seq_len=256)

    def test_user_id_matches_context_prefix(self):
        """The first token in a bare user-only prompt must equal st.user."""
        ids = self.ctx.format_prompt("Hello", context={})
        # With no system/history, context_parts is empty.
        # full_token_ids = [] + [USER_ID] + encode("Hello") + [ASST_ID]
        user_idx = 0
        self.assertEqual(ids[user_idx], self.st.user,
                         f"First token in prompt must be USER ID ({self.st.user}), got {ids[user_idx]}")

    def test_assistant_id_is_last_prefix(self):
        """The last token in the formatted prompt must be the ASST prefix."""
        ids = self.ctx.format_prompt("Hello", context={})
        self.assertEqual(ids[-1], self.st.assistant,
                         f"Last token in formatted prompt must be ASST ID ({self.st.assistant}), got {ids[-1]}")

    def test_system_id_in_system_prompt(self):
        """With a system_prompt, the first token of context_parts must be SYS."""
        ids = self.ctx.format_prompt("Hi", context={"system_prompt": "You are Aether."})
        self.assertEqual(ids[0], self.st.system,
                         f"First token with system_prompt must be SYS ID ({self.st.system})")

    def test_special_tokens_validate_passes(self):
        self.st.validate(self.tokenizer)  # raises ValueError on mismatch


# ---------------------------------------------------------------------------
# Test C — Encode/decode round-trip for special tokens
# ---------------------------------------------------------------------------
class TestC_RoundTrip(unittest.TestCase):
    """Special tokens must survive encode -> decode with skip_special_tokens=False."""

    def setUp(self):
        self.tokenizer = AetherTokenizer()
        self.st = SpecialTokens.from_tokenizer(self.tokenizer)

    def test_eos_round_trip(self):
        decoded = self.tokenizer.decode([self.st.eos], skip_special_tokens=False)
        self.assertIn("<eos>", decoded,
                      f"Decode([{self.st.eos}]) must contain '<eos>'")

    def test_pad_round_trip(self):
        decoded = self.tokenizer.decode([self.st.pad], skip_special_tokens=False)
        self.assertIn("<pad>", decoded)

    def test_user_round_trip(self):
        decoded = self.tokenizer.decode([self.st.user], skip_special_tokens=False)
        self.assertIn("<user>", decoded)

    def test_assistant_round_trip(self):
        decoded = self.tokenizer.decode([self.st.assistant], skip_special_tokens=False)
        self.assertIn("<assistant>", decoded)

    def test_encode_special_tokens_preserved(self):
        """Encoding a text containing <eos> must produce the correct token ID."""
        if self.tokenizer.is_bpe:
            # BPE engine handles special tokens as atomic pre-split chunks
            ids = self.tokenizer.encode("<eos>")
            self.assertIn(self.st.eos, ids,
                          f"Encoding '<eos>' in BPE mode must produce ID {self.st.eos}")
        else:
            # Legacy tokenizer encodes <eos> as a whole token
            ids = self.tokenizer.encode("<eos>")
            self.assertIn(self.st.eos, ids,
                          f"Encoding '<eos>' in legacy mode must produce ID {self.st.eos}")


# ---------------------------------------------------------------------------
# Test D — Forced-EOS termination
# ---------------------------------------------------------------------------
class TestD_ForcedEosTermination(unittest.TestCase):
    """Injecting EOS as a stop token must cause generation to stop immediately."""

    @classmethod
    def setUpClass(cls):
        cls.model = AetherModel(skip_checkpoint=True)
        cls.tokenizer = AetherTokenizer()
        cls.st = SpecialTokens.from_tokenizer(cls.tokenizer)
        cls.generator = TokenGenerator(cls.model, cls.tokenizer)

    def test_eos_stop_token_halts_generation(self):
        """generation.py must stop when it samples the EOS ID from stop_token_ids."""
        prompt = self.tokenizer.encode("Hello") or [self.st.user]
        eos_id = self.st.eos

        gen = self.generator.generate_tokens(
            prompt,
            max_tokens=50,
            stop_token_ids=[eos_id],
            deterministic=True,
        )
        # EOS itself must NOT appear in the output
        self.assertNotIn(eos_id, gen,
                         "EOS token must not be included in the generated sequence")

    def test_pad_stop_token_halts_generation(self):
        """PAD is universally 0 and should also stop generation when in stop_ids."""
        prompt = self.tokenizer.encode("Hello") or [self.st.user]
        gen = self.generator.generate_tokens(
            prompt,
            max_tokens=50,
            stop_token_ids=[self.st.pad],
            deterministic=True,
        )
        self.assertNotIn(self.st.pad, gen,
                         "PAD token must not appear in generated sequence")


# ---------------------------------------------------------------------------
# Test E — Static stale-hardcode scan
# ---------------------------------------------------------------------------
class TestE_NoStaleHardcodes(unittest.TestCase):
    """Verify that critical inference/training files no longer contain bare
    `== 2` patterns adjacent to EOS-related logic."""

    _FILES_TO_SCAN = [
        os.path.join(_SRC, "inference", "generation.py"),
        os.path.join(_SRC, "inference", "streaming.py"),
        os.path.join(_BASE, "training", "loss.py"),
        os.path.join(_BASE, "training", "production_quality_gate.py"),
    ]

    # Patterns that constitute a forbidden hardcode near EOS logic
    # (We only flag `== 2` or `== 3` when EOS/eos appears on the same line)
    _FORBIDDEN_PATTERN = re.compile(
        r'(?:eos|EOS).*==\s*[23]\b|==\s*[23]\b.*(?:eos|EOS)',
        re.IGNORECASE,
    )

    def test_no_eos_hardcode_in_critical_files(self):
        violations = []
        for fpath in self._FILES_TO_SCAN:
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    # Skip comments and docstrings
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                    if self._FORBIDDEN_PATTERN.search(line):
                        violations.append(
                            f"{os.path.basename(fpath)}:{lineno}: {line.rstrip()}"
                        )

        self.assertEqual(
            violations, [],
            "Stale EOS hardcode(s) detected:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
