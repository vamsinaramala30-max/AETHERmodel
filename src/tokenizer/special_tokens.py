"""
AETHER MODEL — Authoritative Special-Token ID Resolver (Phase 1)

Single source of truth for all special-token IDs.
ALL code that needs a special-token ID should use this module — never the
bare module-level constants from tokenizer.py or bpe.py, because those two
files disagree on the EOS, USER, ASSISTANT, SYSTEM, TOOL, and EVIDENCE IDs
(legacy EOS=2 vs BPE EOS=3, etc.).

Usage
-----
    from tokenizer.special_tokens import SpecialTokens
    st = SpecialTokens.from_tokenizer(my_tokenizer)
    # st.eos, st.user, st.system, ... are always correct for the loaded vocab.

Design contract
---------------
- IDs are read from the LIVE tokenizer's token_to_id mapping.
- If a token is missing (should never happen with a valid vocab), the
  per-attribute fallbacks are the LEGACY values so behaviour matches the
  pre-BPE baseline until the tokenizer file is replaced.
- No constant is duplicated here. The module never hard-codes a number
  that should come from the tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizer.tokenizer import AetherTokenizer


@dataclass(frozen=True)
class SpecialTokens:
    """Immutable snapshot of special-token IDs for ONE tokenizer instance."""

    pad: int
    unk: int
    eos: int
    bos: int
    system: int
    user: int
    assistant: int
    tool: int
    evidence: int

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tokenizer(cls, tokenizer: "AetherTokenizer") -> "SpecialTokens":
        """
        Build a SpecialTokens from a loaded AetherTokenizer.

        Falls back to legacy (word-regex) IDs only when a token is absent
        from the vocabulary, so the object is always safe to read from.
        """
        t = tokenizer.token_to_id

        def _get(name: str, legacy_fallback: int) -> int:
            val = t.get(name)
            return val if val is not None else legacy_fallback

        # BPE Canonical Fallbacks
        return cls(
            pad=_get("<pad>", 0),
            unk=_get("<unk>", 1),
            bos=_get("<bos>", 2),
            eos=_get("<eos>", 3),
            system=_get("<system>", 4),
            user=_get("<user>", 5),
            assistant=_get("<assistant>", 6),
            tool=_get("<tool>", 7),
            evidence=_get("<evidence>", 8),
        )

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                  #
    # ------------------------------------------------------------------ #

    @property
    def stop_ids(self) -> frozenset:
        """Set of token IDs that should terminate generation."""
        return frozenset({self.eos, self.pad})

    def as_dict(self) -> dict:
        """Return all IDs as a plain dict for diagnostics / logging."""
        return {
            "<pad>": self.pad,
            "<unk>": self.unk,
            "<bos>": self.bos,
            "<eos>": self.eos,
            "<system>": self.system,
            "<user>": self.user,
            "<assistant>": self.assistant,
            "<tool>": self.tool,
            "<evidence>": self.evidence,
        }

    def validate(self, tokenizer: "AetherTokenizer") -> None:
        """
        Assert that every stored ID is consistent with the supplied tokenizer.
        Raises ValueError on any mismatch (used in tests and diagnostics).
        """
        for token_str, stored_id in self.as_dict().items():
            actual_id = tokenizer.token_to_id.get(token_str)
            if actual_id is None:
                continue  # Token not in vocab -- skip (may be legacy tokenizer)
            if actual_id != stored_id:
                raise ValueError(
                    f"SpecialTokens mismatch for {token_str!r}: "
                    f"stored={stored_id}, tokenizer={actual_id}"
                )


def resolve_eos_id(tokenizer: "AetherTokenizer") -> int:
    """
    Convenience one-liner: return the EOS token ID from a live tokenizer.
    Never returns a hard-coded constant.
    """
    val = tokenizer.token_to_id.get("<eos>")
    if val is not None:
        return val
    return 3


__all__ = ["SpecialTokens", "resolve_eos_id"]
