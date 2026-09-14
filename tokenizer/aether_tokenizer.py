"""
AETHER MODEL — Tokenizer Wrapper
Delegates to src/tokenizer/tokenizer.py
"""

import sys
import os

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tokenizer.tokenizer import AetherTokenizer, SPECIAL_TOKENS

__all__ = ["AetherTokenizer", "SPECIAL_TOKENS"]
