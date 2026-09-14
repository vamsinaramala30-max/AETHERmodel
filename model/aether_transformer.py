"""
AETHER MODEL — Transformer Architecture & Knowledge Matrix
Wrapper delegating to src/model/model.py
"""

import sys
import os

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from model.model import AetherModel as AetherTransformerModel

__all__ = ["AetherTransformerModel"]
