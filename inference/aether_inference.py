"""
AETHER MODEL — Inference Engine Wrapper
Delegates to src/inference/engine.py
"""

import sys
import os

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inference.engine import AetherInferenceEngine

__all__ = ["AetherInferenceEngine"]
