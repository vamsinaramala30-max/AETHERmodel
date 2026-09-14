"""
AETHER MODEL — Safety Engine Wrapper
Delegates input and output safety checks to src/safety/
"""

import sys
import os

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from safety.input_guard import InputGuard
from safety.output_guard import OutputGuard

class AetherSafetyEngine:
    def __init__(self):
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()

    def validate_input(self, prompt: str):
        return self.input_guard.validate(prompt)

    def validate_output(self, text: str):
        return self.output_guard.validate(text)

__all__ = ["AetherSafetyEngine"]
