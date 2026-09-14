"""
AETHER MODEL — Output Safety Guard
Inspects generated text outputs for policy violations or unsafe content prior to returning to clients.
"""

from typing import Tuple

class OutputGuard:
    def __init__(self):
        self.blocked_output_tokens = [
            "<unsafe_payload>",
            "UNAUTHORIZED_EXFILTRATION_TOKEN"
        ]

    def validate(self, text: str) -> Tuple[bool, str]:
        """
        Validates whether generated text output passes safety constraints.
        Returns: (is_safe, sanitized_text)
        """
        if not text:
            return True, ""

        sanitized = text
        for token in self.blocked_output_tokens:
            if token in sanitized:
                sanitized = sanitized.replace(token, "[REDACTED]")

        return True, sanitized
