"""
AETHER MODEL — Input Safety Guard
Inspects incoming prompt requests for injection attempts, malicious payload patterns, and safety policy violations.
"""

import re
from typing import Tuple

class InputGuard:
    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"bypass\s+(safety|security)\s*(filters|checks|controls)?",
            r"system\s+override",
            r"override\s+(all\s+)?safety",
            r"reveal\s+secret\s+keys",
            r"dump\s+(all\s+)?(secret\s+)?(passwords|tokens|credentials|database)",
            r"export\s+(all\s+)?(workspace\s+)?(credentials|passwords|tokens)",
            r"delete\s+all\s+user\s+accounts",
            r"jailbreak"
        ]

    def validate(self, prompt: str) -> Tuple[bool, str, str]:
        """
        Validates whether prompt is safe to process.
        Returns: (is_safe, failure_message, risk_category)
        """
        if not prompt or not prompt.strip():
            return True, "Empty prompt", "none"

        p_lower = prompt.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, p_lower):
                return False, "Prompt injection pattern detected", "injection"

        return True, "Safe", "none"
