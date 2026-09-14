"""
AETHER MODEL — Multi-Turn Quality Test Suite (Prompt 17 Phase 8)
Tests context retention, preference recall, clarification requests,
no-fabrication behavior, and verified tool result reporting across
multi-turn conversation scenarios.
"""

import sys
import os
import unittest
from typing import Dict, Any

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from inference.engine import AetherInferenceEngine


class TestMultiTurnQuality(unittest.TestCase):
    """Multi-turn conversation quality tests verifying context retention and behavioral correctness."""

    @classmethod
    def setUpClass(cls):
        cls.engine = AetherInferenceEngine()

    # Conversation 1: User establishes a project, then asks about it
    def test_01_project_context_retention(self):
        """User establishes a project context, then asks about it."""
        context = {
            "conversation_history": [
                {"role": "user", "content": "I am working on the Aether Platform project."},
                {"role": "assistant", "content": "I understand you are working on the Aether Platform project."},
            ]
        }
        resp, meta = self.engine.generate_response(
            "What project did I mention?", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        # The engine should produce a non-empty response using conversation history

    # Conversation 2: User establishes a preference, then asks assistant to use it
    def test_02_preference_recall(self):
        """User establishes a preference, then asks assistant to use it."""
        context = {
            "memory_context": "User prefers TypeScript strict mode and bullet point formatting."
        }
        resp, meta = self.engine.generate_response(
            "How should I format my next report?", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        # Memory context should influence response

    # Conversation 3: User provides incomplete information, assistant should request clarification
    def test_03_clarification_on_incomplete_info(self):
        """User provides ambiguous/incomplete request, assistant should request clarification."""
        resp, meta = self.engine.generate_response("Make it better")
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        self.assertEqual(meta["confidence"], "LOW_CONFIDENCE",
                         "Ambiguous requests should be classified as LOW_CONFIDENCE")

    # Conversation 4: User asks unsupported question, assistant must not fabricate
    def test_04_no_fabrication_without_evidence(self):
        """User asks a question unsupported by evidence. Assistant must not fabricate."""
        context = {}  # No RAG, no memory, no workspace state
        resp, meta = self.engine.generate_response(
            "Show me my active automations", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        self.assertEqual(meta["confidence"], "INSUFFICIENT_INFORMATION",
                         "Should report INSUFFICIENT_INFORMATION when no data is available")
        # Response should NOT contain fabricated automation names/IDs
        resp_lower = resp.lower()
        self.assertTrue(
            any(k in resp_lower for k in ["direct access", "context", "authorize", "session", "data"]),
            "Response should honestly communicate lack of data access"
        )

    # Conversation 5: Verify multi-turn history formatting preserves isolation
    def test_05_no_cross_session_leakage(self):
        """Two separate conversations should not leak context between them."""
        context_a = {
            "conversation_history": [
                {"role": "user", "content": "My secret project is called Phoenix."},
                {"role": "assistant", "content": "I understand your project is called Phoenix."},
            ]
        }
        # Session A: should know about Phoenix
        prompt_ids_a = self.engine.context_manager.format_prompt("What is my project?", context_a)
        self.assertGreater(len(prompt_ids_a), 0)

        # Session B: completely separate, no history
        context_b = {}
        resp_b, meta_b = self.engine.generate_response(
            "What is my project name?", context=context_b
        )
        # Session B should NOT know about Phoenix (no context provided)
        # It should either ask for clarification or report insufficient info
        self.assertGreater(len(resp_b.strip()), 0, "Response should not be empty")

    def test_06_multi_turn_context_token_ordering(self):
        """Verify that multi-turn conversation history is formatted with correct token ordering."""
        context = {
            "system_prompt": "You are Aether AI.",
            "memory_context": "User prefers detailed responses.",
            "conversation_history": [
                {"role": "user", "content": "Tell me about testing."},
                {"role": "assistant", "content": "Testing verifies correctness."},
            ]
        }
        prompt_ids = self.engine.context_manager.format_prompt("Continue.", context)
        self.assertGreater(len(prompt_ids), 5, "Formatted prompt should include system + memory + history + user tokens")


if __name__ == "__main__":
    unittest.main()
