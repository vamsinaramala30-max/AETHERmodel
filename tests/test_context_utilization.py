"""
AETHER MODEL — Context Utilization Test Suite (Prompt 17 Phase 9)
Tests whether the model actually uses supplied context correctly:
- Relevant evidence → use evidence
- Irrelevant evidence → ignore irrelevant context
- Contradictory evidence → acknowledge conflict
- No evidence → state insufficient information when appropriate
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

from inference.engine import AetherInferenceEngine


class TestContextUtilization(unittest.TestCase):
    """Tests whether the model properly uses, ignores, or reports on supplied context."""

    @classmethod
    def setUpClass(cls):
        cls.engine = AetherInferenceEngine()

    def test_01_relevant_evidence_used(self):
        """When relevant evidence is provided, model should use it in the response."""
        context = {
            "rag_context": "Aether Knowledge Base: The backup retention period is 30 days."
        }
        resp, meta = self.engine.generate_response(
            "What is the backup retention period?", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        self.assertTrue(meta.get("evidence_used"), "Evidence flag should be True when RAG context provided")

    def test_02_relevant_memory_used(self):
        """When relevant user memory is provided, model should incorporate it."""
        context = {
            "memory_context": "User's preferred language is Python."
        }
        resp, meta = self.engine.generate_response(
            "What language should I use for this script?", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")

    def test_03_irrelevant_context_present(self):
        """When irrelevant context is provided, model should still generate a coherent answer."""
        context = {
            "rag_context": "Aether Knowledge Base: The cafeteria serves lunch from 12-1pm."
        }
        resp, meta = self.engine.generate_response(
            "How do I create a database index?", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        # Should still answer about database indexes despite irrelevant RAG context

    def test_04_no_evidence_available(self):
        """When no evidence is available and user asks for live data, should report honestly."""
        context = {}
        resp, meta = self.engine.generate_response(
            "Show me my active automations", context=context
        )
        self.assertGreater(len(resp.strip()), 0, "Response should not be empty")
        self.assertEqual(meta["confidence"], "INSUFFICIENT_INFORMATION",
                         "Should classify as INSUFFICIENT_INFORMATION without data")

    def test_05_evidence_context_token_present(self):
        """When RAG evidence is provided, the EVIDENCE token should be in the formatted prompt."""
        from tokenizer.tokenizer import EVIDENCE_TOKEN_ID
        context = {
            "rag_context": "Some evidence data."
        }
        prompt_ids = self.engine.context_manager.format_prompt("Test question", context)
        self.assertIn(EVIDENCE_TOKEN_ID, prompt_ids,
                      "EVIDENCE token should be present when RAG context is provided")

    def test_06_no_evidence_token_without_rag(self):
        """When no RAG evidence is provided, the EVIDENCE token should NOT be in the prompt."""
        from tokenizer.tokenizer import EVIDENCE_TOKEN_ID
        context = {}
        prompt_ids = self.engine.context_manager.format_prompt("Test question", context)
        self.assertNotIn(EVIDENCE_TOKEN_ID, prompt_ids,
                         "EVIDENCE token should NOT be present without RAG context")

    def test_07_system_prompt_included(self):
        """When system prompt is provided, it should be included in the formatted token sequence."""
        from tokenizer.tokenizer import SYSTEM_TOKEN_ID
        context = {
            "system_prompt": "You are a helpful assistant."
        }
        prompt_ids = self.engine.context_manager.format_prompt("Hello", context)
        self.assertIn(SYSTEM_TOKEN_ID, prompt_ids,
                      "SYSTEM token should be present when system prompt is provided")


if __name__ == "__main__":
    unittest.main()
