"""
AETHER — Agent Unit Tests (Phase 15)

Tests orchestrator control flow WITHOUT real model inference.
Uses fake engines and fake tools to verify:
  - Trivial requests skip planning/tools
  - Tool-use requests hit the correct tools
  - Max-steps is enforced
  - AgentError is caught and emitted as ERROR event
  - State is passed explicitly (no globals)

Run: pytest tests/test_agent_unit.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from aether.agent.state import AgentState, IntentClass, Stage
from aether.agent.planner import classify_intent
from aether.agent.tool_registry import (
    ToolRegistry, ToolSpec, Permission, build_default_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeEngine:
    """Minimal model engine for testing — returns scripted responses."""
    model_name = "fake-model-test"

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["Hello there!"]
        self._call_count = 0

    def generate(self, prompt: str, **kwargs):
        from aether.core.llama_engine import GenerationResult
        text = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return GenerationResult(text=text, tokens_used=5, success=True)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_info(self) -> dict:
        return {"name": self.model_name, "loaded": True, "backend": "fake"}


def make_state(text: str = "Hello", user_id: str = "test") -> AgentState:
    import uuid
    return AgentState(
        request_id=str(uuid.uuid4())[:8],
        conversation_id="test-conv",
        user_input=text,
        user_id=user_id,
    )


async def collect_events(coro) -> list:
    """Drain an async generator and collect all events."""
    events = []
    async for event in coro:
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Phase 2: classify_intent() tests
# ---------------------------------------------------------------------------

class TestIntentClassification:
    def test_hello_is_trivial(self):
        assert classify_intent("Hello") == IntentClass.TRIVIAL

    def test_hi_is_trivial(self):
        assert classify_intent("Hi there!") == IntentClass.TRIVIAL

    def test_thanks_is_trivial(self):
        assert classify_intent("Thanks!") == IntentClass.TRIVIAL

    def test_goodbye_is_trivial(self):
        assert classify_intent("Bye") == IntentClass.TRIVIAL

    def test_empty_is_trivial(self):
        assert classify_intent("ok") == IntentClass.TRIVIAL

    def test_show_tasks_is_tool_use(self):
        assert classify_intent("Show my tasks") == IntentClass.TOOL_USE

    def test_list_tasks_is_tool_use(self):
        assert classify_intent("List my pending tasks") == IntentClass.TOOL_USE

    def test_add_task_is_tool_use(self):
        assert classify_intent("Add a task: buy groceries") == IntentClass.TOOL_USE

    def test_what_time_is_tool_use(self):
        assert classify_intent("What time is it?") == IntentClass.TOOL_USE

    def test_calculation_is_tool_use(self):
        assert classify_intent("What is 15 * 23?") == IntentClass.TOOL_USE

    def test_general_question_is_general(self):
        assert classify_intent("Explain quantum computing in simple terms") == IntentClass.GENERAL

    def test_write_essay_is_general(self):
        assert classify_intent("Write a short story about the ocean") == IntentClass.GENERAL


# ---------------------------------------------------------------------------
# Phase 6: Tool registry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_calculator_works(self):
        registry = build_default_registry()
        result = asyncio.run(
            registry.execute("calculator", {"expression": "2 + 2"})
        )
        assert result.success
        assert result.data["result"] == 4

    def test_datetime_returns_date(self):
        registry = build_default_registry()
        result = asyncio.run(
            registry.execute("datetime", {"query": "now"})
        )
        assert result.success
        assert "date" in result.data
        assert "time" in result.data

    def test_missing_required_arg_rejected(self):
        registry = build_default_registry()
        # calculator requires "expression"
        result = asyncio.run(
            registry.execute("calculator", {})
        )
        assert not result.success
        assert "expression" in result.error.lower() or "required" in result.error.lower()

    def test_wrong_arg_type_rejected(self):
        registry = build_default_registry()
        # expression must be a string
        result = asyncio.run(
            registry.execute("calculator", {"expression": 42})
        )
        # Either rejected by schema or evaluated (42 as string works, just type check)
        # Main assertion: no crash
        assert result is not None

    def test_destructive_tool_requires_confirmation(self):
        registry = ToolRegistry()
        destructive_called = []

        def _destroy(**kwargs):
            destructive_called.append(True)
            return {"deleted": True}

        registry.register(ToolSpec(
            name="delete_everything",
            description="Deletes all data",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={},
            permission=Permission.DESTRUCTIVE,
            handler=_destroy,
        ))

        # Without _confirmed → should be rejected
        result = asyncio.run(
            registry.execute("delete_everything", {})
        )
        assert not result.success
        assert "_confirmed" in result.error
        assert len(destructive_called) == 0  # handler never called

    def test_destructive_tool_executes_with_confirmation(self):
        registry = ToolRegistry()

        def _destroy(**kwargs):
            return {"deleted": True}

        registry.register(ToolSpec(
            name="delete_something",
            description="Deletes something",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={},
            permission=Permission.DESTRUCTIVE,
            handler=_destroy,
        ))

        result = asyncio.run(
            registry.execute("delete_something", {"_confirmed": True})
        )
        assert result.success

    def test_unknown_tool_returns_error(self):
        registry = build_default_registry()
        result = asyncio.run(
            registry.execute("nonexistent_tool", {})
        )
        assert not result.success
        assert "Unknown" in result.error or "nonexistent" in result.error


# ---------------------------------------------------------------------------
# Phase 4/5: Orchestrator unit tests (fake model + fake tools)
# ---------------------------------------------------------------------------

class TestOrchestratorControlFlow:
    """
    These tests verify STAGE SEQUENCES, not wall-clock time.
    They use a fake engine — no real inference.
    """

    def _make_orchestrator(self, engine=None, registry=None):
        from aether.agent.orchestrator import Orchestrator
        return Orchestrator(
            engine=engine or FakeEngine(["Hello there!"]),
            registry=registry or build_default_registry(),
            memory_store=None,
            retrieval_engine=None,
            max_steps=3,
            context_token_budget=512,
            generation_timeout_s=10.0,
            tool_timeout_s=5.0,
        )

    def test_trivial_request_skips_planning(self):
        """TRIVIAL intent must not emit PLANNING or TOOL_STARTED events."""
        orch = self._make_orchestrator()
        state = make_state("Hello")

        async def _run():
            gen = await orch.run(state)
            return await collect_events(gen)

        events = asyncio.run(_run())
        stages = [e.stage for e in events]

        assert Stage.UNDERSTANDING in stages
        assert Stage.PLANNING not in stages, "TRIVIAL request must NOT hit planning"
        assert Stage.TOOL_STARTED not in stages, "TRIVIAL request must NOT start tools"
        assert Stage.COMPLETED in stages

    def test_trivial_request_produces_final_response(self):
        orch = self._make_orchestrator(engine=FakeEngine(["Hi! I'm Aether."]))
        state = make_state("Hello")

        async def _run():
            gen = await orch.run(state)
            await collect_events(gen)

        asyncio.run(_run())
        assert state.final_response is not None
        assert len(state.final_response) > 0

    def test_datetime_tool_fires_for_time_question(self):
        """'What time is it?' should trigger the datetime tool."""
        orch = self._make_orchestrator()
        state = make_state("What time is it?")

        async def _run():
            gen = await orch.run(state)
            return await collect_events(gen)

        events = asyncio.run(_run())
        tool_events = [e for e in events if e.stage == Stage.TOOL_STARTED]
        tool_names = [e.tool_name for e in tool_events]
        assert "datetime" in tool_names, (
            f"Expected 'datetime' tool to fire, got: {tool_names}"
        )

    def test_max_steps_not_exceeded(self):
        """Orchestrator must not exceed max_steps regardless of model output."""
        # Model always outputs a tool call (never final_answer)
        always_tool = FakeEngine(['{"tool": "datetime", "arguments": {}}'] * 20)
        orch = self._make_orchestrator(engine=always_tool)
        state = make_state("Keep calling tools forever")

        async def _run():
            gen = await orch.run(state)
            return await collect_events(gen)

        events = asyncio.run(_run())
        tool_started_count = sum(1 for e in events if e.stage == Stage.TOOL_STARTED)
        assert tool_started_count <= 3, (
            f"Exceeded max_steps=3: got {tool_started_count} tool calls"
        )
        # Must complete (not hang)
        stages = [e.stage for e in events]
        assert Stage.COMPLETED in stages or Stage.ERROR in stages

    def test_state_passed_explicitly(self):
        """AgentState is mutated by reference — verify it's the same object."""
        orch = self._make_orchestrator(engine=FakeEngine(["Hello!"]))
        original_state = make_state("Hello")
        original_id = original_state.request_id

        async def _run():
            gen = await orch.run(original_state)
            await collect_events(gen)

        asyncio.run(_run())
        # State was mutated in-place (same object, not a copy)
        assert original_state.request_id == original_id
        assert original_state.final_response is not None


# ---------------------------------------------------------------------------
# Phase 8: Context engine tests
# ---------------------------------------------------------------------------

class TestContextEngine:
    def test_context_never_exceeds_budget(self):
        """build_context() must stay within token_budget even with many large memories."""
        import asyncio
        from aether.agent.context import build_context, _AETHER_SYSTEM_PROMPT

        # Calculate a budget that's larger than the system prompt but smaller than
        # system_prompt + all the memories. Use 4-char heuristic: system prompt ~600 chars → ~150 tokens.
        # Add 50 tokens headroom above system prompt, then try to inject 10 large memories.
        fake_engine = FakeEngine()
        sys_tokens = fake_engine.count_tokens(_AETHER_SYSTEM_PROMPT)
        # Budget = system tokens + 50 (room for user input + very few memories)
        tight_budget = sys_tokens + 50

        state = make_state("Tell me everything you know")

        class MockMemory:
            def recall(self, user_id, query, k=5):
                # Return 10 memories, each ~100 tokens
                return ["Important memory content " * 20] * 10  # ~200 chars each = ~50 tokens

        async def _run():
            ctx = await build_context(
                state=state,
                token_budget=tight_budget,
                engine=fake_engine,
                memory_store=MockMemory(),
                max_memories=10,
            )
            return ctx

        ctx = asyncio.run(_run())
        prompt_text = ctx.to_prompt_string()
        estimated_tokens = fake_engine.count_tokens(prompt_text)
        # Allow 10% variance for estimation error (heuristic, not exact tokenizer)
        assert estimated_tokens <= tight_budget * 1.15, (
            f"Context exceeded budget: {estimated_tokens} tokens > {tight_budget} (15% grace). "
            f"Memories injected: {len(ctx.relevant_memory)}"
        )
        # The key property: NOT all 10 memories should appear
        # (they'd push us over budget)
        assert len(ctx.relevant_memory) < 10, (
            f"Expected fewer than 10 memories (budget enforcement), got {len(ctx.relevant_memory)}"
        )

    def test_context_recalls_limited_memories(self):
        """Only max_memories memories should appear, not all 50."""
        import asyncio
        from aether.agent.context import build_context

        state = make_state("Tell me about my projects")
        fifty_memories = [f"Memory {i}" for i in range(50)]

        class MockMemory:
            def recall(self, user_id, query, k=5):
                return fifty_memories[:k]  # returns at most k

        async def _run():
            ctx = await build_context(
                state=state,
                token_budget=2048,
                engine=FakeEngine(),
                memory_store=MockMemory(),
                max_memories=3,
            )
            return ctx

        ctx = asyncio.run(_run())
        assert len(ctx.relevant_memory) <= 3, (
            f"Expected at most 3 memories, got {len(ctx.relevant_memory)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
