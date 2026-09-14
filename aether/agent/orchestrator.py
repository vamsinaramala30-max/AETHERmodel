"""
AETHER — Agent Orchestrator (Research & Model Evaluation Harness)

ARCHITECTURAL BOUNDARY NOTICE:
-------------------------------
Application-level AI orchestration is authoritative in AETHER_CORE (AETHER_BAC).
This Python orchestrator loop is scoped to model evaluation, research benchmarking,
and standalone testing. It must NOT be treated as the production application orchestrator
or own application task/project/workspace database state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import AsyncIterator, Optional, TYPE_CHECKING

from aether.agent.context import build_context
from aether.agent.errors import AgentError, safe_call
from aether.agent.executor import run_step
from aether.agent.planner import classify_intent, create_plan, revise_plan
from aether.agent.state import (
    AgentEvent, AgentState, ContextBundle, IntentClass,
    Plan, PlanStep, Stage, ToolResult, make_event,
)
from aether.agent.verifier import check as verify_result

if TYPE_CHECKING:
    from aether.agent.tool_registry import ToolRegistry
    from aether.core.llama_engine import LlamaCppEngine
    from aether.core.memory import MemoryStore
    from aether.core.retrieval import RetrievalEngine

logger = logging.getLogger("aether.agent.orchestrator")


# ---------------------------------------------------------------------------
# JSON tool-call parsing (migrated from agent_loop.py, improved)
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)
_BARE_JSON_RE = re.compile(
    r'(\{[^{}]*"tool"[^{}]*\})',
    re.DOTALL,
)


def parse_tool_call(model_output: str) -> Optional[dict]:
    """
    Extract a tool call dict from model output.

    Tries (in order):
      1. Fenced code block containing JSON with "tool" key.
      2. Bare JSON object containing "tool" key.
      3. Entire output parsed as JSON.
      4. Returns None if nothing parseable found.
    """
    # Try fenced block
    m = _JSON_FENCE_RE.search(model_output)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try bare JSON
    m2 = _BARE_JSON_RE.search(model_output)
    if m2:
        try:
            parsed = json.loads(m2.group(1))
            if "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try full output as JSON
    try:
        parsed = json.loads(model_output.strip())
        if "tool" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Model generation helper (async, wraps sync llama engine)
# ---------------------------------------------------------------------------

async def _generate_async(
    engine: "LlamaCppEngine",
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    request_id: str = "",
) -> str:
    """Run sync generate() in a thread pool so we don't block the event loop."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=request_id,
        )
    )
    if not result.success:
        raise AgentError(
            stage=Stage.VERIFICATION,
            user_message="The model failed to generate a response.",
            cause=Exception(result.error or "generation failed"),
        )
    return result.text


async def _stream_tokens(
    engine: "LlamaCppEngine",
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
    request_id: str = "",
):
    """Yield tokens from llama.cpp stream_generate() via a thread executor."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run_stream():
        try:
            for chunk in engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=request_id,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc), "done": True})

    # Run in thread pool — doesn't block the event loop
    asyncio.get_event_loop().run_in_executor(None, _run_stream)

    while True:
        chunk = await queue.get()
        yield chunk
        if chunk.get("done"):
            break


# ---------------------------------------------------------------------------
# Main orchestrator run()
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Stateless orchestrator — all state lives in AgentState, passed explicitly.
    One instance is fine for the lifetime of the server; it holds no per-request state.
    """

    def __init__(
        self,
        engine: "LlamaCppEngine",
        registry: "ToolRegistry",
        memory_store: Optional["MemoryStore"] = None,
        retrieval_engine: Optional["RetrievalEngine"] = None,
        max_steps: int = 5,
        context_token_budget: int = 2048,
        generation_timeout_s: float = 120.0,
        tool_timeout_s: float = 30.0,
    ):
        self.engine = engine
        self.registry = registry
        self.memory_store = memory_store
        self.retrieval_engine = retrieval_engine
        self.max_steps = max_steps
        self.context_token_budget = context_token_budget
        self.generation_timeout_s = generation_timeout_s
        self.tool_timeout_s = tool_timeout_s

    async def run(self, state: AgentState) -> AsyncIterator[AgentEvent]:
        """
        Run the full agentic loop for one request.

        Yields AgentEvents as they occur. Final response is stored in
        state.final_response. Errors are yielded as ERROR events, not raised.

        Usage:
            async for event in orchestrator.run(state):
                # send event to client via SSE
        """
        return self._run_impl(state)

    async def _run_impl(self, state: AgentState) -> AsyncIterator[AgentEvent]:
        rid = state.request_id

        try:
            # ── UNDERSTANDING ──────────────────────────────────────────
            state.stage = Stage.UNDERSTANDING
            yield make_event(Stage.UNDERSTANDING, rid)

            intent = classify_intent(state.user_input)
            state.intent = intent
            logger.info(f"[{rid}] Intent: {intent.value}")

            # ── CONTEXT LOADING ────────────────────────────────────────
            state.stage = Stage.CONTEXT_LOADING
            yield make_event(Stage.CONTEXT_LOADING, rid)

            tool_schema_text = self.registry.tool_schemas_for_prompt()
            context = await safe_call(
                build_context(
                    state=state,
                    token_budget=self.context_token_budget,
                    engine=self.engine,
                    memory_store=self.memory_store,
                    retrieval_engine=self.retrieval_engine,
                    tool_schema_text=tool_schema_text if intent == IntentClass.TOOL_USE else "",
                    max_memories=5,
                    max_knowledge_chunks=3,
                ),
                stage=Stage.CONTEXT_LOADING,
                user_message="Failed to load context.",
                timeout_s=15.0,
            )
            state.context = context

            # ── TRIVIAL SHORT-CIRCUIT ──────────────────────────────────
            if intent == IntentClass.TRIVIAL:
                state.stage = Stage.VERIFICATION
                yield make_event(Stage.VERIFICATION, rid)
                prompt = context.to_prompt_string()
                final_text = await safe_call(
                    _generate_async(self.engine, prompt, temperature=0.5, request_id=f"{rid}_trivial"),
                    stage=Stage.VERIFICATION,
                    user_message="Failed to generate a response.",
                    timeout_s=self.generation_timeout_s,
                )
                state.final_response = final_text
                state.stage = Stage.COMPLETED
                yield make_event(Stage.COMPLETED, rid)
                return

            # ── PLANNING ───────────────────────────────────────────────
            state.stage = Stage.PLANNING
            yield make_event(Stage.PLANNING, rid)
            available_tools = [s.name for s in self.registry.list_tools()]
            plan = create_plan(state, available_tools)
            state.plan = plan

            # ── TOOL-CALL LOOP ─────────────────────────────────────────
            # For TOOL_USE: run the rule-based plan first, then let model
            # decide if more steps are needed. For GENERAL: go straight to
            # model-driven loop with max_steps iterations.
            steps_taken = 0

            # Execute rule-based plan steps (if any)
            for step in plan.steps:
                if steps_taken >= self.max_steps:
                    break
                steps_taken += 1

                state.stage = Stage.TOOL_STARTED
                yield make_event(Stage.TOOL_STARTED, rid, tool_name=step.tool_name)

                tool_result = await run_step(
                    step, state, self.registry, timeout_s=self.tool_timeout_s
                )
                state.tool_results.append(tool_result)
                state.observations.append(tool_result.as_observation())

                state.stage = Stage.TOOL_COMPLETED
                yield make_event(
                    Stage.TOOL_COMPLETED, rid,
                    tool_name=step.tool_name,
                    result_summary=tool_result.summary(),
                )

                # Verify result and check if we should stop or replan
                verdict = verify_result(state, step, tool_result)
                if verdict.needs_replan:
                    plan = revise_plan(state, verdict.reason, available_tools)
                    state.plan = plan
                if verdict.should_stop:
                    break

            # Model-driven tool-call loop (up to remaining max_steps)
            # Rebuild context with observations included
            context = await build_context(
                state=state,
                token_budget=self.context_token_budget,
                engine=self.engine,
                memory_store=self.memory_store,
                retrieval_engine=self.retrieval_engine,
                tool_schema_text=tool_schema_text,
            )
            state.context = context
            prompt = context.to_prompt_string()

            while steps_taken < self.max_steps:
                steps_taken += 1

                raw_output = await safe_call(
                    _generate_async(
                        self.engine, prompt,
                        max_tokens=512,
                        temperature=0.1,  # low temp for structured output
                        request_id=f"{rid}_step{steps_taken}",
                    ),
                    stage=Stage.TOOL_STARTED,
                    user_message="The model failed to respond.",
                    timeout_s=self.generation_timeout_s,
                )

                tool_call = parse_tool_call(raw_output)

                if tool_call is None:
                    # Model produced prose → treat as final answer
                    logger.info(f"[{rid}] No tool call detected, treating as final answer")
                    state.final_response = raw_output
                    break

                tool_name = tool_call.get("tool", "")
                tool_args = tool_call.get("arguments", {})

                if tool_name == "final_answer":
                    state.final_response = tool_args.get("answer", raw_output)
                    logger.info(f"[{rid}] Model called final_answer at step {steps_taken}")
                    break

                # Execute model-requested tool
                state.stage = Stage.TOOL_STARTED
                yield make_event(Stage.TOOL_STARTED, rid, tool_name=tool_name)

                step = PlanStep(tool_name=tool_name, tool_args=tool_args)
                tool_result = await run_step(
                    step, state, self.registry, timeout_s=self.tool_timeout_s
                )
                state.tool_results.append(tool_result)
                obs = tool_result.as_observation()
                state.observations.append(obs)

                state.stage = Stage.TOOL_COMPLETED
                yield make_event(
                    Stage.TOOL_COMPLETED, rid,
                    tool_name=tool_name,
                    result_summary=tool_result.summary(),
                )

                verdict = verify_result(state, step, tool_result)
                if verdict.should_stop:
                    break

                # Feed observation back into prompt for next step
                prompt = f"{prompt}\n\n{obs}\nContinue reasoning or call final_answer."

            # ── VERIFICATION / FINAL GENERATION ───────────────────────
            state.stage = Stage.VERIFICATION
            yield make_event(Stage.VERIFICATION, rid)

            if state.final_response is None:
                # Reached max_steps or loop exited without a final answer → generate
                context = await build_context(
                    state=state,
                    token_budget=self.context_token_budget,
                    engine=self.engine,
                    memory_store=self.memory_store,
                    retrieval_engine=self.retrieval_engine,
                    tool_schema_text="",  # no tools in final gen
                )
                final_prompt = context.to_prompt_string()
                state.final_response = await safe_call(
                    _generate_async(
                        self.engine, final_prompt,
                        temperature=0.3,
                        request_id=f"{rid}_final",
                    ),
                    stage=Stage.VERIFICATION,
                    user_message="Failed to generate a final response.",
                    timeout_s=self.generation_timeout_s,
                )

            state.stage = Stage.COMPLETED
            yield make_event(Stage.COMPLETED, rid)

        except AgentError as exc:
            logger.error(
                f"[{rid}] AgentError at stage={exc.stage.value}: {exc}",
                exc_info=True,
            )
            state.error = exc.user_message
            state.stage = Stage.ERROR
            yield make_event(Stage.ERROR, rid, error_message=exc.user_message)

        except Exception as exc:
            # Catch-all — should not happen, but ensures the stream always closes cleanly
            logger.critical(
                f"[{rid}] Unhandled exception in orchestrator: {exc}",
                exc_info=True,
            )
            user_msg = "An unexpected error occurred. Please try again."
            state.error = user_msg
            state.stage = Stage.ERROR
            yield make_event(Stage.ERROR, rid, error_message=user_msg)
