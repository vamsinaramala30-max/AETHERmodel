"""
AETHER — Agentic Tool-Use Loop (Phase 5)

Design guarantees:
1. max_steps is a hard cap — the loop CANNOT run forever.
2. Every exit path returns a typed dict with a "status" field.
3. Unknown tool names are an explicit error, not silently ignored.
4. Tool exceptions are caught and reported as typed errors.
5. The model is asked to emit structured JSON for tool calls — fallback
   heuristics handle models that produce imperfect JSON.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Callable, Optional

from aether.core.model_engine import ModelEngine, GenerationResult

logger = logging.getLogger("aether.agent")

# ---------------------------------------------------------------------------
# Tool schemas (shown to the model as part of the system prompt)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "add_task",
        "description": "Add a task to the user's task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The task title."},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Task priority level.",
                },
                "notes": {"type": "string", "description": "Optional additional notes."},
            },
            "required": ["title"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Search the knowledge base for information relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "k": {"type": "integer", "description": "Number of results to return (1-5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember_fact",
        "description": "Store an important fact or user preference for future recall.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember."},
            },
            "required": ["fact"],
        },
    },
    {
        "name": "final_answer",
        "description": "Provide the final answer to the user. Use this when you have enough information.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The final answer text."},
            },
            "required": ["answer"],
        },
    },
]

_TOOL_SCHEMA_TEXT = json.dumps(TOOL_SCHEMAS, indent=2)

_AGENT_SYSTEM_PROMPT = f"""You are Aether, an intelligent assistant with access to tools.

When you need to use a tool, respond with ONLY a JSON object in this exact format:
```json
{{"tool": "<tool_name>", "arguments": {{...}}}}
```

Available tools:
{_TOOL_SCHEMA_TEXT}

When you have gathered enough information to answer, call the "final_answer" tool.
Do NOT generate text outside the JSON block when calling a tool.
If no tool is needed, call "final_answer" directly with your response.
"""


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

# Match ```json ... ``` or bare JSON objects
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```|(\{[^{}]*\"tool\"[^{}]*\})",
    re.DOTALL,
)


def parse_tool_call(model_output: str) -> Optional[dict]:
    """
    Extract a tool call dict from model output.

    Tries:
    1. Fenced code block with JSON.
    2. Bare JSON object containing "tool" key.
    3. Returns None if no tool call is found (treat as final answer prose).
    """
    # Try fenced block first
    m = _JSON_BLOCK_RE.search(model_output)
    if m:
        raw = m.group(1) or m.group(2)
        try:
            parsed = json.loads(raw)
            if "tool" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse entire output as JSON
    try:
        parsed = json.loads(model_output.strip())
        if "tool" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Bounded agentic reasoning loop.

    Guarantees:
    - Cannot exceed max_steps iterations.
    - Returns a typed status dict at every exit path.
    - Tool exceptions are caught and returned as error states — never raised.
    - Unknown tool names produce an explicit "error" status.

    Args:
        model_engine: ModelEngine instance for generation.
        tools:        Dict mapping tool name → callable.
                      Each callable receives **kwargs from tool_call["arguments"].
        max_steps:    Hard cap on loop iterations.
    """

    def __init__(
        self,
        model_engine: ModelEngine,
        tools: dict[str, Callable],
        max_steps: int = 5,
    ):
        self.model = model_engine
        self.tools = tools
        self.max_steps = max_steps

    def run(
        self,
        user_input: str,
        user_id: str = "default",
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run the agent loop for a user input.

        Returns a dict with:
            status: "done" | "error"
            response: str (final answer, present when status == "done")
            detail: str (error description, present when status == "error")
            steps_taken: int
            tool_calls: list of {tool, arguments, result} dicts
        """
        rid = request_id or str(uuid.uuid4())[:8]
        logger.info(f"[{rid}] AgentLoop.run() | user='{user_id}' | max_steps={self.max_steps}")

        # Build system-aware prompt
        current_prompt = f"{_AGENT_SYSTEM_PROMPT}\n\nUser: {user_input}"
        tool_call_log: list[dict] = []

        for step in range(self.max_steps):
            logger.info(f"[{rid}] Step {step + 1}/{self.max_steps}")

            # Generate
            result: GenerationResult = self.model.generate(
                prompt=current_prompt,
                max_tokens=512,
                temperature=0.1,  # low temperature for structured output
                request_id=f"{rid}_s{step}",
            )

            if not result.success:
                logger.error(f"[{rid}] Generation failed at step {step + 1}: {result.error}")
                return {
                    "status": "error",
                    "detail": f"Generation failed: {result.error}",
                    "steps_taken": step + 1,
                    "tool_calls": tool_call_log,
                }

            raw_output = result.text

            # Try to parse a tool call
            tool_call = parse_tool_call(raw_output)

            if tool_call is None:
                # Model produced prose without a tool call — treat as final answer
                logger.info(f"[{rid}] No tool call detected; treating as final answer")
                return {
                    "status": "done",
                    "response": raw_output,
                    "steps_taken": step + 1,
                    "tool_calls": tool_call_log,
                }

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("arguments", {})

            # final_answer is a pseudo-tool — return immediately
            if tool_name == "final_answer":
                answer = tool_args.get("answer", raw_output)
                logger.info(f"[{rid}] Agent called final_answer at step {step + 1}")
                return {
                    "status": "done",
                    "response": answer,
                    "steps_taken": step + 1,
                    "tool_calls": tool_call_log,
                }

            # Validate tool exists
            tool_fn = self.tools.get(tool_name)
            if tool_fn is None:
                logger.warning(f"[{rid}] Unknown tool requested: '{tool_name}'")
                return {
                    "status": "error",
                    "detail": f"Unknown tool: '{tool_name}'. Available: {list(self.tools.keys())}",
                    "steps_taken": step + 1,
                    "tool_calls": tool_call_log,
                }

            # Execute tool
            try:
                tool_result = tool_fn(**tool_args)
                logger.info(
                    f"[{rid}] Tool '{tool_name}' executed successfully | "
                    f"result_preview='{str(tool_result)[:80]}'"
                )
                tool_call_log.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                })
            except TypeError as exc:
                # Wrong arguments passed by model
                return {
                    "status": "error",
                    "detail": f"Tool '{tool_name}' called with invalid arguments: {exc}",
                    "steps_taken": step + 1,
                    "tool_calls": tool_call_log,
                }
            except Exception as exc:
                logger.error(f"[{rid}] Tool '{tool_name}' raised: {exc}", exc_info=True)
                return {
                    "status": "error",
                    "detail": f"Tool '{tool_name}' execution failed: {exc}",
                    "steps_taken": step + 1,
                    "tool_calls": tool_call_log,
                }

            # Feed tool result back into the next prompt
            current_prompt = (
                f"{current_prompt}\n\n"
                f"Tool '{tool_name}' result: {json.dumps(tool_result)}\n"
                f"Continue reasoning or call final_answer."
            )

        # Reached max_steps without a final_answer — explicit error, not silent loop
        logger.warning(f"[{rid}] max_steps={self.max_steps} exceeded without final_answer")
        return {
            "status": "error",
            "detail": f"Agent did not produce a final answer within {self.max_steps} steps.",
            "steps_taken": self.max_steps,
            "tool_calls": tool_call_log,
        }
