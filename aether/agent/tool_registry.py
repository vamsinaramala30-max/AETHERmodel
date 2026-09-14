"""
AETHER — Tool Registry (Phase 6)

Replaces the plain dict[str, Callable] used in agent_loop.py.

Key properties:
- Every tool has a JSON Schema for its inputs — validated BEFORE execution.
- Permissions: READ_ONLY | WRITE | DESTRUCTIVE.
  Destructive tools require args["_confirmed"] = True or are rejected.
- No shell/eval/arbitrary code tools — ever.
- calculator and datetime are pure Python built-ins with no external state.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import inspect
import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("aether.agent.tool_registry")

# ---------------------------------------------------------------------------
# Permission enum
# ---------------------------------------------------------------------------

class Permission(str, Enum):
    READ_ONLY   = "read_only"
    WRITE       = "write"
    DESTRUCTIVE = "destructive"  # Requires explicit _confirmed=True in args


# ---------------------------------------------------------------------------
# ToolResult (lightweight return type for registry)
# ---------------------------------------------------------------------------

@dataclass
class ToolCallResult:
    """Typed result from ToolRegistry.execute()."""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None

    def as_observation(self) -> str:
        if self.success:
            try:
                data_text = json.dumps(self.data, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                data_text = str(self.data)
            return f"[Tool: {self.tool_name}]\nResult: {data_text}"
        return f"[Tool: {self.tool_name}]\nError: {self.error}"


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """
    Full specification for a registered tool.

    Args:
        name:          Unique tool name (used by the model and executor).
        description:   One-sentence description shown to the model in the system prompt.
        input_schema:  JSON Schema dict for the tool's arguments.
        output_schema: JSON Schema dict for the tool's return value (informational).
        permission:    READ_ONLY | WRITE | DESTRUCTIVE.
        handler:       The callable that executes the tool.
                       May be sync (wrapped automatically) or async.
    """
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    permission: Permission
    handler: Callable[..., Any]
    tags: list[str] = field(default_factory=list)  # e.g. ["tasks", "read"]


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Central registry for all agent tools.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolSpec(...))
        result = await registry.execute("add_task", {"title": "Buy milk"})
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Raises ValueError if name is already taken."""
        if spec.name in self._specs:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec
        logger.info(f"Tool registered: '{spec.name}' ({spec.permission.value})")

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._specs.pop(name, None)

    def get(self, name: str) -> ToolSpec:
        """Look up a tool by name. Raises KeyError if not found."""
        if name not in self._specs:
            available = list(self._specs.keys())
            raise KeyError(f"Unknown tool: '{name}'. Available: {available}")
        return self._specs[name]

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tools (for system prompt generation)."""
        return list(self._specs.values())

    def tool_schemas_for_prompt(self) -> str:
        """Serialise all tool specs as JSON for injection into the system prompt."""
        schemas = [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.input_schema,
            }
            for s in self._specs.values()
        ]
        return json.dumps(schemas, indent=2)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, name: str, args: dict) -> ToolCallResult:
        """
        Execute a tool by name with the given args.

        Steps:
          1. Look up the tool spec (KeyError → error result).
          2. Validate args against input_schema (ValidationError → error result).
          3. Check DESTRUCTIVE permission gate.
          4. Call the handler (sync or async).

        Returns a ToolCallResult — never raises.
        """
        # 1. Look up
        try:
            spec = self.get(name)
        except KeyError as exc:
            return ToolCallResult(tool_name=name, success=False, error=str(exc))

        # 2. Schema validation
        try:
            _validate_schema(args, spec.input_schema, tool_name=name)
        except ValueError as exc:
            logger.warning(f"Tool '{name}' schema validation failed: {exc}")
            return ToolCallResult(tool_name=name, success=False, error=str(exc))

        # 3. Destructive gate
        if spec.permission == Permission.DESTRUCTIVE:
            if not args.get("_confirmed"):
                msg = (
                    f"Tool '{name}' is destructive and requires '_confirmed: true' in args. "
                    "Pass '_confirmed: true' to proceed."
                )
                logger.warning(msg)
                return ToolCallResult(tool_name=name, success=False, error=msg)

        # 4. Execute
        try:
            if inspect.iscoroutinefunction(spec.handler):
                data = await spec.handler(**args)
            else:
                # Run sync handlers in thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(None, lambda: spec.handler(**args))
            logger.info(f"Tool '{name}' executed successfully")
            return ToolCallResult(tool_name=name, success=True, data=data)
        except Exception as exc:
            logger.error(f"Tool '{name}' handler raised: {exc}", exc_info=True)
            return ToolCallResult(tool_name=name, success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Schema validation (minimal jsonschema without the dependency)
# ---------------------------------------------------------------------------

def _validate_schema(args: dict, schema: dict, tool_name: str = "") -> None:
    """
    Validate args against a JSON Schema subset.
    Raises ValueError with a clear message on failure.

    Supports: type, properties, required, enum.
    Full jsonschema validation can replace this if jsonschema is available.
    """
    try:
        import jsonschema  # type: ignore[import]
        jsonschema.validate(args, schema)
        return
    except ImportError:
        pass  # fall back to lightweight manual validation
    except Exception as exc:
        raise ValueError(f"Tool '{tool_name}' args validation failed: {exc}") from exc

    # Lightweight fallback
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for req_field in required:
        if req_field == "_confirmed":
            continue  # this is a permission field, not a model-provided arg
        if req_field not in args:
            raise ValueError(
                f"Tool '{tool_name}' missing required arg: '{req_field}'. "
                f"Required: {required}"
            )

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, value in args.items():
        if key == "_confirmed":
            continue
        prop_schema = properties.get(key, {})
        expected_type_str = prop_schema.get("type")
        if expected_type_str and expected_type_str in type_map:
            expected_type = type_map[expected_type_str]
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Tool '{tool_name}' arg '{key}': expected {expected_type_str}, "
                    f"got {type(value).__name__}"
                )
        enum_vals = prop_schema.get("enum")
        if enum_vals is not None and value not in enum_vals:
            raise ValueError(
                f"Tool '{tool_name}' arg '{key}': must be one of {enum_vals}, got '{value}'"
            )


# ---------------------------------------------------------------------------
# Built-in tools: calculator + datetime (pure Python, no external state)
# ---------------------------------------------------------------------------

_SAFE_CALC_RE = re.compile(r"^[0-9+\-*/().%\s]+$")
_SAFE_FUNCS = {
    "sqrt": math.sqrt, "pow": pow, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil, "log": math.log,
    "pi": math.pi, "e": math.e,
}


def _calculator_handler(expression: str) -> dict:
    """
    Evaluate a safe numeric expression. No exec/eval on arbitrary strings.
    Only allows: digits, operators (+,-,*,/,%), parentheses, whitespace,
    and a small set of math functions.
    """
    expression = expression.strip()
    # Check for safe characters first (basic gate)
    if not _SAFE_CALC_RE.match(expression):
        # Allow named math functions too
        safe_names = "|".join(_SAFE_FUNCS.keys())
        if not re.match(rf"^[0-9+\-*/().%\s{safe_names},]+$", expression):
            raise ValueError(f"Unsafe expression: '{expression}'")
    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_FUNCS)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        raise ValueError(f"Could not evaluate '{expression}': {exc}") from exc


def _datetime_handler(query: str = "now", timezone: str = "UTC") -> dict:
    """Return current date/time. Pure Python, no external calls."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return {
        "query": query,
        "utc_iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": "UTC",
    }


CALCULATOR_SPEC = ToolSpec(
    name="calculator",
    description="Evaluate a safe numeric math expression. Supports +,-,*,/,%, sqrt, pow, abs, round, floor, ceil, log, pi, e.",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 * (3 + 4)' or 'sqrt(16)'",
            }
        },
        "required": ["expression"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "result": {"type": "number"},
        },
    },
    permission=Permission.READ_ONLY,
    handler=_calculator_handler,
    tags=["math", "utility"],
)

DATETIME_SPEC = ToolSpec(
    name="datetime",
    description="Get the current date, time, and day of the week in UTC.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to return: 'now', 'date', 'time', or 'day'",
                "enum": ["now", "date", "time", "day"],
            }
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "utc_iso": {"type": "string"},
            "date": {"type": "string"},
            "time": {"type": "string"},
            "day_of_week": {"type": "string"},
        },
    },
    permission=Permission.READ_ONLY,
    handler=_datetime_handler,
    tags=["datetime", "utility"],
)


def build_default_registry() -> ToolRegistry:
    """
    Create a ToolRegistry pre-loaded with the built-in pure tools.
    Task tools are registered separately (they need user_id binding).
    """
    registry = ToolRegistry()
    registry.register(CALCULATOR_SPEC)
    registry.register(DATETIME_SPEC)
    return registry
