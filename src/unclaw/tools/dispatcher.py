"""Resolve and execute registered tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unclaw.tools.contracts import (
    ToolArgumentDefinition,
    ToolCall,
    ToolResult,
    resolve_tool_argument_spec,
)
from unclaw.tools.registry import ToolRegistry


_GENERIC_ARGUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "command": ("cmd",),
    "content": ("text", "contents", "body"),
    "destination_path": ("destination", "destination_file", "dest", "to"),
    "path": ("file_path", "filepath", "filename"),
    "query": ("q", "search", "search_query", "search_term"),
    "source_path": ("source", "source_file", "src", "from"),
    "url": ("link", "href"),
}
_TOOL_SPECIFIC_ARGUMENT_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "write_text_file": {
        "content": ("text", "contents", "body"),
    },
    "fetch_url_text": {
        "url": ("link", "href"),
    },
    "fast_web_search": {
        "query": ("q", "search", "search_query"),
    },
    "search_web": {
        "query": ("q", "search", "search_query"),
    },
}


def _coerce_argument(value: Any, expected_type: str) -> Any:
    """Coerce simple primitive argument types emitted by LLMs.

    LLMs frequently emit numbers as strings ("5" instead of 5).
    This function converts safe primitive types before tool execution.
    """

    if expected_type in {"int", "integer"}:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass

    if expected_type in {"float", "number"}:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass

    if expected_type in {"bool", "boolean"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False

    # default: return unchanged
    return value


def _coerce_arguments(
    arguments: dict[str, Any],
    schema: dict[str, ToolArgumentDefinition],
) -> dict[str, Any]:
    """Apply type coercion based on tool argument schema."""
    coerced: dict[str, Any] = {}

    for name, value in arguments.items():
        raw_spec = schema.get(name)
        if raw_spec is not None:
            expected_type = resolve_tool_argument_spec(raw_spec).value_type
            coerced[name] = _coerce_argument(value, expected_type)
        else:
            coerced[name] = value

    return coerced


def _repair_obvious_argument_aliases(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, ToolArgumentDefinition],
) -> dict[str, Any]:
    """Recover safe near-miss argument names like ``text`` -> ``content``.

    Recovery is intentionally conservative:
    - only when the target schema key exists and is currently missing
    - only when exactly one obvious alias is present
    - never when the alias is itself a declared schema key
    """
    if not arguments or not schema:
        return dict(arguments)

    repaired = dict(arguments)
    declared_keys = set(schema)
    tool_specific_aliases = _TOOL_SPECIFIC_ARGUMENT_ALIASES.get(tool_name, {})

    for target_name in declared_keys:
        if target_name in repaired:
            continue

        candidate_aliases = tool_specific_aliases.get(
            target_name,
            _GENERIC_ARGUMENT_ALIASES.get(target_name, ()),
        )
        matching_aliases = [
            alias
            for alias in candidate_aliases
            if alias in repaired and alias not in declared_keys
        ]
        if len(matching_aliases) != 1:
            continue

        alias_name = matching_aliases[0]
        repaired[target_name] = repaired.pop(alias_name)

    return repaired


def normalize_tool_call_for_execution(
    registry: ToolRegistry,
    call: ToolCall,
) -> ToolCall:
    """Return a repaired + type-coerced tool call ready for execution."""
    registered_tool = registry.get(call.tool_name)
    if registered_tool is None:
        return call

    schema = dict(registered_tool.definition.arguments)
    repaired_arguments = _repair_obvious_argument_aliases(
        tool_name=call.tool_name,
        arguments=call.arguments,
        schema=schema,
    )
    coerced_arguments = _coerce_arguments(repaired_arguments, schema)
    return ToolCall(
        tool_name=call.tool_name,
        arguments=coerced_arguments,
    )


@dataclass(slots=True)
class ToolDispatcher:
    """Execute one tool call against a registry."""

    registry: ToolRegistry

    def dispatch(self, call: ToolCall) -> ToolResult:
        registered_tool = self.registry.get(call.tool_name)
        if registered_tool is None:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Unknown tool '{call.tool_name}'.",
            )

        # -----------------------------
        # Type coercion layer
        # -----------------------------
        try:
            call = normalize_tool_call_for_execution(
                self.registry,
                call,
            )

        except Exception as exc:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Failed to process tool arguments: {exc}",
            )

        # -----------------------------
        # Tool execution
        # -----------------------------
        try:
            result = registered_tool.handler(call)

        except (ValueError, OSError) as exc:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Tool '{call.tool_name}' failed: {exc}",
            )

        except Exception as exc:  # Defensive boundary for unexpected tool crashes.
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Tool '{call.tool_name}' failed unexpectedly: {exc}",
            )

        if not isinstance(result, ToolResult):
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=f"Tool '{call.tool_name}' returned an invalid result object.",
            )

        if result.tool_name != call.tool_name:
            return ToolResult.failure(
                tool_name=call.tool_name,
                error=(
                    f"Tool '{call.tool_name}' returned a mismatched result "
                    f"for '{result.tool_name}'."
                ),
            )

        return result


__all__ = ["ToolDispatcher", "normalize_tool_call_for_execution"]
