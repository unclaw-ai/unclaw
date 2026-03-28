from __future__ import annotations

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry


def test_capability_context_includes_phase1a_verify_before_guess_guidance() -> None:
    summary = build_runtime_capability_summary(
        tool_registry=create_default_tool_registry(),
        memory_summary_available=False,
        model_can_call_tools=True,
    )

    context = build_runtime_capability_context(summary)

    assert "prefer the tool over guessing" in context
    assert (
        "Do not ask for clarification when the current request already gives enough "
        "information for the first obvious tool call." in context
    )
    assert "look something up online" in context
    assert "For obvious local date, time, day, OS, hostname, or machine-spec questions" in context
