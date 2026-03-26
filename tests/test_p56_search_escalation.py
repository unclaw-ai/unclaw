"""Tests for structural routing fallback plus post-tool grounding guidance.

Proves:
- _build_request_routing_note routes only explicit structural hints
- semantic prompts without structural hints do not trigger deterministic retry notes
- _build_post_tool_grounding_note pushes search_web escalation after fast_web_search
- timeout/partial-failure note is included when a tool timed out
- capability context describes search_web as the deeper grounded path
"""

from __future__ import annotations

import pytest

from unclaw.core.capabilities import (
    RuntimeCapabilitySummary,
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.routing import (
    _build_request_routing_note,
    _looks_like_deep_search_request,
)
from unclaw.core.runtime_support import (
    _build_post_tool_grounding_note,
)
from unclaw.tools.contracts import ToolResult
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.unit


def _web_capable_summary(*, fast_web: bool = True) -> RuntimeCapabilitySummary:
    return RuntimeCapabilitySummary(
        available_builtin_tool_names=(
            ("fast_web_search", "search_web") if fast_web else ("search_web",)
        ),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=True,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
        fast_web_search_available=fast_web,
    )


def _thin_fast_web_result(query: str = "McFly et Carlito") -> ToolResult:
    return ToolResult.ok(
        tool_name="fast_web_search",
        output_text=f"- {query}: resultat rapide",
        payload={
            "query": query,
            "result_count": 1,
            "match_quality": "exact",
            "supported_point_count": 1,
            "grounding_note": f"- {query}: resultat rapide",
        },
    )


def _structural_summary(**overrides) -> RuntimeCapabilitySummary:
    payload = {
        "available_builtin_tool_names": (),
        "local_file_read_available": False,
        "local_directory_listing_available": False,
        "url_fetch_available": False,
        "web_search_available": False,
        "system_info_available": False,
        "memory_summary_available": False,
        "model_can_call_tools": True,
        "local_file_write_available": False,
        "session_history_recall_available": False,
        "long_term_memory_available": False,
        "shell_command_execution_available": False,
        "fast_web_search_available": False,
    }
    payload.update(overrides)
    return RuntimeCapabilitySummary(**payload)


def test_deep_search_classifier_is_disabled() -> None:
    for user_input in (
        "fais une biographie complete de Banksy",
        "tell me everything you know about them",
        "deep dive into this topic",
    ):
        assert _looks_like_deep_search_request(user_input) is False


def test_routing_note_routes_structural_url_request() -> None:
    note = _build_request_routing_note(
        user_input="https://example.com",
        capability_summary=_structural_summary(
            available_builtin_tool_names=("fetch_url_text",),
            url_fetch_available=True,
        ),
    )

    assert note is not None
    assert "specific public URL" in note
    assert "fetch_url_text" in note


def test_routing_note_routes_structural_terminal_request() -> None:
    note = _build_request_routing_note(
        user_input="`pwd`",
        capability_summary=_structural_summary(
            available_builtin_tool_names=("run_terminal_command",),
            shell_command_execution_available=True,
        ),
    )

    assert note is not None
    assert "explicit local shell or terminal request" in note
    assert "run_terminal_command" in note


def test_routing_note_routes_structural_directory_request() -> None:
    note = _build_request_routing_note(
        user_input="src/unclaw/core/",
        capability_summary=_structural_summary(
            available_builtin_tool_names=("list_directory", "read_text_file"),
            local_directory_listing_available=True,
            local_file_read_available=True,
        ),
    )

    assert note is not None
    assert "explicit local directory inspection request" in note
    assert "list_directory" in note


def test_routing_note_routes_structural_file_request() -> None:
    note = _build_request_routing_note(
        user_input="src/unclaw/core/runtime.py",
        capability_summary=_structural_summary(
            available_builtin_tool_names=("read_text_file", "list_directory"),
            local_file_read_available=True,
            local_directory_listing_available=True,
        ),
    )

    assert note is not None
    assert "explicit local file inspection request" in note
    assert "read_text_file" in note


@pytest.mark.parametrize(
    "user_input",
    [
        "qui est Marine Leleu",
        "who is Ada Lovelace",
        "McFly et Carlito, fais leur bio complete",
        "tell me everything you know about them",
        "fais une recherche complete",
    ],
)
def test_routing_note_returns_none_for_semantic_prompts_without_structural_hints(
    user_input: str,
) -> None:
    note = _build_request_routing_note(
        user_input=user_input,
        capability_summary=_web_capable_summary(),
    )
    assert note is None


def test_post_tool_note_names_search_web_as_deeper_path_after_fast_web() -> None:
    fast_result = _thin_fast_web_result()
    note = _build_post_tool_grounding_note(
        tool_results=[fast_result],
        tool_definitions=[FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION],
    )
    assert "search_web" in note
    assert any(
        word in note.lower()
        for word in ("deeper", "richer", "full", "complete", "bio", "research")
    )


def test_post_tool_note_explicitly_discourages_inflating_fast_web_to_full_bio() -> None:
    fast_result = _thin_fast_web_result()
    note = _build_post_tool_grounding_note(
        tool_results=[fast_result],
        tool_definitions=[FAST_WEB_SEARCH_DEFINITION, SEARCH_WEB_DEFINITION],
    )
    assert "fast_web_search" in note or "grounding" in note.lower()


def test_post_tool_note_includes_timeout_guidance_when_tool_timed_out() -> None:
    timed_out_result = ToolResult.failure(
        tool_name="search_web",
        error="search_web timed out for query 'McFly et Carlito'",
    )
    note = _build_post_tool_grounding_note(
        tool_results=[timed_out_result],
        tool_definitions=[SEARCH_WEB_DEFINITION],
    )
    assert any(word in note.lower() for word in ("timed out", "timeout", "partial"))


def test_post_tool_note_no_timeout_guidance_when_no_timeout() -> None:
    ok_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Some web results",
        payload={"query": "test", "evidence_count": 5, "finding_count": 3},
    )
    note = _build_post_tool_grounding_note(
        tool_results=[ok_result],
        tool_definitions=[SEARCH_WEB_DEFINITION],
    )
    assert "timed out" not in note.lower()


def test_capability_context_describes_search_web_as_deep_path() -> None:
    registry = create_default_tool_registry()
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    context = build_runtime_capability_context(summary)
    assert "search_web" in context
    assert "deeper" in context or "grounded" in context or "condenses" in context


def test_capability_context_includes_duo_joint_entity_guidance() -> None:
    registry = create_default_tool_registry()
    summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=True,
    )
    context = build_runtime_capability_context(summary)
    assert "duo" in context.lower() or "unit" in context.lower() or "together" in context.lower()


def test_routing_note_is_none_when_tools_not_callable() -> None:
    no_tool_summary = RuntimeCapabilitySummary(
        available_builtin_tool_names=(),
        local_file_read_available=False,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    note = _build_request_routing_note(
        user_input="fais une bio complete",
        capability_summary=no_tool_summary,
    )
    assert note is None
