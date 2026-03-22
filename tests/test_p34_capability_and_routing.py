"""Targeted tests for P3-4 — capability context and routing for local actions.

Proves:
- removed notes tools stay absent from the runtime capability context
- write_text_file appears in the capability context when it is registered
- write_text_file description explicitly states overwrite=false as the default
- system_info appears in the capability context (regression guard)
- Router system prompt covers local actions via the chat route
- Existing web_search routing behavior is not broken
- build_runtime_capability_summary populates local_file_write_available
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.capabilities import (
    RuntimeCapabilitySummary,
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.router import RouteKind, _ROUTER_SYSTEM_PROMPT, route_request
from unclaw.llm.base import LLMResponse
from unclaw.settings import load_settings
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_capability_summary(*, model_can_call_tools: bool = False) -> RuntimeCapabilitySummary:
    """Build a summary using the default tool registry."""
    registry = create_default_tool_registry()
    return build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=model_can_call_tools,
    )


def _empty_capability_summary() -> RuntimeCapabilitySummary:
    """Build a summary from an empty registry (no tools registered)."""
    registry = ToolRegistry()
    return build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )


# ---------------------------------------------------------------------------
# P3-4: local_file_write_available flag and removed notes runtime surface
# ---------------------------------------------------------------------------


def test_local_file_write_available_true_when_write_tool_registered() -> None:
    summary = _full_capability_summary()
    assert summary.local_file_write_available is True


def test_local_file_write_available_false_when_no_tools_registered() -> None:
    summary = _empty_capability_summary()
    assert summary.local_file_write_available is False


# ---------------------------------------------------------------------------
# P3-4: removed notes tools are no longer exposed
# ---------------------------------------------------------------------------


def test_default_registry_does_not_register_removed_notes_tools() -> None:
    registry = create_default_tool_registry()
    for name in ("create_note", "read_note", "list_notes", "update_note"):
        assert registry.get(name) is None


def test_capability_context_does_not_advertise_removed_notes_tools() -> None:
    for summary in (_full_capability_summary(), _empty_capability_summary()):
        context = build_runtime_capability_context(summary)
        assert "create_note" not in context
        assert "read_note" not in context
        assert "list_notes" not in context
        assert "update_note" not in context
        assert "Local notes" not in context


# ---------------------------------------------------------------------------
# P3-4: capability context includes write_text_file when available
# ---------------------------------------------------------------------------


def test_capability_context_includes_write_text_file_when_available() -> None:
    summary = _full_capability_summary()
    context = build_runtime_capability_context(summary)
    assert "write_text_file" in context


def test_capability_context_write_text_file_states_overwrite_false_default() -> None:
    """The write_text_file capability line must explicitly state overwrite=false default.

    This is the key P3-4 honesty requirement: the model must not assume overwrite=true
    is safe without explicit user intent.
    """
    summary = _full_capability_summary()
    context = build_runtime_capability_context(summary)
    # The context must mention that overwrite defaults to false
    assert "overwrite=false" in context
    # And that overwrite=true requires explicit user intent
    assert "explicitly" in context


def test_capability_context_write_text_file_omitted_from_tools_section_when_unavailable() -> None:
    summary = _empty_capability_summary()
    context = build_runtime_capability_context(summary)
    # listed as unavailable
    assert "write_text_file" in context
    assert "Local file write" in context


# ---------------------------------------------------------------------------
# P3-4: system_info still appears (regression guard)
# ---------------------------------------------------------------------------


def test_capability_context_includes_system_info_when_available() -> None:
    summary = _full_capability_summary()
    context = build_runtime_capability_context(summary)
    assert "system_info" in context


# ---------------------------------------------------------------------------
# P3-4: router system prompt covers local actions via chat route
# ---------------------------------------------------------------------------


def test_router_system_prompt_mentions_local_actions() -> None:
    """The router prompt must explicitly guide local actions to the chat route.

    This prevents the model from mistakenly routing system info or file-write
    requests to web_search.
    """
    assert "local actions" in _ROUTER_SYSTEM_PROMPT
    # Must include at least one concrete local action category
    assert any(
        term in _ROUTER_SYSTEM_PROMPT
        for term in ("system information", "file operations")
    )


def test_router_system_prompt_still_routes_web_search_for_online_facts() -> None:
    """Existing web_search guidance must not be removed."""
    assert "web_search" in _ROUTER_SYSTEM_PROMPT
    assert "current" in _ROUTER_SYSTEM_PROMPT or "externally verifiable" in _ROUTER_SYSTEM_PROMPT


def test_router_system_prompt_still_uses_chat_as_local_route() -> None:
    """chat route must remain the path for local reasoning."""
    assert "chat" in _ROUTER_SYSTEM_PROMPT
    assert "local reasoning" in _ROUTER_SYSTEM_PROMPT or "local actions" in _ROUTER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# P3-4: router routes local-action requests to chat (not web_search)
# ---------------------------------------------------------------------------


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])


def _make_fake_provider(route: str, search_query: str = ""):
    """Return a fake OllamaProvider that emits a fixed classification."""

    class _FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url: str = "", default_timeout_seconds: float = 60.0) -> None:
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del profile, timeout_seconds, thinking_enabled, content_callback, tools
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=f'{{"route":"{route}","search_query":"{search_query}"}}',
                created_at="2026-03-17T00:00:00Z",
                finish_reason="stop",
            )

    return _FakeOllamaProvider


def test_router_routes_system_info_request_to_chat(monkeypatch) -> None:
    """A system information request must stay on the chat route, not web_search."""
    settings = _load_repo_settings()
    registry = create_default_tool_registry(settings)
    capability_summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider("chat"),
    )
    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="What is my operating system and local time?",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )
    assert route.kind is RouteKind.CHAT
    assert route.search_query is None


def test_router_routes_sticky_note_style_request_to_chat(monkeypatch) -> None:
    """A sticky-note-style local action request must stay on chat, not web_search."""
    settings = _load_repo_settings()
    registry = create_default_tool_registry(settings)
    capability_summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider("chat"),
    )
    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="Create a sticky note on my desktop saying buy milk.",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )
    assert route.kind is RouteKind.CHAT
    assert route.search_query is None


def test_router_still_routes_online_research_to_web_search(monkeypatch) -> None:
    """Existing web_search routing behavior must not be broken."""
    settings = _load_repo_settings()
    registry = create_default_tool_registry(settings)
    capability_summary = build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider("web_search", "latest AI news"),
    )
    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="What is the latest news about AI today?",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )
    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query == "latest AI news"
