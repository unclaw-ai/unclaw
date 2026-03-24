from __future__ import annotations

import pytest

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.routing import _build_request_routing_note

pytestmark = pytest.mark.unit


def _summary(**overrides) -> RuntimeCapabilitySummary:
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


def test_routing_note_routes_direct_public_url_fetch_requests() -> None:
    note = _build_request_routing_note(
        user_input="Open https://example.com and summarize it.",
        capability_summary=_summary(
            available_builtin_tool_names=("fetch_url_text",),
            url_fetch_available=True,
        ),
    )

    assert note is not None
    assert "specific public URL" in note
    assert "fetch_url_text first" in note


def test_routing_note_routes_french_identity_requests_to_fast_web_search() -> None:
    note = _build_request_routing_note(
        user_input="Qui est Marine Leleu ?",
        capability_summary=_summary(
            available_builtin_tool_names=("fast_web_search", "search_web"),
            web_search_available=True,
            fast_web_search_available=True,
        ),
    )

    assert note is not None
    assert "fast_web_search" in note
    assert "exact entity wording from the user" in note


def test_routing_note_routes_french_deep_research_to_search_web() -> None:
    note = _build_request_routing_note(
        user_input="Fais une bio complete de Marine Leleu.",
        capability_summary=_summary(
            available_builtin_tool_names=("fast_web_search", "search_web"),
            web_search_available=True,
            fast_web_search_available=True,
        ),
    )

    assert note is not None
    assert "Call search_web" in note
    assert "Do not stop at fast_web_search alone" in note


def test_routing_note_keeps_duo_entities_grouped() -> None:
    note = _build_request_routing_note(
        user_input="Qui sont McFly et Carlito ?",
        capability_summary=_summary(
            available_builtin_tool_names=("fast_web_search", "search_web"),
            web_search_available=True,
            fast_web_search_available=True,
        ),
    )

    assert note is not None
    assert "duo or joint entity biography lookup" in note
    assert "Do not separate the duo" in note


def test_routing_note_routes_explicit_terminal_requests() -> None:
    note = _build_request_routing_note(
        user_input="Run `ls -la` in the terminal.",
        capability_summary=_summary(
            available_builtin_tool_names=("run_terminal_command",),
            shell_command_execution_available=True,
        ),
    )

    assert note is not None
    assert "explicit local shell or terminal request" in note
    assert "run_terminal_command" in note


def test_routing_note_routes_system_info_requests() -> None:
    note = _build_request_routing_note(
        user_input="What is the local date and time on this machine?",
        capability_summary=_summary(
            available_builtin_tool_names=("system_info",),
            system_info_available=True,
        ),
    )

    assert note is not None
    assert "obvious local machine or runtime question" in note
    assert "system_info now" in note


def test_routing_note_routes_directory_listing_requests() -> None:
    note = _build_request_routing_note(
        user_input="List the contents of src/unclaw/core directory.",
        capability_summary=_summary(
            available_builtin_tool_names=("list_directory", "read_text_file"),
            local_directory_listing_available=True,
            local_file_read_available=True,
        ),
    )

    assert note is not None
    assert "explicit local directory inspection request" in note
    assert "list_directory" in note
    assert "read that file next" in note


def test_routing_note_routes_file_read_requests() -> None:
    note = _build_request_routing_note(
        user_input="Read src/unclaw/core/runtime.py",
        capability_summary=_summary(
            available_builtin_tool_names=("read_text_file", "list_directory"),
            local_file_read_available=True,
            local_directory_listing_available=True,
        ),
    )

    assert note is not None
    assert "explicit local file inspection request" in note
    assert "read_text_file" in note


def test_routing_note_returns_none_without_any_routing_signal() -> None:
    note = _build_request_routing_note(
        user_input="Explain recursion in simple terms.",
        capability_summary=_summary(),
    )

    assert note is None


def test_routing_note_returns_none_when_matching_tool_is_unavailable() -> None:
    note = _build_request_routing_note(
        user_input="What is the local date and time on this machine?",
        capability_summary=_summary(),
    )

    assert note is None


def test_routing_note_handles_accents_via_normalized_text() -> None:
    note = _build_request_routing_note(
        user_input="Fais une recherche détaillée sur Joséphine Baker.",
        capability_summary=_summary(
            available_builtin_tool_names=("search_web",),
            web_search_available=True,
        ),
    )

    assert note is not None
    assert "Call search_web" in note
    assert "exact query terms" in note


def test_routing_note_preserves_current_limitation_wording_when_file_read_is_unavailable() -> None:
    note = _build_request_routing_note(
        user_input="Read src/unclaw/core/runtime.py",
        capability_summary=_summary(
            available_builtin_tool_names=("list_directory",),
            local_directory_listing_available=True,
        ),
    )

    assert note is not None
    assert "direct file read is unavailable" in note
    assert "Start with list_directory" in note
