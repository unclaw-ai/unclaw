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
        user_input="https://example.com",
        capability_summary=_summary(
            available_builtin_tool_names=("fetch_url_text",),
            url_fetch_available=True,
        ),
    )

    assert note is not None
    assert "specific public URL" in note
    assert "fetch_url_text first" in note


def test_routing_note_returns_none_for_identity_requests() -> None:
    note = _build_request_routing_note(
        user_input="Qui est Marine Leleu ?",
        capability_summary=_summary(
            available_builtin_tool_names=("fast_web_search", "search_web"),
            web_search_available=True,
            fast_web_search_available=True,
        ),
    )

    assert note is None


def test_routing_note_returns_none_for_deep_research_requests() -> None:
    note = _build_request_routing_note(
        user_input="Fais une bio complete de Marine Leleu.",
        capability_summary=_summary(
            available_builtin_tool_names=("fast_web_search", "search_web"),
            web_search_available=True,
            fast_web_search_available=True,
        ),
    )

    assert note is None


def test_routing_note_returns_none_for_joint_entity_requests() -> None:
    note = _build_request_routing_note(
        user_input="Qui sont McFly et Carlito ?",
        capability_summary=_summary(
            available_builtin_tool_names=("fast_web_search", "search_web"),
            web_search_available=True,
            fast_web_search_available=True,
        ),
    )

    assert note is None


def test_routing_note_routes_explicit_terminal_requests() -> None:
    note = _build_request_routing_note(
        user_input="`ls -la`",
        capability_summary=_summary(
            available_builtin_tool_names=("run_terminal_command",),
            shell_command_execution_available=True,
        ),
    )

    assert note is not None
    assert "explicit local shell or terminal request" in note
    assert "run_terminal_command" in note


def test_routing_note_routes_directory_listing_requests() -> None:
    note = _build_request_routing_note(
        user_input="src/unclaw/core/",
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
        user_input="src/unclaw/core/runtime.py",
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


@pytest.mark.parametrize(
    "user_input",
    [
        "What is the local date and time on this machine?",
        "Read the config file for me.",
        "List the current directory contents.",
        "Run a shell command to show my current directory.",
    ],
)
def test_routing_note_returns_none_for_lexical_requests_without_structural_hints(
    user_input: str,
) -> None:
    note = _build_request_routing_note(
        user_input=user_input,
        capability_summary=_summary(
            available_builtin_tool_names=(
                "system_info",
                "read_text_file",
                "list_directory",
                "run_terminal_command",
            ),
            system_info_available=True,
            local_file_read_available=True,
            local_directory_listing_available=True,
            shell_command_execution_available=True,
        ),
    )

    assert note is None


def test_routing_note_returns_none_when_matching_tool_is_unavailable() -> None:
    note = _build_request_routing_note(
        user_input="https://example.com",
        capability_summary=_summary(),
    )

    assert note is None


def test_routing_note_returns_none_for_explicit_web_lookup_requests() -> None:
    note = _build_request_routing_note(
        user_input="Search the web for the latest news about Marine Leleu.",
        capability_summary=_summary(
            available_builtin_tool_names=("search_web",),
            web_search_available=True,
        ),
    )

    assert note is None


def test_routing_note_handles_accents_via_normalized_text_without_web_retry() -> None:
    note = _build_request_routing_note(
        user_input="Fais une recherche détaillée sur Joséphine Baker.",
        capability_summary=_summary(
            available_builtin_tool_names=("search_web",),
            web_search_available=True,
        ),
    )

    assert note is None


def test_routing_note_preserves_current_limitation_wording_when_file_read_is_unavailable() -> None:
    note = _build_request_routing_note(
        user_input="src/unclaw/core/runtime.py",
        capability_summary=_summary(
            available_builtin_tool_names=("list_directory",),
            local_directory_listing_available=True,
        ),
    )

    assert note is not None
    assert "direct file read is unavailable" in note
    assert "Start with list_directory" in note
