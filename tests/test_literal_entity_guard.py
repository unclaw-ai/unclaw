from __future__ import annotations

from unclaw.tools.contracts import ToolCall
from unclaw.tools.web_entity_guard import (
    apply_entity_guard_to_tool_calls,
    extract_user_entity_surface,
)


def _make_search_call(tool_name: str, query: str) -> ToolCall:
    return ToolCall(tool_name=tool_name, arguments={"query": query})


def test_fast_web_search_preserves_explicit_literal_entity() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("fast_web_search", "Marine Le Pen biographie")],
        "Marine Leleu",
    )

    assert guarded[0].arguments["query"] == "Marine Leleu biographie"


def test_search_web_preserves_explicit_literal_entity() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("search_web", "Marine Le Pen recent profile")],
        "Marine Leleu",
    )

    assert guarded[0].arguments["query"] == "Marine Leleu recent profile"


def test_guard_is_noop_when_query_already_uses_literal_entity() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("fast_web_search", "Marine Leleu biographie")],
        "Marine Leleu",
    )

    assert guarded[0].arguments["query"] == "Marine Leleu biographie"


def test_guard_does_not_mutate_without_explicit_literal_entity_in_current_input() -> None:
    user_input = "can you search the web for the latest updates on climate change and renewable energy"
    user_entity_surface = extract_user_entity_surface(user_input)

    assert user_entity_surface == ""

    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("search_web", "Marine Le Pen")],
        user_entity_surface,
    )

    assert guarded[0].arguments["query"] == "Marine Le Pen"


def test_explicit_literal_surface_wins_over_trailing_noise_tolerance() -> None:
    guarded = apply_entity_guard_to_tool_calls(
        [_make_search_call("fast_web_search", "Inoxtag")],
        "Inoxtag N",
    )

    assert guarded[0].arguments["query"] == "Inoxtag N"
