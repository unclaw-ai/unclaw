from __future__ import annotations

import json
from types import SimpleNamespace

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import FAST_WEB_SEARCH_DEFINITION, FETCH_URL_TEXT_DEFINITION


def _build_native_runtime(project_root, set_profile_tool_mode):
    settings = load_settings(project_root=project_root)
    set_profile_tool_mode(settings, "main", tool_mode="native")
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    return settings, session_manager, tracer, command_handler


def test_post_fast_web_search_clamps_substantive_reply(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T10:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marine Leleu"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {"query": "Marine Leleu"},
                                    }
                                }
                            ],
                        }
                    },
                )

            if call_count == 2:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=(
                        "Marine Leleu is a French endurance athlete, author, and podcast host. "
                        "She is also known for major speaking tours."
                    ),
                    created_at="2026-03-24T10:00:01Z",
                    finish_reason="stop",
                )

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=json.dumps(
                    {
                        "final_reply": (
                            "Marine Leleu is a French endurance athlete. "
                            "I couldn't confirm a fuller biography from that quick "
                            "grounding probe alone."
                        )
                    }
                ),
                created_at="2026-03-24T10:00:02Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Marine Leleu\n- Marine Leleu is a French endurance athlete.",
                payload={
                    "query": call.arguments["query"],
                    "result_count": 1,
                    "match_quality": "exact",
                    "supported_point_count": 1,
                    "grounding_note": (
                        "Marine Leleu\n- Marine Leleu is a French endurance athlete."
                    ),
            },
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Tell me everything you know about Marine Leleu.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Tell me everything you know about Marine Leleu.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "Marine Leleu is a French endurance athlete. "
            "I couldn't confirm a fuller biography from that quick grounding probe alone."
        )
        assert call_count == 3
    finally:
        session_manager.close()


def test_corrected_ambiguity_is_no_longer_recentered_by_runtime_heuristics(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            system_messages = [
                message.content for message in messages if message.role is LLMRole.SYSTEM
            ]
            if call_count == 1:
                assert all(
                    "Entity recentering hint:" not in message
                    for message in system_messages
                )
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T11:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marty McFly and Carlito"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {
                                            "query": "Marty McFly and Carlito"
                                        },
                                    }
                                }
                            ],
                        }
                    },
                )

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="McFly et Carlito are a French YouTube comedy duo.",
                created_at="2026-03-24T11:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "McFly et Carlito\n- McFly et Carlito are a French YouTube comedy duo."
            ),
            payload={
                "query": call.arguments["query"],
                "result_count": 1,
                "grounding_note": (
                    "McFly et Carlito\n"
                    "- McFly et Carlito are a French YouTube comedy duo."
                ),
            },
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Qui sont McFly et Carlito ?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Je pensais a d'autres personnalites.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "Non, je parle bien du duo YouTube francais McFly et Carlito.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Compris.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "Fais leur bio courte.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Fais leur bio courte.",
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        assert reply == "McFly et Carlito are a French YouTube comedy duo."
        assert call_count == 3
        assert len(observed_tool_calls) == 1
        assert observed_tool_calls[0].arguments["query"] == "Marty McFly and Carlito"
    finally:
        session_manager.close()


def test_multi_entity_mismatch_clamps_substantive_model_reply(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T12:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "McFly et Carlito"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {"query": "McFly et Carlito"},
                                    }
                                }
                            ],
                        }
                    },
                )

            if call_count == 2:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content=(
                        "Marty McFly is a fictional character, and Carlito is a wrestler."
                    ),
                    created_at="2026-03-24T12:00:01Z",
                    finish_reason="stop",
                )

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=json.dumps(
                    {
                        "final_reply": (
                            "The quick web grounding appeared to match a different "
                            "entity, so I couldn't confirm the requested details "
                            "from that result alone."
                        )
                    }
                ),
                created_at="2026-03-24T12:00:02Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "McFly et Carlito\n"
                "!= results appear to be for a different entity"
            ),
                payload={
                    "query": call.arguments["query"],
                    "result_count": 1,
                    "match_quality": "mismatch",
                    "supported_point_count": 0,
                    "grounding_note": (
                        "McFly et Carlito\n"
                        "!= results appear to be for a different entity"
                ),
            },
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Who are McFly et Carlito?",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Who are McFly et Carlito?",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "The quick web grounding appeared to match a different entity, so I "
            "couldn't confirm the requested details from that result alone."
        )
        assert call_count == 3
    finally:
        session_manager.close()


def test_timeout_fallback_stays_honest_when_model_overclaims(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T13:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fetch_url_text",
                            arguments={"url": "https://example.com/slow"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fetch_url_text",
                                        "arguments": {"url": "https://example.com/slow"},
                                    }
                                }
                            ],
                        }
                    },
                )

            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="The page confirms the acquisition details.",
                created_at="2026-03-24T13:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FETCH_URL_TEXT_DEFINITION,
        lambda call: ToolResult.failure(
            tool_name=call.tool_name,
            error="Tool 'fetch_url_text' timed out after 0.05 seconds.",
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Fetch the acquisition page.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Fetch the acquisition page.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert call_count == 2
    finally:
        session_manager.close()


def test_cross_turn_entity_not_contaminated_by_previous_entity(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    """Regression: entity from a previous turn must not bleed into the current turn.

    Scenario:
      Turn N-1: user asks about 'Inoxtag N' (with keyboard noise).
      Turn N:   user asks 'Qui est Marine Leleu ?'
    The search tool call on Turn N must use 'Marine Leleu', NOT 'Inoxtag N'.
    """
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T14:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marine Leleu"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {"query": "Marine Leleu"},
                                    }
                                }
                            ],
                        }
                    },
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Marine Leleu est une athlète française d'endurance.",
                created_at="2026-03-24T14:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Marine Leleu\n- Marine Leleu est une athlète française.",
            payload={
                "query": call.arguments["query"],
                "result_count": 1,
                "grounding_note": "Marine Leleu\n- Marine Leleu est une athlète française.",
            },
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        # Previous turn: history contains "Inoxtag N" as entity
        session_manager.add_message(
            MessageRole.USER,
            "Inoxtag N",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Voici des informations sur Inoxtag.",
            session_id=session.id,
        )
        # Current turn: explicit new entity
        session_manager.add_message(
            MessageRole.USER,
            "Qui est Marine Leleu ?",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Qui est Marine Leleu ?",
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        assert len(observed_tool_calls) == 1, "Exactly one search call expected"
        actual_query = observed_tool_calls[0].arguments["query"]
        # The query must reference Marine Leleu, NOT the stale "Inoxtag N"
        assert "Marine Leleu" in actual_query or "marine leleu" in actual_query.casefold(), (
            f"Cross-turn contamination detected: search used {actual_query!r} "
            "instead of Marine Leleu"
        )
        assert "inoxtag" not in actual_query.casefold(), (
            f"Stale entity leaked into current-turn search: {actual_query!r}"
        )
    finally:
        session_manager.close()


def test_non_recherche_correction_is_not_treated_as_refusal(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    """Regression: 'non recherche Marine Leleu' must trigger a search, not abort.

    The phrase 'non recherche X' is a correction utterance recentering toward X.
    It must NOT be interpreted as 'do not search'.
    """
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    observed_tool_calls: list[ToolCall] = []
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            system_texts = " ".join(
                m.content for m in messages if m.role is LLMRole.SYSTEM
            )
            assert "Entity recentering hint:" not in system_texts, (
                "Entity recentering note must stay out of the normal first-pass path"
            )
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-24T15:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fast_web_search",
                            arguments={"query": "Marine Leleu"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fast_web_search",
                                        "arguments": {"query": "Marine Leleu"},
                                    }
                                }
                            ],
                        }
                    },
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Marine Leleu est une athlète française.",
                created_at="2026-03-24T15:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        FAST_WEB_SEARCH_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Marine Leleu\n- Marine Leleu est une athlète française.",
            payload={
                "query": call.arguments["query"],
                "result_count": 1,
                "grounding_note": "Marine Leleu\n- Marine Leleu est une athlète française.",
            },
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Qui est Marine Leleu ?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Je ne trouve pas d'informations.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "non recherche Marine Leleu",
            session_id=session.id,
        )

        run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="non recherche Marine Leleu",
            tracer=tracer,
            tool_registry=tool_registry,
            tool_call_callback=observed_tool_calls.append,
        )

        # The correction must have triggered a search, not an abort
        assert call_count >= 1, "'non recherche X' must not abort the search"
    finally:
        session_manager.close()
