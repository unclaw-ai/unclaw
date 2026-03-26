from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import yaml

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
from unclaw.tools.web_tools import FETCH_URL_TEXT_DEFINITION, SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.integration


def _is_grounded_finalizer_call(messages: list[object]) -> bool:
    if not messages:
        return False
    first_content = getattr(messages[0], "content", "")
    return isinstance(first_content, str) and first_content.startswith(
        "Grounded reply finalizer for one runtime turn."
    )


def _build_native_runtime(
    project_root,
    set_profile_tool_mode,
    *,
    tool_timeout_seconds: float = 0.05,
):
    app_config_path = project_root / "config" / "app.yaml"
    app_payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(app_payload, dict)
    runtime_payload = app_payload.setdefault("runtime", {})
    assert isinstance(runtime_payload, dict)
    runtime_payload["tool_timeout_seconds"] = tool_timeout_seconds
    app_config_path.write_text(
        yaml.safe_dump(app_payload, sort_keys=False),
        encoding="utf-8",
    )

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


def _tool_call_response(tool_name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        provider="ollama",
        model_name="fake-model",
        content="",
        created_at="2026-03-25T09:00:00Z",
        finish_reason="stop",
        tool_calls=(
            ToolCall(
                tool_name=tool_name,
                arguments=arguments,
            ),
        ),
        raw_payload={
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    }
                ],
            }
        },
    )


def test_search_web_timeout_retries_once_with_lighter_args_and_no_extra_model_call(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    search_calls: list[ToolCall] = []
    tool_registry = ToolRegistry()

    def _search_tool(call: ToolCall) -> ToolResult:
        search_calls.append(call)
        if len(search_calls) == 1:
            time.sleep(0.20)
            return ToolResult.ok(
                tool_name=call.tool_name,
                output_text="First attempt finished too late.",
            )
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text=(
                "Search query: retry topic\n"
                "Sources fetched: 1 of 1 attempted\n"
                "Evidence kept: 2\n"
            ),
            payload={
                "query": "retry topic",
                "summary_points": ["Recovered grounded fact."],
                "display_sources": [
                    {
                        "title": "Retry Source",
                        "url": "https://example.com/retry-source",
                    }
                ],
            },
        )

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)
    captured_calls: list[dict[str, object]] = []

    def _second_step(  # type: ignore[no-untyped-def]
        *,
        profile,
        messages,
        timeout_seconds=None,
        thinking_enabled=False,
        content_callback=None,
        tools=None,
    ):
        del timeout_seconds, thinking_enabled, content_callback, tools
        tool_messages = [
            message.content for message in messages if message.role is LLMRole.TOOL
        ]
        assert len(tool_messages) == 1
        assert "Search query: retry topic" in tool_messages[0]
        assert "Evidence kept: 2" in tool_messages[0]
        assert "timed out" not in tool_messages[0]
        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content="Recovered grounded fact.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        )

    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "search_web",
            {"query": "retry topic", "max_results": 8},
        ),
        _second_step,
        captured_calls=captured_calls,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Research retry topic.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Research retry topic.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        stored_tool_messages = [
            message.content
            for message in session_manager.list_messages(session.id)
            if message.role is MessageRole.TOOL
        ]
        assert "Recovered grounded fact." in reply
        assert "https://example.com/retry-source" in reply
        assert len(search_calls) == 2
        assert search_calls[0].arguments == {"query": "retry topic", "max_results": 8}
        assert search_calls[1].arguments == {"query": "retry topic", "max_results": 3}
        assert fake_provider.call_count() == 2
        assert len(
            [
                captured_call
                for captured_call in captured_calls
                if not _is_grounded_finalizer_call(
                    list(captured_call.get("messages", []))
                )
            ]
        ) == 2
        assert len(stored_tool_messages) == 1
        assert stored_tool_messages[0].startswith("Tool: search_web\nOutcome: success\n")
        assert "timed out" not in stored_tool_messages[0]
    finally:
        session_manager.close()


def test_search_web_timeout_retry_stops_after_second_timeout(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    search_calls: list[ToolCall] = []
    tool_registry = ToolRegistry()

    def _search_tool(call: ToolCall) -> ToolResult:
        search_calls.append(call)
        assert len(search_calls) <= 2
        time.sleep(0.20)
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text="This result always arrives too late.",
        )

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)

    def _second_step(  # type: ignore[no-untyped-def]
        *,
        profile,
        messages,
        timeout_seconds=None,
        thinking_enabled=False,
        content_callback=None,
        tools=None,
    ):
        del timeout_seconds, thinking_enabled, content_callback, tools
        tool_messages = [
            message.content for message in messages if message.role is LLMRole.TOOL
        ]
        assert len(tool_messages) == 1
        assert "timed out after 0.05 seconds" in tool_messages[0]
        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content="The search timed out again, so I couldn't confirm it.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        )

    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "search_web",
            {"query": "retry topic", "max_results": 9},
        ),
        _second_step,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Research retry topic.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Research retry topic.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert len(search_calls) == 2
        assert search_calls[1].arguments == {"query": "retry topic", "max_results": 3}
        assert fake_provider.call_count() == 2
    finally:
        session_manager.close()


def test_search_web_non_timeout_failure_does_not_retry(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    search_calls: list[ToolCall] = []
    tool_registry = ToolRegistry()

    def _search_tool(call: ToolCall) -> ToolResult:
        search_calls.append(call)
        return ToolResult.failure(
            tool_name=call.tool_name,
            error="Connection refused",
        )

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)

    def _second_step(  # type: ignore[no-untyped-def]
        *,
        profile,
        messages,
        timeout_seconds=None,
        thinking_enabled=False,
        content_callback=None,
        tools=None,
    ):
        del timeout_seconds, thinking_enabled, content_callback, tools
        tool_messages = [
            message.content for message in messages if message.role is LLMRole.TOOL
        ]
        assert len(tool_messages) == 1
        assert "Connection refused" in tool_messages[0]
        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content="The search failed, so I couldn't confirm it.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        )

    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "search_web",
            {"query": "retry topic", "max_results": 8},
        ),
        _second_step,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Research retry topic.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Research retry topic.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "The tool step failed, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert len(search_calls) == 1
        assert fake_provider.call_count() == 2
    finally:
        session_manager.close()


def test_non_search_web_timeout_does_not_retry(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    build_scripted_ollama_provider,
) -> None:
    project_root = make_temp_project()
    _settings, session_manager, tracer, command_handler = _build_native_runtime(
        project_root,
        set_profile_tool_mode,
    )
    fetch_calls: list[ToolCall] = []
    tool_registry = ToolRegistry()

    def _fetch_tool(call: ToolCall) -> ToolResult:
        fetch_calls.append(call)
        time.sleep(0.20)
        return ToolResult.ok(
            tool_name=call.tool_name,
            output_text="This result always arrives too late.",
        )

    tool_registry.register(FETCH_URL_TEXT_DEFINITION, _fetch_tool)

    def _second_step(  # type: ignore[no-untyped-def]
        *,
        profile,
        messages,
        timeout_seconds=None,
        thinking_enabled=False,
        content_callback=None,
        tools=None,
    ):
        del timeout_seconds, thinking_enabled, content_callback, tools
        tool_messages = [
            message.content for message in messages if message.role is LLMRole.TOOL
        ]
        assert len(tool_messages) == 1
        assert "timed out after 0.05 seconds" in tool_messages[0]
        return LLMResponse(
            provider="ollama",
            model_name=profile.model_name,
            content="The fetch timed out, so I couldn't confirm it.",
            created_at="2026-03-25T09:00:01Z",
            finish_reason="stop",
        )

    fake_provider = build_scripted_ollama_provider(
        _tool_call_response(
            "fetch_url_text",
            {"url": "https://example.com/slow-page"},
        ),
        _second_step,
    )
    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", fake_provider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Fetch the slow page.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Fetch the slow page.",
            tracer=tracer,
            tool_registry=tool_registry,
        )

        assert reply == (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
        assert len(fetch_calls) == 1
        assert fake_provider.call_count() == 2
    finally:
        session_manager.close()
