from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.research_flow import run_search_command
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.reliability


def test_run_user_turn_stays_stable_across_repeated_turns_with_long_history(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
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
    captured_messages: list[list[LLMMessage]] = []

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            captured_messages.append(list(messages))
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="Stable follow-up reply.",
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        for index in range(45):
            session_manager.add_message(
                MessageRole.USER,
                f"Earlier question {index}",
                session_id=session.id,
            )
            session_manager.add_message(
                MessageRole.ASSISTANT,
                f"Earlier answer {index}",
                session_id=session.id,
            )

        first_prompt = "Need a short recap now."
        session_manager.add_message(
            MessageRole.USER,
            first_prompt,
            session_id=session.id,
        )
        expected_first_history = session_manager.list_messages(session.id)[-20:]

        first_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=first_prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        second_prompt = "Repeat the short recap."
        session_manager.add_message(
            MessageRole.USER,
            second_prompt,
            session_id=session.id,
        )
        expected_second_history = session_manager.list_messages(session.id)[-20:]

        second_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=second_prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        assert first_reply == "Stable follow-up reply."
        assert second_reply == "Stable follow-up reply."
        assert len(expected_first_history) == 20
        assert len(expected_second_history) == 20

        first_runtime_history = _non_system_message_pairs(captured_messages[0])
        second_runtime_history = _non_system_message_pairs(captured_messages[1])

        assert first_runtime_history == _chat_message_pairs(expected_first_history)
        assert second_runtime_history == _chat_message_pairs(expected_second_history)
        assert sum(1 for _, content in first_runtime_history if content == first_prompt) == 1
        assert sum(1 for _, content in second_runtime_history if content == second_prompt) == 1

        persisted_tail = session_manager.list_messages(session.id)[-4:]
        assert [message.role for message in persisted_tail] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert [message.content for message in persisted_tail] == [
            first_prompt,
            "Stable follow-up reply.",
            second_prompt,
            "Stable follow-up reply.",
        ]
    finally:
        session_manager.close()


def test_run_user_turn_stays_stable_across_repeated_turns_with_unicode_heavy_history(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
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
    captured_messages: list[list[LLMMessage]] = []
    decomposed_cafe = "cafe\u0301"
    reply_text = (
        f"Réponse stable — café / {decomposed_cafe} / مرحبًا / こんにちは / 😀."
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            captured_messages.append(list(messages))
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=reply_text,
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        multilingual_fragments = (
            ("café", "déjà vu"),
            (decomposed_cafe, "mañana"),
            ("مرحبًا", "أهلاً"),
            ("こんにちは", "さようなら"),
            ("😀", "🎉"),
        )
        for index in range(14):
            left, right = multilingual_fragments[index % len(multilingual_fragments)]
            session_manager.add_message(
                MessageRole.USER,
                f"Earlier question {index} — {left} / « {right} »",
                session_id=session.id,
            )
            session_manager.add_message(
                MessageRole.ASSISTANT,
                f"Earlier answer {index} — {right} / {left}",
                session_id=session.id,
            )

        first_prompt = (
            f"Peux-tu résumer café / {decomposed_cafe} / مرحبًا / こんにちは / 😀 ?"
        )
        session_manager.add_message(
            MessageRole.USER,
            first_prompt,
            session_id=session.id,
        )
        expected_first_history = session_manager.list_messages(session.id)[-20:]

        first_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=first_prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        second_prompt = (
            f"Encore plus bref sur café / {decomposed_cafe} / مرحبًا / こんにちは / 😀."
        )
        session_manager.add_message(
            MessageRole.USER,
            second_prompt,
            session_id=session.id,
        )
        expected_second_history = session_manager.list_messages(session.id)[-20:]

        second_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=second_prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        assert first_reply == reply_text
        assert second_reply == reply_text
        assert len(expected_first_history) == 20
        assert len(expected_second_history) == 20

        first_runtime_history = _non_system_message_pairs(captured_messages[0])
        second_runtime_history = _non_system_message_pairs(captured_messages[1])

        assert first_runtime_history == _chat_message_pairs(expected_first_history)
        assert second_runtime_history == _chat_message_pairs(expected_second_history)
        assert sum(1 for _, content in first_runtime_history if content == first_prompt) == 1
        assert sum(1 for _, content in second_runtime_history if content == second_prompt) == 1
        assert any(content == f"Earlier question 11 — {decomposed_cafe} / « mañana »" for _, content in first_runtime_history)

        persisted_tail = session_manager.list_messages(session.id)[-4:]
        assert [message.role for message in persisted_tail] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert [message.content for message in persisted_tail] == [
            first_prompt,
            reply_text,
            second_prompt,
            reply_text,
        ]
    finally:
        session_manager.close()


def test_search_backed_follow_up_stays_grounded_in_a_longer_session(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
) -> None:
    project_root = make_temp_project()
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
    captured_messages: list[list[LLMMessage]] = []
    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del timeout_seconds, thinking_enabled, content_callback, tools
            nonlocal call_count
            call_count += 1
            captured_messages.append(list(messages))
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-16T10:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="search_web",
                            arguments={"query": "latest news about Ollama"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "search_web",
                                        "arguments": {
                                            "query": "latest news about Ollama"
                                        },
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
                        "Ollama shipped a new update with improved search grounding."
                    ),
                    created_at="2026-03-16T10:00:01Z",
                    finish_reason="stop",
                )
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content="The grounded finding still holds.",
                created_at="2026-03-16T10:00:02Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    tool_registry = ToolRegistry()
    tool_registry.register(
        SEARCH_WEB_DEFINITION,
        lambda _call: ToolResult.ok(
            tool_name="search_web",
            output_text=(
                "Search query: latest news about Ollama\n"
                "Sources fetched: 2 of 2 attempted\n"
                "Evidence kept: 4\n"
            ),
            payload={
                "query": "latest news about Ollama",
                "summary_points": [
                    "Ollama shipped a new update with improved search grounding."
                ],
                "display_sources": [
                    {
                        "title": "Ollama Blog",
                        "url": "https://ollama.com/blog/search-update",
                    },
                    {
                        "title": "Release Notes",
                        "url": "https://example.com/releases/ollama-search",
                    },
                ],
            },
        ),
    )

    try:
        session = session_manager.ensure_current_session()
        for index in range(15):
            session_manager.add_message(
                MessageRole.USER,
                f"Earlier topic {index}",
                session_id=session.id,
            )
            session_manager.add_message(
                MessageRole.ASSISTANT,
                f"Earlier reply {index}",
                session_id=session.id,
            )

        search_reply = run_search_command(
            session_manager=session_manager,
            command_handler=command_handler,
            tracer=tracer,
            tool_call=ToolCall(
                tool_name="search_web",
                arguments={"query": "latest news about Ollama"},
            ),
            tool_registry=tool_registry,
        ).assistant_reply

        for index in range(8):
            session_manager.add_message(
                MessageRole.USER,
                f"Later follow-up {index}",
                session_id=session.id,
            )
            session_manager.add_message(
                MessageRole.ASSISTANT,
                f"Later answer {index}",
                session_id=session.id,
            )

        follow_up_prompt = "What still looks reliable from that?"
        session_manager.add_message(
            MessageRole.USER,
            follow_up_prompt,
            session_id=session.id,
        )
        recent_history = session_manager.list_messages(session.id)[-20:]

        follow_up_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=follow_up_prompt,
            tracer=tracer,
            tool_registry=ToolRegistry(),
        )

        assert "Sources:" in search_reply
        assert "https://ollama.com/blog/search-update" in search_reply
        assert follow_up_reply == "The grounded finding still holds."
        assert len(recent_history) == 20
        assert any(message.role is MessageRole.TOOL for message in recent_history)

        follow_up_messages = captured_messages[-1]
        assert len([message for message in follow_up_messages if message.role is not LLMRole.SYSTEM]) == 20
        assert any(
            message.role is LLMRole.SYSTEM
            and "Search-backed answer contract:" in message.content
            for message in follow_up_messages
        )
        tool_messages = [
            message.content
            for message in follow_up_messages
            if message.role is LLMRole.TOOL
        ]
        assert len(tool_messages) == 1
        assert tool_messages[0].startswith("UNTRUSTED TOOL OUTPUT:")
        assert "Grounding rules:" in tool_messages[0]
        assert "Supported facts:" in tool_messages[0]
        assert "Ollama Blog: https://ollama.com/blog/search-update" in tool_messages[0]
        assert follow_up_messages[-1].role is LLMRole.USER
        assert follow_up_messages[-1].content == follow_up_prompt
    finally:
        session_manager.close()


def test_session_persistence_stays_consistent_with_two_near_concurrent_writers(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.create_session(title="Concurrent writers", make_current=False)
        step_barrier = threading.Barrier(2)
        exceptions: list[BaseException] = []
        message_count = 6

        def _writer(prefix: str, role: MessageRole) -> None:
            manager = SessionManager.from_settings(settings)
            try:
                for index in range(message_count):
                    step_barrier.wait(timeout=5)
                    manager.add_message(
                        role,
                        f"{prefix}-{index}",
                        session_id=session.id,
                    )
            except BaseException as exc:  # pragma: no cover - asserted below
                exceptions.append(exc)
                try:
                    step_barrier.abort()
                except threading.BrokenBarrierError:
                    pass
            finally:
                manager.close()

        user_thread = threading.Thread(
            target=_writer,
            args=("user-writer", MessageRole.USER),
        )
        assistant_thread = threading.Thread(
            target=_writer,
            args=("assistant-writer", MessageRole.ASSISTANT),
        )

        user_thread.start()
        assistant_thread.start()
        user_thread.join(timeout=10)
        assistant_thread.join(timeout=10)

        assert not user_thread.is_alive()
        assert not assistant_thread.is_alive()
        assert exceptions == []

        stored_messages = session_manager.list_messages(session.id)
        assert len(stored_messages) == message_count * 2
        assert all(message.session_id == session.id for message in stored_messages)
        assert [message.created_at for message in stored_messages] == sorted(
            message.created_at for message in stored_messages
        )
        assert {message.content for message in stored_messages} == {
            *(f"user-writer-{index}" for index in range(message_count)),
            *(f"assistant-writer-{index}" for index in range(message_count)),
        }
        assert [
            message.content
            for message in stored_messages
            if message.content.startswith("user-writer-")
        ] == [f"user-writer-{index}" for index in range(message_count)]
        assert [
            message.content
            for message in stored_messages
            if message.content.startswith("assistant-writer-")
        ] == [f"assistant-writer-{index}" for index in range(message_count)]
    finally:
        session_manager.close()


def _chat_message_pairs(messages) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    return [(message.role.value, message.content) for message in messages]


def _non_system_message_pairs(messages: list[LLMMessage]) -> list[tuple[str, str]]:
    return [
        (message.role.value, message.content)
        for message in messages
        if message.role is not LLMRole.SYSTEM
    ]
