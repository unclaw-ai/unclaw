from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from unclaw.core.command_handler import CommandHandler
from unclaw.core.research_flow import build_tool_history_content
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.core.search_grounding import (
    SearchGroundingContext,
    SearchGroundingFinding,
    shape_reply_with_grounding,
    should_apply_search_grounding,
)
from unclaw.llm.base import LLMMessage, LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolResult


def _build_grounding(
    *,
    query: str,
    current_date: date = date(2026, 3, 18),
    supported_texts: tuple[str, ...],
    uncertain_texts: tuple[str, ...] = (),
    birth_date: date | None = None,
) -> SearchGroundingContext:
    supported = tuple(
        SearchGroundingFinding(
            text=text,
            support_count=2,
            score=8.0,
            source_titles=("Example Source",),
            source_urls=("https://example.com/source",),
            confidence="supported",
        )
        for text in supported_texts
    )
    uncertain = tuple(
        SearchGroundingFinding(
            text=text,
            support_count=1,
            score=4.0,
            source_titles=("Example Source",),
            source_urls=("https://example.com/source",),
            confidence="uncertain",
        )
        for text in uncertain_texts
    )
    return SearchGroundingContext(
        query=query,
        current_date=current_date,
        supported_findings=supported,
        uncertain_findings=uncertain,
        display_sources=(("Example Source", "https://example.com/source"),),
        birth_date=birth_date,
    )


def _build_jordan_search_tool_history() -> str:
    return build_tool_history_content(
        ToolResult.ok(
            tool_name="search_web",
            output_text="Search query: who is Jordan Lee\n",
            payload={
                "query": "who is Jordan Lee",
                "display_sources": [
                    {
                        "title": "Company Bio",
                        "url": "https://company.example.com/jordan-lee",
                    }
                ],
                "synthesized_findings": [
                    {
                        "text": "Jordan Lee is a product designer and engineer.",
                        "score": 8.1,
                        "support_count": 2,
                        "source_titles": ["Company Bio"],
                        "source_urls": ["https://company.example.com/jordan-lee"],
                    }
                ],
            },
        ),
        tool_call=SimpleNamespace(
            tool_name="search_web",
            arguments={"query": "who is Jordan Lee"},
        ),
    )


def test_should_apply_search_grounding_uses_semantic_query_analysis_for_multilingual_follow_up(
    monkeypatch,
    make_temp_project,
) -> None:
    settings = load_settings(project_root=make_temp_project())
    captured_messages: list[list[LLMMessage]] = []

    class FakeProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del profile, timeout_seconds, thinking_enabled, content_callback, tools
            captured_messages.append(list(messages))
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content='{"applies_to_grounding": true, "query_kind": "person_profile", "is_follow_up": true}',
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    grounding = _build_grounding(
        query="who is Jordan Lee",
        supported_texts=("Jordan Lee is a product designer and engineer.",),
    )

    assert should_apply_search_grounding(
        query="Hazlo mas corto.",
        grounding=grounding,
        settings=settings,
        model_profile_name="main",
    ) is True
    assert len(captured_messages) == 1
    assert "stay grounded in the most recent search grounding context" in captured_messages[0][0].content


def test_should_apply_search_grounding_skips_semantic_query_analysis_for_obvious_follow_up(
    monkeypatch,
    make_temp_project,
) -> None:
    settings = load_settings(project_root=make_temp_project())

    class FailingProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            raise AssertionError("Obvious grounded follow-ups should not call the semantic analyzer.")

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FailingProvider)

    grounding = _build_grounding(
        query="latest news about Ollama",
        supported_texts=("Ollama shipped a new update with improved search grounding.",),
    )

    assert should_apply_search_grounding(
        query="Summarize that more briefly.",
        grounding=grounding,
        settings=settings,
        model_profile_name="main",
    ) is True


def test_shape_reply_with_grounding_uses_semantic_review_for_multilingual_age_rewrite(
    monkeypatch,
    make_temp_project,
) -> None:
    settings = load_settings(project_root=make_temp_project())
    captured_messages: list[list[LLMMessage]] = []

    class FakeProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del profile, timeout_seconds, thinking_enabled, content_callback, tools
            captured_messages.append(list(messages))
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    '{"rewrite_required": true, "query_kind": "age_request", '
                    '"safe_answer": "Encontre una fecha de nacimiento de 1998-05-04. '
                    'El 2026-03-14, eso da 27 anos.", '
                    '"issues": ["unsupported_age", "stale_relative_date"]}'
                ),
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeProvider)

    grounding = _build_grounding(
        query="how old is Alex Rivera",
        current_date=date(2026, 3, 14),
        supported_texts=("Alex Rivera was born on 1998-05-04.",),
        birth_date=date(1998, 5, 4),
    )

    reply = shape_reply_with_grounding(
        "Tiene 26 anos desde mayo de 2024.",
        grounding=grounding,
        query="Que edad tiene Alex Rivera?",
        settings=settings,
        model_profile_name="main",
    )

    assert reply == (
        "Encontre una fecha de nacimiento de 1998-05-04. "
        "El 2026-03-14, eso da 27 anos."
    )
    assert len(captured_messages) == 1
    assert "Review a candidate answer against grounded search evidence" in captured_messages[0][0].content


def test_shape_reply_with_grounding_skips_semantic_review_for_safe_generic_search_reply(
    monkeypatch,
    make_temp_project,
) -> None:
    settings = load_settings(project_root=make_temp_project())

    class FailingProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            raise AssertionError("Safe generic grounded replies should not call the semantic review pass.")

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FailingProvider)

    grounding = _build_grounding(
        query="latest news about Ollama",
        supported_texts=("Ollama shipped a new update with improved search grounding.",),
    )
    reply_text = "Ollama shipped a new update with improved search grounding."

    assert (
        shape_reply_with_grounding(
            reply_text,
            grounding=grounding,
            query="latest news about Ollama",
            settings=settings,
            model_profile_name="main",
        )
        == reply_text
    )


def test_run_user_turn_keeps_prior_search_grounding_when_semantic_follow_up_says_yes(
    monkeypatch,
    make_temp_project,
) -> None:
    settings = load_settings(project_root=make_temp_project())
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
    semantic_query_calls = 0
    main_model_messages: list[LLMMessage] = []

    class AppProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            nonlocal semantic_query_calls
            del profile, timeout_seconds, thinking_enabled, content_callback, tools
            system_prompt = messages[0].content if messages else ""
            if "stay grounded in the most recent search grounding context" in system_prompt:
                semantic_query_calls += 1
                return LLMResponse(
                    provider="ollama",
                    model_name="qwen3.5:4b",
                    content='{"applies_to_grounding": true, "query_kind": "person_profile", "is_follow_up": true}',
                    created_at="2026-03-18T10:00:00Z",
                    finish_reason="stop",
                )

            main_model_messages.extend(messages)
            assert any(
                message.role is LLMRole.SYSTEM
                and "Search-backed answer contract:" in message.content
                for message in messages
            )
            assert any(
                message.role is LLMRole.TOOL
                and "Jordan Lee is a product designer and engineer." in message.content
                for message in messages
            )
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Resumen breve.",
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", AppProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Who is Jordan Lee?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.TOOL,
            _build_jordan_search_tool_history(),
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Jordan Lee is a product designer and engineer.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "Hazlo mas corto.",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="Hazlo mas corto.",
            tracer=tracer,
        )

        assert reply == "Resumen breve."
        assert semantic_query_calls == 1
        assert main_model_messages
    finally:
        session_manager.close()


def test_run_user_turn_drops_prior_search_grounding_when_semantic_query_says_no(
    monkeypatch,
    make_temp_project,
) -> None:
    settings = load_settings(project_root=make_temp_project())
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
    semantic_query_calls = 0
    main_model_messages: list[LLMMessage] = []

    class AppProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="http://127.0.0.1:11434", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, *, timeout_seconds=None, thinking_enabled=False, content_callback=None, tools=None):  # type: ignore[no-untyped-def]
            nonlocal semantic_query_calls
            del profile, timeout_seconds, thinking_enabled, content_callback, tools
            system_prompt = messages[0].content if messages else ""
            if "stay grounded in the most recent search grounding context" in system_prompt:
                semantic_query_calls += 1
                return LLMResponse(
                    provider="ollama",
                    model_name="qwen3.5:4b",
                    content='{"applies_to_grounding": false, "query_kind": "general", "is_follow_up": false}',
                    created_at="2026-03-18T10:00:00Z",
                    finish_reason="stop",
                )

            main_model_messages.extend(messages)
            assert not any(
                message.role is LLMRole.SYSTEM
                and "Search-backed answer contract:" in message.content
                for message in messages
            )
            assert not any(
                message.role is LLMRole.TOOL
                and "Jordan Lee is a product designer and engineer." in message.content
                for message in messages
            )
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="4",
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", AppProvider)

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER,
            "Who is Jordan Lee?",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.TOOL,
            _build_jordan_search_tool_history(),
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.ASSISTANT,
            "Jordan Lee is a product designer and engineer.",
            session_id=session.id,
        )
        session_manager.add_message(
            MessageRole.USER,
            "What is 2 + 2?",
            session_id=session.id,
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input="What is 2 + 2?",
            tracer=tracer,
        )

        assert reply == "4"
        assert semantic_query_calls == 1
        assert main_model_messages
    finally:
        session_manager.close()
