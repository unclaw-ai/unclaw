from __future__ import annotations

from datetime import date

from unclaw.core.search_grounding import (
    SearchGroundingContext,
    SearchGroundingFinding,
    shape_reply_with_grounding,
    should_apply_search_grounding,
)
from unclaw.llm.base import LLMMessage, LLMResponse
from unclaw.settings import load_settings


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
