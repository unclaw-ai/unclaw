from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.router import (
    RouteKind,
    _extract_anchor_spans,
    _guard_exact_spans,
    route_request,
)
from unclaw.llm.base import LLMProviderError, LLMResponse
from unclaw.settings import load_settings


def test_route_request_preserves_router_reformulated_query_for_web_search(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "fais une recherche en ligne sur Marine Leleu et fais moi sa biographie"
    reformulated_query = "biographie de Marine Leleu"

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
            del profile, timeout_seconds, thinking_enabled, content_callback, tools
            assert "Return JSON only" in messages[0].content
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    '{"route":"web_search",'
                    f'"search_query":"{reformulated_query}"'
                    "}"
                ),
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query == reformulated_query
    assert route.search_query != user_input


def test_route_request_preserves_router_reformulated_weather_query(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "Quelle est la météo à Paris aujourd'hui ?"
    reformulated_query = "météo Paris aujourd'hui"

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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=(
                    '{"route":"web_search",'
                    f'"search_query":"{reformulated_query}"'
                    "}"
                ),
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query == reformulated_query


def test_route_request_keeps_plain_local_request_on_chat_route(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)

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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content='{"route":"chat","search_query":""}',
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="Explique en une phrase ce que fait ce programme.",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.CHAT
    assert route.search_query is None


def test_route_request_falls_back_to_normal_chat_when_classifier_output_is_invalid(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)

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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="not-json",
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="Peux-tu vérifier cette information en ligne ?",
        thinking_enabled=True,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.CHAT_WITH_THINKING
    assert route.search_query is None


def test_route_request_uses_dedicated_router_profile_when_available(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    seen_profiles: list[str] = []

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
            del messages, timeout_seconds, thinking_enabled, content_callback, tools
            seen_profiles.append(profile.name)
            return LLMResponse(
                provider="ollama",
                model_name="llama3.2:3b",
                content='{"route":"chat","search_query":""}',
                created_at="2026-03-19T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="Explain the difference between RAM and storage.",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.CHAT
    assert route.planner_profile_name == "router"
    assert route.planner_available is True
    assert route.planner_fallback_reason is None
    assert seen_profiles == ["router"]


def test_route_request_uses_dedicated_router_for_codex_profile(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    seen_profiles: list[str] = []

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
            del messages, timeout_seconds, thinking_enabled, content_callback, tools
            seen_profiles.append(profile.name)
            return LLMResponse(
                provider="ollama",
                model_name="qwen3:1.7b",
                content='{"route":"chat","search_query":""}',
                created_at="2026-03-19T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="codex",
        user_input="Explain how this refactor affects the codebase.",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.CHAT
    assert route.planner_profile_name == "router"
    assert route.planner_available is True
    assert seen_profiles == ["router"]


def test_route_request_falls_back_to_responder_when_router_profile_fails(
    monkeypatch,
) -> None:
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    seen_profiles: list[str] = []

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
            del messages, timeout_seconds, thinking_enabled, content_callback, tools
            seen_profiles.append(profile.name)
            if profile.name == "router":
                raise LLMProviderError("router offline")
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content='{"route":"web_search","search_query":"latest AI news"}',
                created_at="2026-03-19T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.router.OllamaProvider", FakeOllamaProvider)

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input="What is the latest AI news today?",
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query == "latest AI news"
    assert route.planner_profile_name == "router"
    assert route.planner_available is False
    assert route.planner_fallback_reason is not None
    assert "falling back" in route.planner_fallback_reason.lower()
    assert seen_profiles == ["router", "main"]


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])


def _build_capability_summary(settings):
    registry = create_default_tool_registry(settings)
    return build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )


# ---------------------------------------------------------------------------
# P5-1: exact-span anchor extraction unit tests
# ---------------------------------------------------------------------------


def test_extract_anchor_spans_person_name() -> None:
    """Multi-word capitalized person names are extracted as anchors."""
    anchors = _extract_anchor_spans("fais une recherche sur Marine Leleu")
    assert "Marine Leleu" in anchors


def test_extract_anchor_spans_french_sentence_start_not_captured_as_anchor() -> None:
    """A single capitalized word at sentence start (not a multi-word proper name)
    is not extracted as an anchor — only multi-word spans qualify."""
    anchors = _extract_anchor_spans("Quelle est la météo à Paris aujourd'hui ?")
    # "Quelle" is a sentence-start capital, not a multi-word proper name.
    # "Paris" is a single-word capital — requires 2+ words to qualify.
    assert not any("Quelle" in a for a in anchors)
    assert not any("Paris" == a for a in anchors)


def test_extract_anchor_spans_quoted_string() -> None:
    """Quoted strings (double quotes) are extracted as anchors."""
    anchors = _extract_anchor_spans('what is "SqueezeIE"')
    assert "SqueezeIE" in anchors


def test_extract_anchor_spans_single_quoted_string() -> None:
    """Quoted strings (single quotes) are extracted as anchors."""
    anchors = _extract_anchor_spans("look up 'OpenClaw project'")
    assert "OpenClaw project" in anchors


def test_extract_anchor_spans_guillemets() -> None:
    """French guillemet-quoted strings are extracted as anchors."""
    anchors = _extract_anchor_spans("cherche des infos sur «Marine Leleu»")
    assert "Marine Leleu" in anchors


def test_extract_anchor_spans_repo_path() -> None:
    """Slash-path tokens (user/repo pattern) are extracted as anchors."""
    anchors = _extract_anchor_spans("look up mattiascockburn/squeezie")
    assert "mattiascockburn/squeezie" in anchors


def test_extract_anchor_spans_all_caps_hyphen() -> None:
    """ALL-CAPS-HYPHEN identifiers like GLM-OCR are extracted as anchors."""
    anchors = _extract_anchor_spans("cherche des infos sur GLM-OCR")
    assert "GLM-OCR" in anchors


def test_extract_anchor_spans_camelcase() -> None:
    """CamelCase identifiers like OpenClaw are extracted as anchors."""
    anchors = _extract_anchor_spans("résume moi OpenClaw")
    assert "OpenClaw" in anchors


def test_extract_anchor_spans_camelcase_multi_hump() -> None:
    """Multi-hump CamelCase (SqueezeIE) is extracted as an anchor."""
    anchors = _extract_anchor_spans('what is "SqueezeIE" all about')
    # CamelCase extraction; also caught via quotes above
    assert any("SqueezeIE" in a for a in anchors)


def test_extract_anchor_spans_no_anchors_in_broad_query() -> None:
    """Broad informational queries produce no anchors — reformulation is safe."""
    anchors = _extract_anchor_spans("what is the capital of France")
    assert anchors == []


def test_extract_anchor_spans_no_anchors_plain_lowercase() -> None:
    """All-lowercase queries with no special tokens produce no anchors."""
    anchors = _extract_anchor_spans("how does recursion work in python")
    assert anchors == []


def test_extract_anchor_spans_deduplicates() -> None:
    """Duplicate spans (different detection paths) are deduplicated."""
    anchors = _extract_anchor_spans('résume moi "Marine Leleu" et Marine Leleu')
    count = sum(1 for a in anchors if a.lower() == "marine leleu")
    assert count == 1


# ---------------------------------------------------------------------------
# P5-1: _guard_exact_spans unit tests
# ---------------------------------------------------------------------------


def test_guard_returns_reformulation_when_anchor_preserved() -> None:
    """When all anchors survive in the reformulation, it is returned unchanged."""
    result = _guard_exact_spans(
        "fais une recherche sur Marine Leleu",
        "biographie de Marine Leleu",
    )
    assert result == "biographie de Marine Leleu"


def test_guard_returns_none_when_person_name_replaced() -> None:
    """When a person name is replaced in the reformulation, guard returns None."""
    result = _guard_exact_spans(
        "fais une recherche sur Marine Leleu",
        "Marine Le Pen",  # wrong person — entity drift
    )
    assert result is None


def test_guard_returns_none_when_repo_path_dropped() -> None:
    """When a repo path is dropped from the reformulation, guard returns None."""
    result = _guard_exact_spans(
        "look up mattiascockburn/squeezie",
        "squeezie github",  # path dropped
    )
    assert result is None


def test_guard_returns_none_when_technical_id_dropped() -> None:
    """When a technical identifier is dropped, guard returns None."""
    result = _guard_exact_spans(
        "cherche des infos sur GLM-OCR",
        "OCR model summary",  # GLM-OCR rewritten
    )
    assert result is None


def test_guard_returns_none_when_quoted_string_dropped() -> None:
    """When a quoted string is absent from the reformulation, guard returns None."""
    result = _guard_exact_spans(
        'what is "SqueezeIE"',
        "squeeze IE definition",  # quoted span dropped/rewritten
    )
    assert result is None


def test_guard_returns_reformulation_when_no_anchors() -> None:
    """Broad queries with no anchors always use the reformulation as-is."""
    result = _guard_exact_spans(
        "what is the capital of France",
        "France capital city",
    )
    assert result == "France capital city"


def test_guard_returns_none_when_search_query_is_none() -> None:
    """When search_query is None, guard returns None (no-op)."""
    result = _guard_exact_spans("some query", None)
    assert result is None


def test_guard_returns_none_when_search_query_empty() -> None:
    """When search_query is empty, guard returns None (no-op)."""
    result = _guard_exact_spans("some query", "")
    assert result is None


def test_guard_case_insensitive_anchor_check() -> None:
    """Anchor check is case-insensitive: lowercase anchor in query still passes."""
    result = _guard_exact_spans(
        "tell me about Marine Leleu",
        "marine leleu biography",  # anchor present but lowercased in query
    )
    assert result == "marine leleu biography"


def test_guard_camelcase_preserved_in_reformulation() -> None:
    """CamelCase identifier present in reformulation → reformulation returned."""
    result = _guard_exact_spans(
        "résume moi OpenClaw",
        "OpenClaw project summary",
    )
    assert result == "OpenClaw project summary"


def test_guard_camelcase_dropped_in_reformulation() -> None:
    """CamelCase identifier absent from reformulation → guard returns None."""
    result = _guard_exact_spans(
        "résume moi OpenClaw",
        "open source claw project",  # CamelCase lost
    )
    assert result is None


# ---------------------------------------------------------------------------
# P5-1: route_request integration tests for exact-span guard
# ---------------------------------------------------------------------------


def _make_fake_provider_class(search_query_value: str):
    """Build a FakeOllamaProvider that returns the given search_query."""

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
            del profile, messages, timeout_seconds, thinking_enabled, content_callback, tools
            import json as _json

            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=_json.dumps(
                    {"route": "web_search", "search_query": search_query_value}
                ),
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    return FakeOllamaProvider


def test_route_request_guard_detects_person_name_drift(monkeypatch) -> None:
    """When the classifier replaces a person name (Marine Leleu → Marine Le Pen),
    the guard must set search_query to None so the runtime falls back to
    the original user input containing the correct entity.
    """
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "fais une recherche sur Marine Leleu"
    drifted_query = "Marine Le Pen"  # entity replaced by classifier

    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider_class(drifted_query),
    )

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query is None, (
        "Guard must discard a reformulation that replaced the person name. "
        "search_query=None signals the runtime to fall back to user input."
    )


def test_route_request_guard_passes_faithful_reformulation(monkeypatch) -> None:
    """When the classifier preserves the person name in the reformulation,
    the guard leaves it unchanged and the reformulation is used.
    """
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "fais une recherche sur Marine Leleu"
    faithful_query = "Marine Leleu biographie"  # entity preserved

    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider_class(faithful_query),
    )

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query == faithful_query, (
        "Guard must pass through a reformulation that preserved the person name."
    )


def test_route_request_guard_passes_broad_query_reformulation(monkeypatch) -> None:
    """For broad queries with no risky spans, reformulation is always used."""
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "what is the weather like in paris today"
    reformulated = "Paris weather today"

    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider_class(reformulated),
    )

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query == reformulated


def test_route_request_guard_detects_repo_path_drift(monkeypatch) -> None:
    """When a slash-path token is dropped from the reformulation, guard returns None."""
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "look up mattiascockburn/squeezie"
    drifted_query = "squeezie github"  # repo path dropped

    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider_class(drifted_query),
    )

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query is None


def test_route_request_guard_detects_camelcase_drift(monkeypatch) -> None:
    """When a CamelCase identifier is lost in reformulation, guard returns None."""
    settings = _load_repo_settings()
    capability_summary = _build_capability_summary(settings)
    user_input = "résume moi OpenClaw"
    drifted_query = "open claw project"  # CamelCase lost

    monkeypatch.setattr(
        "unclaw.core.router.OllamaProvider",
        _make_fake_provider_class(drifted_query),
    )

    route = route_request(
        settings=settings,
        model_profile_name="main",
        user_input=user_input,
        thinking_enabled=False,
        capability_summary=capability_summary,
    )

    assert route.kind is RouteKind.WEB_SEARCH
    assert route.search_query is None
