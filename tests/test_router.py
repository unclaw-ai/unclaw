from __future__ import annotations

from pathlib import Path

from unclaw.core.capabilities import build_runtime_capability_summary
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.router import RouteKind, route_request
from unclaw.llm.base import LLMResponse
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


def _load_repo_settings():
    return load_settings(project_root=Path(__file__).resolve().parents[1])


def _build_capability_summary(settings):
    registry = create_default_tool_registry(settings)
    return build_runtime_capability_summary(
        tool_registry=registry,
        memory_summary_available=False,
        model_can_call_tools=False,
    )
