"""Capability-aware routing for the current runtime phase."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.constants import ROUTER_CLASSIFIER_TIMEOUT_SECONDS
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMMessage, LLMProviderError, LLMRole
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.settings import Settings

_ROUTER_SYSTEM_PROMPT = (
    "Decide whether the user's request should stay on normal chat or use a "
    "web-backed search route. Return JSON only with keys route and "
    "search_query. route must be one of: chat, web_search, unclear. Use "
    "web_search only when the user explicitly asks for online research or when "
    "a responsible answer needs current or externally verifiable public facts. "
    "Use chat for conversation, stable general knowledge, or local reasoning. "
    "Keep search_query empty unless route is web_search."
)


class RouteKind(StrEnum):
    """Supported runtime routes for the current phase."""

    COMMAND = "command"
    CHAT = "chat"
    CHAT_WITH_THINKING = "chat_with_thinking"
    WEB_SEARCH = "web_search"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Result returned by the current lightweight router."""

    kind: RouteKind
    model_profile_name: str
    search_query: str | None = None


@dataclass(frozen=True, slots=True)
class _RouterClassifierDecision:
    route: str
    search_query: str | None = None


def route_request(
    *,
    settings: Settings,
    model_profile_name: str,
    user_input: str | None = None,
    thinking_enabled: bool,
    capability_summary: RuntimeCapabilitySummary | None = None,
    is_command: bool = False,
    allow_web_search_routing: bool = True,
) -> RouteDecision:
    """Select the current route with a conservative capability-aware check."""
    profile = settings.models.get(model_profile_name)
    if profile is None:
        raise ConfigurationError(
            f"Model profile '{model_profile_name}' is not defined in settings."
        )

    chat_route_kind = _resolve_default_chat_route(
        thinking_enabled=thinking_enabled,
        thinking_supported=profile.thinking_supported,
    )

    if is_command:
        return RouteDecision(
            kind=RouteKind.COMMAND,
            model_profile_name=model_profile_name,
        )

    if not allow_web_search_routing:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
        )

    normalized_user_input = (user_input or "").strip()
    if not normalized_user_input:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
        )

    if capability_summary is None or not capability_summary.web_search_available:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
        )

    classifier_decision = _classify_route_with_model(
        settings=settings,
        model_profile_name=model_profile_name,
        user_input=normalized_user_input,
    )
    if classifier_decision is None:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
        )
    if classifier_decision.route != RouteKind.WEB_SEARCH.value:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
        )

    return RouteDecision(
        kind=RouteKind.WEB_SEARCH,
        model_profile_name=model_profile_name,
        # Carry the router's reformulated query through when present. The
        # runtime keeps the original user input as a fallback if this is None.
        search_query=classifier_decision.search_query,
    )


def _resolve_default_chat_route(
    *,
    thinking_enabled: bool,
    thinking_supported: bool,
) -> RouteKind:
    if thinking_enabled and thinking_supported:
        return RouteKind.CHAT_WITH_THINKING
    return RouteKind.CHAT


def _classify_route_with_model(
    *,
    settings: Settings,
    model_profile_name: str,
    user_input: str,
) -> _RouterClassifierDecision | None:
    try:
        profile = resolve_model_profile(settings, model_profile_name)
    except ConfigurationError:
        return None

    provider = _create_router_provider(settings, provider_name=profile.provider)
    if provider is None:
        return None

    classifier_profile = replace(profile, temperature=0.0)
    try:
        response = provider.chat(
            profile=classifier_profile,
            messages=_build_router_messages(user_input),
            timeout_seconds=ROUTER_CLASSIFIER_TIMEOUT_SECONDS,
            thinking_enabled=False,
        )
    except LLMProviderError:
        return None

    return _parse_router_response(response.content)


def _create_router_provider(
    settings: Settings,
    *,
    provider_name: str,
) -> OllamaProvider | None:
    if provider_name != OllamaProvider.provider_name:
        return None

    return OllamaProvider(
        default_timeout_seconds=settings.app.providers.ollama.timeout_seconds,
    )


def _build_router_messages(user_input: str) -> tuple[LLMMessage, LLMMessage]:
    return (
        LLMMessage(role=LLMRole.SYSTEM, content=_ROUTER_SYSTEM_PROMPT),
        LLMMessage(
            role=LLMRole.USER,
            content=(
                f"Current date: {date.today().isoformat()}\n"
                f"User request: {user_input}"
            ),
        ),
    )


def _parse_router_response(content: str) -> _RouterClassifierDecision | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    route = payload.get("route")
    if route not in {
        RouteKind.CHAT.value,
        RouteKind.WEB_SEARCH.value,
        "unclear",
    }:
        return None

    search_query = payload.get("search_query")
    if not isinstance(search_query, str):
        search_query = None
    elif not search_query.strip():
        search_query = None
    else:
        search_query = search_query.strip()

    return _RouterClassifierDecision(route=route, search_query=search_query)
