"""Capability-aware routing for the current runtime phase."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.constants import ROUTER_PROFILE_NAME
from unclaw.errors import ConfigurationError
from unclaw.llm.base import LLMMessage, LLMProviderError, LLMRole
from unclaw.llm.model_profiles import resolve_model_profile, resolve_router_profile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.settings import Settings

_ROUTER_SYSTEM_PROMPT = (
    "Decide whether the user's request should stay on normal chat or use a "
    "web-backed search route. Return JSON only with keys route and "
    "search_query. route must be one of: chat, web_search, unclear. Use "
    "web_search only when the user explicitly asks for online research or when "
    "a responsible answer needs current or externally verifiable public facts. "
    "Use chat for conversation, stable general knowledge, local reasoning, "
    "and local actions such as system information, notes, or file operations. "
    "Arithmetic, mathematical calculations, simple unit conversions, and "
    "definitional facts that do not depend on real-time data must use chat. "
    "Keep search_query empty unless route is web_search. "
    "When route is web_search, copy all proper nouns, person names, usernames, "
    "repo names, quoted strings, and technical identifiers verbatim from the "
    "user request into search_query. Do not substitute or paraphrase named entities."
)


# ---------------------------------------------------------------------------
# Exact-span anchor extraction for search query fidelity guard — P5-1.
# Explicitly required by mission P5-1.  Complies with mandatory_rules.md
# rule 5 allowed exception: scoped to query preparation only; structural
# (no language-specific lists); not core routing architecture; easy to remove.
# ---------------------------------------------------------------------------

# Quoted spans: content between double quotes, single quotes, or guillemets.
_QUOTED_SPAN_RE = re.compile(
    r'"([^"]{2,80})"'
    r"|'([^']{2,80})'"
    r"|«([^»]{2,80})»",
)

# Multi-word capitalized span: two or more consecutive words each starting
# with an uppercase letter (likely person names, brand names, proper nouns).
_CAPITALIZED_MULTI_WORD_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b",
)

# Technical token: single tokens with non-common structure.
# Covers: path-like (user/repo), ALL-CAPS-HYPHEN (GLM-OCR), CamelCase (OpenClaw).
_TECHNICAL_TOKEN_RE = re.compile(
    r"(?:"
    r"\b[a-zA-Z0-9][a-zA-Z0-9_-]{2,}/[a-zA-Z0-9][a-zA-Z0-9_-]{2,}\b"  # path: user/repo
    r"|\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b"  # ALL-CAPS-HYPHEN: GLM-OCR
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+\b"  # CamelCase: OpenClaw, SqueezeIE
    r")",
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
    planner_profile_name: str | None = None
    planner_available: bool = False
    planner_fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _RouterClassifierDecision:
    route: str
    search_query: str | None = None


def route_request(
    *,
    settings: Settings,
    model_profile_name: str,
    planner_profile_name: str | None = None,
    user_input: str | None = None,
    thinking_enabled: bool,
    capability_summary: RuntimeCapabilitySummary | None = None,
    is_command: bool = False,
    allow_web_search_routing: bool = True,
) -> RouteDecision:
    """Select the current route with a conservative capability-aware check."""
    # Legacy compatibility only: route selection now comes from settings.router.
    del planner_profile_name
    profile = settings.models.get(model_profile_name)
    if profile is None:
        raise ConfigurationError(
            f"Model profile '{model_profile_name}' is not defined in settings."
        )

    chat_route_kind = _resolve_default_chat_route(
        thinking_enabled=thinking_enabled,
        thinking_supported=profile.thinking_supported,
    )
    route_selector_profile_name = _resolve_route_selector_profile_name(settings)

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

    classifier_decision, planner_available, planner_fallback_reason = (
        _classify_route_with_optional_router(
            settings=settings,
            model_profile_name=model_profile_name,
            route_selector_profile_name=route_selector_profile_name,
            user_input=normalized_user_input,
        )
    )
    if classifier_decision is None:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
            planner_profile_name=route_selector_profile_name,
            planner_available=planner_available,
            planner_fallback_reason=planner_fallback_reason,
        )
    if classifier_decision.route != RouteKind.WEB_SEARCH.value:
        return RouteDecision(
            kind=chat_route_kind,
            model_profile_name=model_profile_name,
            planner_profile_name=route_selector_profile_name,
            planner_available=planner_available,
            planner_fallback_reason=planner_fallback_reason,
        )

    return RouteDecision(
        kind=RouteKind.WEB_SEARCH,
        model_profile_name=model_profile_name,
        # Guard the reformulated query against exact-span drift.
        # Returns None when risky spans from user_input were dropped,
        # signalling the runtime to fall back to the original user input.
        search_query=_guard_exact_spans(
            normalized_user_input, classifier_decision.search_query
        ),
        planner_profile_name=route_selector_profile_name,
        planner_available=planner_available,
        planner_fallback_reason=planner_fallback_reason,
    )


def _resolve_route_selector_profile_name(settings: Settings) -> str | None:
    if settings.router.enabled:
        return ROUTER_PROFILE_NAME
    return None


def _classify_route_with_optional_router(
    *,
    settings: Settings,
    model_profile_name: str,
    route_selector_profile_name: str | None,
    user_input: str,
) -> tuple[_RouterClassifierDecision | None, bool, str | None]:
    if not route_selector_profile_name:
        return (
            _classify_route_with_model(
                settings=settings,
                model_profile_name=model_profile_name,
                user_input=user_input,
            ),
            False,
            None,
        )

    router_decision = _classify_route_with_model(
        settings=settings,
        model_profile_name=route_selector_profile_name,
        user_input=user_input,
    )
    if router_decision is not None:
        return router_decision, True, None

    fallback_reason = (
        "Dedicated router profile "
        f"'{route_selector_profile_name}' was unavailable for route selection; "
        "falling back to the responder profile."
    )
    legacy_decision = _classify_route_with_model(
        settings=settings,
        model_profile_name=model_profile_name,
        user_input=user_input,
    )
    return legacy_decision, False, fallback_reason


def _extract_anchor_spans(text: str) -> list[str]:
    """Extract exact spans from user input that must survive search query reformulation.

    Returns unique anchor strings: quoted content, multi-word proper-name
    sequences, and technical identifiers that the reformulated query must keep.

    Does not use language-specific lists (complies with mandatory_rules.md rule 5).
    Called only from _guard_exact_spans; not part of the main routing logic.
    """
    anchors: list[str] = []
    seen: set[str] = set()

    def _add(span: str) -> None:
        normalized = span.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            anchors.append(normalized)
            seen.add(key)

    # 1. Quoted spans — always treat as exact user intent.
    for match in _QUOTED_SPAN_RE.finditer(text):
        _add(match.group(1) or match.group(2) or match.group(3) or "")

    # 2. Multi-word capitalized spans (two or more consecutive Title-case words).
    for match in _CAPITALIZED_MULTI_WORD_RE.finditer(text):
        _add(match.group(0))

    # 3. Technical tokens (path-like, ALL-CAPS-HYPHEN, CamelCase).
    for match in _TECHNICAL_TOKEN_RE.finditer(text):
        _add(match.group(0))

    return anchors


def _guard_exact_spans(user_input: str, search_query: str | None) -> str | None:
    """Guard the reformulated search query against exact-span drift.

    Extracts risky exact spans (quoted strings, multi-word proper-name sequences,
    technical identifiers) from ``user_input`` and checks whether each is present
    (case-insensitively) in ``search_query``.

    Returns ``search_query`` unchanged when:
    - no risky spans are found in user_input (broad informational queries), or
    - all spans survive in the reformulation.

    Returns ``None`` when any span is absent, signalling the caller
    (_prepare_web_search_route) to fall back to the original user input so the
    exact entity name reaches the search engine.

    Explicitly required by mission P5-1.  Complies with mandatory_rules.md
    rule 5 allowed exception and rule 10: explicitly justified, visibly scoped,
    easy to audit, easy to remove.
    """
    if not search_query:
        return None

    anchors = _extract_anchor_spans(user_input)
    if not anchors:
        # No risky spans detected — normal reformulation is safe to use.
        return search_query

    lowered_query = search_query.lower()
    if all(anchor.lower() in lowered_query for anchor in anchors):
        # All anchors survived in the reformulation — reformulation is faithful.
        return search_query

    # At least one exact span was dropped during reformulation.
    # Return None to signal the caller to fall back to the original user input,
    # which preserves the exact entity name(s) for the search engine.
    return None


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
        if model_profile_name == ROUTER_PROFILE_NAME:
            profile = resolve_router_profile(settings)
        else:
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
            timeout_seconds=settings.router.timeout_seconds,
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
