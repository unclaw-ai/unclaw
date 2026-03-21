"""Core contracts for Unclaw tool execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict


class ToolPermissionLevel(StrEnum):
    """Minimal permission categories for built-in tools."""

    LOCAL_READ = "local_read"
    LOCAL_WRITE = "local_write"
    NETWORK = "network"


type ToolArgumentType = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "array",
    "object",
]


@dataclass(frozen=True, slots=True)
class ToolArgumentSpec:
    """Describe one tool argument for provider schema emission."""

    description: str
    value_type: ToolArgumentType = "string"


type ToolArgumentDefinition = str | ToolArgumentSpec


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Describe one callable tool exposed to the runtime."""

    name: str
    description: str
    permission_level: ToolPermissionLevel
    arguments: Mapping[str, ToolArgumentDefinition]


_TOOL_ARGUMENT_TYPE_ALIASES: dict[str, ToolArgumentType] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
}


def resolve_tool_argument_spec(argument: ToolArgumentDefinition) -> ToolArgumentSpec:
    """Return a normalized tool argument spec.

    Legacy plain-string entries stay supported:
    - recognized type aliases like ``"int"`` map to structured types
    - every other string is treated as a description with string typing
    """

    if isinstance(argument, ToolArgumentSpec):
        return ToolArgumentSpec(
            description=argument.description,
            value_type=_normalize_explicit_tool_argument_type(argument.value_type),
        )

    normalized_type = _TOOL_ARGUMENT_TYPE_ALIASES.get(argument.strip().lower())
    return ToolArgumentSpec(
        description=argument,
        value_type="string" if normalized_type is None else normalized_type,
    )


def _normalize_explicit_tool_argument_type(value_type: str) -> ToolArgumentType:
    normalized = _TOOL_ARGUMENT_TYPE_ALIASES.get(value_type.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported tool argument type: {value_type!r}.")
    return normalized


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One requested tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Structured result returned by a tool handler."""

    tool_name: str
    success: bool
    output_text: str
    payload: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def ok(
        cls,
        *,
        tool_name: str,
        output_text: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            tool_name=tool_name,
            success=True,
            output_text=output_text,
            payload=dict(payload) if payload is not None else None,
            error=None,
        )

    @classmethod
    def failure(
        cls,
        *,
        tool_name: str,
        error: str,
        output_text: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        resolved_output = output_text if output_text is not None else error
        return cls(
            tool_name=tool_name,
            success=False,
            output_text=resolved_output,
            payload=dict(payload) if payload is not None else None,
            error=error,
        )


type ToolHandler = Callable[[ToolCall], ToolResult]


# ---------------------------------------------------------------------------
# Search payload schema — shared contract between the web-search tool
# (producer) and search_grounding / research_flow (consumers).
# ---------------------------------------------------------------------------


class SearchFindingPayload(TypedDict):
    """One synthesized finding in the search tool payload."""

    text: str
    score: float
    support_count: int
    source_titles: list[str]
    source_urls: list[str]


class SearchDisplaySourcePayload(TypedDict):
    """One display source entry in the search tool payload."""

    title: str
    url: str


class SearchResultSourcePayload(TypedDict):
    """One detailed result source in the search tool payload."""

    title: str
    url: str
    takeaway: str
    depth: int
    fetched: bool
    evidence_count: int
    fetch_error: str | None
    used_snippet_fallback: bool
    usefulness: float


class SearchEvidencePayload(TypedDict):
    """One evidence entry in the search tool payload."""

    text: str
    url: str
    source_title: str
    score: float
    depth: int
    query_relevance: float
    evidence_quality: float
    novelty: float
    supporting_urls: list[str]
    supporting_titles: list[str]


class SearchWebPayload(TypedDict):
    """Formal schema for the search_web tool result payload.

    Produced by web_tools.search_web(), consumed by search_grounding
    and research_flow.  Any key change must be reflected here so both
    sides stay in sync.
    """

    query: str
    provider: str
    initial_result_count: int
    considered_candidate_count: int
    fetch_attempt_count: int
    fetch_success_count: int
    evidence_count: int
    statement_count: int
    fact_cluster_count: int
    finding_count: int
    summary_points: list[str]
    display_sources: list[SearchDisplaySourcePayload]
    synthesized_findings: list[SearchFindingPayload]
    results: list[SearchResultSourcePayload]
    evidence: list[SearchEvidencePayload]


class WeatherResolvedLocationPayload(TypedDict):
    """Canonical location metadata returned by the weather tool."""

    name: str
    latitude: float
    longitude: float
    timezone: str
    admin1: str | None
    country: str | None


class WeatherCurrentConditionsPayload(TypedDict):
    """Current weather conditions returned by the weather tool."""

    observed_at: str
    weather_code: int
    weather_description: str
    temperature_c: float
    apparent_temperature_c: float
    precipitation_mm: float
    wind_speed_kph: float
    wind_direction_degrees: int


class WeatherDailyForecastPayload(TypedDict):
    """One daily forecast entry returned by the weather tool."""

    date: str
    weather_code: int
    weather_description: str
    temperature_min_c: float
    temperature_max_c: float
    precipitation_probability_max_pct: int | None
    precipitation_sum_mm: float | None
    wind_speed_max_kph: float | None


class WeatherLookupPayload(TypedDict):
    """Formal schema for the dedicated weather tool result payload."""

    provider: str
    location_query: str
    forecast_days: int
    resolved_location: WeatherResolvedLocationPayload
    current: WeatherCurrentConditionsPayload | None
    daily_forecast: list[WeatherDailyForecastPayload]


__all__ = [
    "SearchDisplaySourcePayload",
    "SearchEvidencePayload",
    "SearchFindingPayload",
    "SearchResultSourcePayload",
    "SearchWebPayload",
    "ToolArgumentDefinition",
    "ToolArgumentSpec",
    "ToolArgumentType",
    "ToolCall",
    "ToolDefinition",
    "ToolHandler",
    "ToolPermissionLevel",
    "ToolResult",
    "WeatherCurrentConditionsPayload",
    "WeatherDailyForecastPayload",
    "WeatherLookupPayload",
    "WeatherResolvedLocationPayload",
    "resolve_tool_argument_spec",
]
