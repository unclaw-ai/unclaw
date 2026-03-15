"""Core contracts for Unclaw tool execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict


class ToolPermissionLevel(StrEnum):
    """Minimal permission categories for built-in tools."""

    LOCAL_READ = "local_read"
    NETWORK = "network"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Describe one callable tool exposed to the runtime."""

    name: str
    description: str
    permission_level: ToolPermissionLevel
    arguments: Mapping[str, str]


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


__all__ = [
    "SearchDisplaySourcePayload",
    "SearchEvidencePayload",
    "SearchFindingPayload",
    "SearchResultSourcePayload",
    "SearchWebPayload",
    "ToolCall",
    "ToolDefinition",
    "ToolHandler",
    "ToolPermissionLevel",
    "ToolResult",
]
