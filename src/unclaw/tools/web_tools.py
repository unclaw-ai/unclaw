"""Lightweight web tools for the early Unclaw runtime.

This module is the stable public entry-point for web tools. Internal
implementation is split across focused submodules:

- web_safety   – SSRF protection, IP blocking, hostname validation
- web_html     – HTML parsing, link extraction
- web_text     – text normalization, tokenization, noise detection
- web_fetch    – HTTP fetching, content decoding
- web_search   – DuckDuckGo integration, URL classification, ranking
- web_retrieval – iterative retrieval, evidence extraction
- web_synthesis – evidence clustering, finding synthesis, output formatting
- web_research – model-driven 3-layer research pipeline
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError

from pathlib import Path

from unclaw.async_utils import run_blocking
from unclaw.tools.contracts import (
    FastSearchWebPayload,
    ResearchMergedNotePayload,
    ResearchSourceNotePayload,
    SearchDisplaySourcePayload,
    SearchEvidencePayload,
    SearchFindingPayload,
    SearchResultSourcePayload,
    SearchWebPayload,
    ToolCall,
    ToolArgumentSpec,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_fetch import (
    _DEFAULT_MAX_FETCH_CHARS,
    _DEFAULT_TIMEOUT_SECONDS,
    _fetch_text_document,
    _format_text_excerpt,
)
from unclaw.tools.web_research import (
    MAIN_RESEARCH_BUDGET,
    ResearchBudget,
    ResearchConfig,
    build_fast_grounding_note,
    format_research_output,
    resolve_research_budget,
    run_research_pipeline,
)
from unclaw.tools.web_retrieval import (
    _RetrievalBudget,
    _run_iterative_retrieval,
)
from unclaw.tools.web_safety import _BlockedFetchTargetError, _is_supported_url
from unclaw.tools.web_search import (
    _DEFAULT_MAX_SEARCH_RESULTS,
    _MAX_SEARCH_RESULTS,
    _SEARCH_PROVIDER_NAME,
    _build_search_query,
    _build_staged_search_queries,
    _canonicalize_url,
    _deduplicate_search_results,
    _parse_duckduckgo_html_results,
    _rank_search_results,
    _search_results_look_weak,
    _search_public_web,
)
from unclaw.tools.web_text import _fold_for_match
from unclaw.tools.web_synthesis import (
    _format_search_results,
    _select_output_sources,
    _synthesize_search_knowledge,
)

FETCH_URL_TEXT_DEFINITION = ToolDefinition(
    name="fetch_url_text",
    description="Fetch a URL and extract readable text content.",
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "url": ToolArgumentSpec(description="HTTP or HTTPS URL to fetch."),
        "max_chars": ToolArgumentSpec(
            description="Optional maximum number of characters to return.",
            value_type="integer",
        ),
        "timeout_seconds": ToolArgumentSpec(
            description="Optional request timeout in seconds.",
            value_type="number",
        ),
    },
)

SEARCH_WEB_DEFINITION = ToolDefinition(
    name="search_web",
    description=(
        "Search the public web with bounded iterative retrieval and return a compact summary."
    ),
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "query": ToolArgumentSpec(description="Search query string."),
        "max_results": ToolArgumentSpec(
            description="Optional maximum number of search results to consider.",
            value_type="integer",
        ),
        "timeout_seconds": ToolArgumentSpec(
            description="Optional request timeout in seconds.",
            value_type="number",
        ),
    },
    required_arguments=frozenset({"query"}),
)

FAST_WEB_SEARCH_DEFINITION = ToolDefinition(
    name="fast_web_search",
    description=(
        "Quick lightweight web grounding probe for entity resolution. "
        "Returns a tiny grounding note — use before full search_web when "
        "unsure about a person, place, organization, or product name."
    ),
    permission_level=ToolPermissionLevel.NETWORK,
    arguments={
        "query": ToolArgumentSpec(
            description="Entity or topic to quickly ground (e.g. a person name).",
        ),
    },
    required_arguments=frozenset({"query"}),
)

_FAST_SEARCH_MAX_RESULTS = 3
_FAST_SEARCH_TIMEOUT_SECONDS = 6.0


def _summarize_fast_grounding_snippets(
    *,
    query: str,
    snippets: list[tuple[str, str, str]],
) -> tuple[str, int]:
    if not snippets:
        return "no_results", 0

    search_query = _build_search_query(query)
    if not search_query.entity_tokens:
        return "exact", len(snippets)

    full_match_found = False
    partial_match_found = False
    supported_point_count = 0

    for title, url, snippet_text in snippets:
        folded_metadata = _fold_for_match(f"{title} {snippet_text} {url}")
        metadata_tokens = folded_metadata.split()
        token_hits = sum(
            1 for token in search_query.entity_tokens if token in metadata_tokens
        )
        normalized_entity_hit = bool(
            search_query.normalized_entity
            and search_query.normalized_entity in folded_metadata
        )
        aligned = normalized_entity_hit or token_hits >= len(search_query.entity_tokens)

        if aligned:
            supported_point_count += 1
            full_match_found = True
            continue

        if token_hits > 0:
            partial_match_found = True

    if full_match_found:
        return "exact", supported_point_count
    if partial_match_found:
        return "partial", 0
    return "mismatch", 0


def register_web_tools(
    registry: ToolRegistry,
    *,
    allow_private_networks: bool = False,
    research_config: ResearchConfig | None = None,
    research_budget: ResearchBudget | None = None,
    workspace_base_dir: Path | None = None,
) -> None:
    """Register the built-in lightweight web tools.

    When ``research_config`` is provided, the search_web tool uses the
    3-layer model-driven research pipeline for condensation.

    When ``workspace_base_dir`` is provided, search_web persists research
    artifacts under that directory (file-backed workspace).
    """
    effective_budget = research_budget or MAIN_RESEARCH_BUDGET
    effective_config = research_config

    def fetch_handler(call: ToolCall) -> ToolResult:
        return fetch_url_text(
            call,
            allow_private_networks=allow_private_networks,
        )

    def search_handler(call: ToolCall) -> ToolResult:
        return search_web(
            call,
            research_config=effective_config,
            research_budget=effective_budget,
            workspace_base_dir=workspace_base_dir,
        )

    def fast_search_handler(call: ToolCall) -> ToolResult:
        return fast_web_search(call, budget=effective_budget)

    registry.register(FETCH_URL_TEXT_DEFINITION, fetch_handler)
    registry.register(SEARCH_WEB_DEFINITION, search_handler)
    registry.register(FAST_WEB_SEARCH_DEFINITION, fast_search_handler)


def _run_bounded_staged_search_full(
    *,
    query: str,
    max_results: int,
    timeout_seconds: float,
    fast_mode: bool,
) -> tuple[Any, list[dict[str, str]], tuple[str, ...]]:
    """Run staged search and return (search_query, ranked_results, executed_queries).

    This variant collects executed staged queries for observability/debugging.
    Calls search-layer functions through the imported names in this module so
    that test patches applied to unclaw.tools.web_tools.* continue to work.
    """
    search_query = _build_search_query(query)
    staged_queries = _build_staged_search_queries(query, fast_mode=fast_mode)
    collected_results: list[dict[str, str]] = []
    executed_queries: list[str] = []

    for pass_number, staged_query in enumerate(staged_queries, start=1):
        response_text = _search_public_web(
            query=staged_query,
            timeout_seconds=timeout_seconds,
        )
        raw_results = _parse_duckduckgo_html_results(
            response_text,
            max_results=max_results,
        )
        for result in raw_results:
            annotated_result = dict(result)
            annotated_result["_pass_number"] = str(pass_number)
            collected_results.append(annotated_result)
        executed_queries.append(staged_query)

        ranked_results = _rank_search_results(
            _deduplicate_search_results(collected_results),
            query=search_query,
        )
        if not _search_results_look_weak(ranked_results[:max_results], query=search_query):
            break

    ranked_results = _rank_search_results(
        _deduplicate_search_results(collected_results),
        query=search_query,
    )
    return search_query, ranked_results[:max_results], tuple(executed_queries)


def _run_bounded_staged_search(
    *,
    query: str,
    max_results: int,
    timeout_seconds: float,
    fast_mode: bool,
) -> tuple[Any, list[dict[str, str]]]:
    """Run staged search and return (search_query, ranked_results).

    Backward-compatible 2-tuple wrapper around _run_bounded_staged_search_full.
    """
    search_query, ranked_results, _executed = _run_bounded_staged_search_full(
        query=query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
        fast_mode=fast_mode,
    )
    return search_query, ranked_results


def _prioritize_research_pages(
    pages: dict[str, tuple[str, str]] | None,
    *,
    outcome,
) -> dict[str, tuple[str, str]] | None:
    if not pages:
        return pages

    ordered_pages: dict[str, tuple[str, str]] = {}
    by_canonical_url: dict[str, tuple[str, tuple[str, str]]] = {}
    for url, payload in pages.items():
        by_canonical_url[_canonicalize_url(url) or url] = (url, payload)

    for source in outcome.sources:
        canonical_source_url = _canonicalize_url(source.url) or source.url
        match = by_canonical_url.get(canonical_source_url)
        if match is None:
            continue
        page_url, payload = match
        ordered_pages[page_url] = payload

    for page_url, payload in pages.items():
        ordered_pages.setdefault(page_url, payload)

    return ordered_pages


def fetch_url_text(
    call: ToolCall,
    *,
    allow_private_networks: bool = False,
) -> ToolResult:
    """Fetch a URL and return a compact text version of the response body."""
    tool_name = FETCH_URL_TEXT_DEFINITION.name

    try:
        url = _read_string_argument(call.arguments, "url")
        max_chars = _read_positive_int_argument(
            call.arguments,
            "max_chars",
            default=_DEFAULT_MAX_FETCH_CHARS,
        )
        timeout_seconds = _read_positive_number_argument(
            call.arguments,
            "timeout_seconds",
            default=_DEFAULT_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    if not _is_supported_url(url):
        return ToolResult.failure(
            tool_name=tool_name,
            error="Argument 'url' must be a valid HTTP or HTTPS URL.",
        )

    try:
        document = _fetch_text_document(
            url,
            max_chars=max_chars,
            timeout_seconds=timeout_seconds,
            allow_private_networks=allow_private_networks,
            accept_header=(
                "text/plain, text/html, application/json;q=0.9, */*;q=0.1"
            ),
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except HTTPError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"HTTP error {exc.code} while fetching '{url}': {exc.reason}",
        )
    except URLError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not fetch '{url}': {exc.reason}",
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not fetch '{url}': {exc}",
        )

    output_text = "\n".join(
        [
            f"URL: {document.resolved_url}",
            f"Status: {document.status_code or 'unknown'}",
            f"Content-Type: {document.content_type}",
            "",
            _format_text_excerpt(
                document.text_excerpt,
                truncated=document.truncated,
            ),
        ]
    )
    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload={
            "requested_url": url,
            "resolved_url": document.resolved_url,
            "status_code": document.status_code,
            "content_type": document.content_type,
            "truncated": document.truncated,
        },
    )


async def fetch_url_text_async(
    call: ToolCall,
    *,
    allow_private_networks: bool = False,
) -> ToolResult:
    """Expose the blocking URL fetch tool through an awaitable boundary."""

    return await run_blocking(
        fetch_url_text,
        call,
        allow_private_networks=allow_private_networks,
    )


def search_web(
    call: ToolCall,
    *,
    research_config: ResearchConfig | None = None,
    research_budget: ResearchBudget | None = None,
    workspace_base_dir: Path | None = None,
) -> ToolResult:
    """Search the public web, iteratively fetch bounded sources, and summarize evidence.

    When research_config is provided, runs the 3-layer research pipeline:
    Layer A — discovery + bounded fetch (existing retrieval)
    Layer B — model-driven per-source condensation
    Layer C — model-driven merged research note

    When workspace_base_dir is provided, research artifacts are persisted to
    disk under a timestamped directory.  The payload includes workspace_id
    and workspace_dir for callers to locate and reuse the artifacts.
    Persistence failures are silently swallowed — they never break the tool.
    """
    tool_name = SEARCH_WEB_DEFINITION.name

    try:
        query = _read_string_argument(call.arguments, "query")
        max_results = _read_limited_positive_int_argument(
            call.arguments,
            "max_results",
            default=_DEFAULT_MAX_SEARCH_RESULTS,
            maximum=_MAX_SEARCH_RESULTS,
        )
        timeout_seconds = _read_positive_number_argument(
            call.arguments,
            "timeout_seconds",
            default=_DEFAULT_TIMEOUT_SECONDS,
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    try:
        search_query, ranked_results, executed_queries = _run_bounded_staged_search_full(
            query=query,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            fast_mode=False,
        )
    except _BlockedFetchTargetError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))
    except HTTPError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=(
                f"Search provider returned HTTP error {exc.code} "
                f"for '{query}': {exc.reason}"
            ),
        )
    except URLError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not search the web for '{query}': {exc.reason}",
        )
    except OSError as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not search the web for '{query}': {exc}",
        )
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    # Collect page texts for the research pipeline when config is available.
    page_text_collector: dict[str, tuple[str, str]] | None = (
        {} if research_config is not None else None
    )

    outcome = _run_iterative_retrieval(
        results=ranked_results,
        query=search_query,
        timeout_seconds=timeout_seconds,
        budget=_RetrievalBudget(max_initial_results=max_results),
        page_text_collector=page_text_collector,
    )

    # --- Deterministic synthesis (always runs, preserves baseline) ---
    synthesis = _synthesize_search_knowledge(
        outcome.evidence_items,
        query=search_query,
    )
    summary_points = tuple(finding.text for finding in synthesis.findings)
    display_sources = _select_output_sources(
        sources=outcome.sources,
        synthesis=synthesis,
    )

    # --- Research pipeline (Layer B + C) when configured ---
    workspace = None
    effective_budget = research_budget or MAIN_RESEARCH_BUDGET
    if research_config is not None and page_text_collector:
        try:
            prioritized_pages = _prioritize_research_pages(
                page_text_collector,
                outcome=outcome,
            )
            workspace = run_research_pipeline(
                query=query,
                fetched_pages=prioritized_pages or page_text_collector,
                budget=effective_budget,
                config=research_config,
            )
        except Exception:
            # Research pipeline failure must not break the tool.
            workspace = None

    # --- File-backed workspace persistence ---
    workspace_ref = None
    if workspace is not None and workspace_base_dir is not None:
        try:
            from unclaw.tools.web_workspace import (
                create_workspace_dir,
                persist_research_workspace,
                prune_old_workspaces,
            )

            ws_path, ws_id = create_workspace_dir(workspace_base_dir, query)
            persist_research_workspace(
                workspace_dir=ws_path,
                workspace=workspace,
                query=query,
                executed_queries=executed_queries,
            )
            prune_old_workspaces(workspace_base_dir)
            from unclaw.tools.web_workspace import SearchWorkspaceRef

            workspace_ref = SearchWorkspaceRef(
                workspace_id=ws_id,
                workspace_dir=str(ws_path),
            )
        except Exception:
            # Persistence failure must never break the tool.
            workspace_ref = None

    # --- Build output text ---
    # Use research note format only when model-driven condensation produced
    # the merged note.  The deterministic fallback does not have the
    # noise filtering of the evidence synthesis pipeline, so falling back
    # to the proven format is safer.
    if (
        workspace is not None
        and workspace.has_merged_note
        and workspace.merged_note is not None
        and workspace.merged_note.model_generated
    ):
        fetch_stats = {
            "considered_candidate_count": outcome.considered_candidate_count,
            "fetch_attempt_count": outcome.fetch_attempt_count,
            "fetch_success_count": outcome.fetch_success_count,
        }
        output_text = format_research_output(
            workspace=workspace,
            query=query,
            fetch_stats=fetch_stats,
        )
    else:
        output_text = _format_search_results(
            query=query,
            outcome=outcome,
            summary_points=summary_points,
            synthesis=synthesis,
        )

    # --- Build payload (always includes baseline fields) ---
    payload: dict[str, Any] = {
        "query": query,
        "provider": _SEARCH_PROVIDER_NAME,
        "initial_result_count": outcome.initial_result_count,
        "considered_candidate_count": outcome.considered_candidate_count,
        "fetch_attempt_count": outcome.fetch_attempt_count,
        "fetch_success_count": outcome.fetch_success_count,
        "evidence_count": len(outcome.evidence_items),
        "statement_count": len(synthesis.statements),
        "fact_cluster_count": len(synthesis.fact_clusters),
        "finding_count": len(synthesis.findings),
        "summary_points": list(summary_points),
        "display_sources": [
            SearchDisplaySourcePayload(
                title=source.title,
                url=source.url,
            )
            for source in display_sources
        ],
        "synthesized_findings": [
            SearchFindingPayload(
                text=finding.text,
                score=finding.score,
                support_count=finding.support_count,
                source_titles=list(finding.source_titles),
                source_urls=list(finding.source_urls),
            )
            for finding in synthesis.findings
        ],
        "results": [
            SearchResultSourcePayload(
                title=source.title,
                url=source.url,
                takeaway=source.takeaway,
                depth=source.depth,
                fetched=source.fetched,
                evidence_count=source.evidence_count,
                fetch_error=source.fetch_error,
                used_snippet_fallback=source.used_snippet_fallback,
                usefulness=source.usefulness,
            )
            for source in outcome.sources
        ],
        "evidence": [
            SearchEvidencePayload(
                text=evidence.text,
                url=evidence.url,
                source_title=evidence.source_title,
                score=evidence.score,
                depth=evidence.depth,
                query_relevance=evidence.query_relevance,
                evidence_quality=evidence.evidence_quality,
                novelty=evidence.novelty,
                supporting_urls=list(evidence.supporting_urls),
                supporting_titles=list(evidence.supporting_titles),
            )
            for evidence in outcome.evidence_items
        ],
    }

    # Enrich payload with research pipeline artifacts when available.
    if workspace is not None:
        payload["research_source_notes"] = [
            ResearchSourceNotePayload(
                url=note.url,
                title=note.title,
                condensed_text=note.condensed_text,
                model_generated=note.model_generated,
            )
            for note in workspace.source_notes
        ]
        if workspace.merged_note is not None:
            payload["research_merged_note"] = ResearchMergedNotePayload(
                text=workspace.merged_note.text,
                source_count=workspace.merged_note.source_count,
                model_generated=workspace.merged_note.model_generated,
            )
        payload["research_model_driven"] = any(
            note.model_generated for note in workspace.source_notes
        ) or (
            workspace.merged_note is not None
            and workspace.merged_note.model_generated
        )

    # Include file-backed workspace reference when persistence succeeded.
    if workspace_ref is not None:
        payload["workspace_id"] = workspace_ref.workspace_id
        payload["workspace_dir"] = workspace_ref.workspace_dir

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=output_text,
        payload=payload,
    )


async def search_web_async(call: ToolCall) -> ToolResult:
    """Expose the blocking web-search tool through an awaitable boundary."""

    return await run_blocking(search_web, call)


def fast_web_search(
    call: ToolCall,
    *,
    budget: ResearchBudget | None = None,
) -> ToolResult:
    """Quick lightweight web grounding probe for entity resolution.

    Fetches 1-3 search results and builds a tiny grounding note.
    No heavy multi-source merge or model-driven condensation.
    Designed for small local models to resolve ambiguous entities.
    """
    tool_name = FAST_WEB_SEARCH_DEFINITION.name
    effective_budget = budget or MAIN_RESEARCH_BUDGET

    try:
        query = _read_string_argument(call.arguments, "query")
    except ValueError as exc:
        return ToolResult.failure(tool_name=tool_name, error=str(exc))

    max_results = effective_budget.fast_grounding_max_results
    max_chars = effective_budget.fast_grounding_max_chars

    try:
        _search_query, ranked_results = _run_bounded_staged_search(
            query=query,
            max_results=max_results,
            timeout_seconds=_FAST_SEARCH_TIMEOUT_SECONDS,
            fast_mode=True,
        )
    except (_BlockedFetchTargetError, HTTPError, URLError, OSError, ValueError) as exc:
        return ToolResult.failure(
            tool_name=tool_name,
            error=f"Could not search for '{query}': {exc}",
        )

    snippets: list[tuple[str, str, str]] = [
        (result.get("title", ""), result.get("url", ""), result.get("snippet", ""))
        for result in ranked_results
        if result.get("title") and result.get("snippet")
    ]

    grounding_note = build_fast_grounding_note(
        query=query,
        snippets=snippets[:max_results],
        max_chars=max_chars,
    )
    match_quality, supported_point_count = _summarize_fast_grounding_snippets(
        query=query,
        snippets=snippets[:max_results],
    )

    payload: FastSearchWebPayload = {
        "query": query,
        "provider": _SEARCH_PROVIDER_NAME,
        "result_count": len(snippets[:max_results]),
        "match_quality": match_quality,
        "supported_point_count": supported_point_count,
        "grounding_note": grounding_note,
    }

    return ToolResult.ok(
        tool_name=tool_name,
        output_text=grounding_note,
        payload=payload,
    )


async def fast_web_search_async(call: ToolCall) -> ToolResult:
    """Expose the blocking fast web search through an awaitable boundary."""

    return await run_blocking(fast_web_search, call)


# --- Argument validation helpers ---


def _read_string_argument(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Argument '{key}' must be a non-empty string.")
    return value.strip()


def _read_positive_int_argument(
    arguments: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Argument '{key}' must be an integer.")
    if value < 1:
        raise ValueError(f"Argument '{key}' must be greater than zero.")
    return value


def _read_limited_positive_int_argument(
    arguments: Mapping[str, Any],
    key: str,
    *,
    default: int,
    maximum: int,
) -> int:
    value = _read_positive_int_argument(arguments, key, default=default)
    if value > maximum:
        raise ValueError(
            f"Argument '{key}' must be less than or equal to {maximum}."
        )
    return value


def _read_positive_number_argument(
    arguments: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Argument '{key}' must be a number.")
    if value <= 0:
        raise ValueError(f"Argument '{key}' must be greater than zero.")
    return float(value)


__all__ = [
    "FAST_WEB_SEARCH_DEFINITION",
    "FETCH_URL_TEXT_DEFINITION",
    "SEARCH_WEB_DEFINITION",
    "fast_web_search",
    "fast_web_search_async",
    "fetch_url_text",
    "fetch_url_text_async",
    "search_web",
    "search_web_async",
    "register_web_tools",
]
