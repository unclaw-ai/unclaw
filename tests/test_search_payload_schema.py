"""Tests for the formal search payload schema (P1-5).

Verifies that:
- SearchWebPayload TypedDict keys match what the producer constructs
- search_grounding consumers accept and correctly parse a schema-compliant payload
- research_flow consumers accept and correctly parse a schema-compliant payload
"""

from __future__ import annotations

from datetime import date

from unclaw.core.research_flow import (
    append_search_sources_section,
    build_tool_history_content,
)
from unclaw.core.search_grounding import (
    build_search_grounding_context,
    build_search_tool_history_summary,
    parse_search_tool_history,
    shape_search_backed_reply,
)
from unclaw.tools.contracts import (
    SearchDisplaySourcePayload,
    SearchEvidencePayload,
    SearchFindingPayload,
    SearchResultSourcePayload,
    SearchWebPayload,
    ToolResult,
)


def _build_minimal_payload() -> SearchWebPayload:
    """Build a minimal but complete schema-compliant search payload."""
    return SearchWebPayload(
        query="test query",
        provider="DuckDuckGo",
        initial_result_count=2,
        considered_candidate_count=2,
        fetch_attempt_count=1,
        fetch_success_count=1,
        evidence_count=1,
        statement_count=1,
        fact_cluster_count=1,
        finding_count=1,
        summary_points=["Test finding about the topic."],
        display_sources=[
            SearchDisplaySourcePayload(
                title="Test Source",
                url="https://example.com/article",
            ),
        ],
        synthesized_findings=[
            SearchFindingPayload(
                text="Test finding about the topic.",
                score=7.5,
                support_count=2,
                source_titles=["Test Source"],
                source_urls=["https://example.com/article"],
            ),
        ],
        results=[
            SearchResultSourcePayload(
                title="Test Source",
                url="https://example.com/article",
                takeaway="Relevant article about the topic.",
                depth=0,
                fetched=True,
                evidence_count=1,
                fetch_error=None,
                used_snippet_fallback=False,
                usefulness=8.0,
            ),
        ],
        evidence=[
            SearchEvidencePayload(
                text="Test finding about the topic.",
                url="https://example.com/article",
                source_title="Test Source",
                score=7.5,
                depth=0,
                query_relevance=6.0,
                evidence_quality=5.0,
                novelty=1.0,
                supporting_urls=["https://example.com/article"],
                supporting_titles=["Test Source"],
            ),
        ],
    )


def test_schema_keys_match_producer_payload() -> None:
    """SearchWebPayload TypedDict keys cover all keys the producer emits."""
    payload = _build_minimal_payload()
    expected_keys = {
        "query",
        "provider",
        "initial_result_count",
        "considered_candidate_count",
        "fetch_attempt_count",
        "fetch_success_count",
        "evidence_count",
        "statement_count",
        "fact_cluster_count",
        "finding_count",
        "summary_points",
        "display_sources",
        "synthesized_findings",
        "results",
        "evidence",
    }
    assert set(payload.keys()) == expected_keys


def test_search_grounding_context_from_typed_payload() -> None:
    """build_search_grounding_context accepts a schema-compliant payload."""
    payload = _build_minimal_payload()
    context = build_search_grounding_context(
        payload,
        query="test query",
        current_date=date(2026, 3, 15),
    )
    assert context is not None
    assert context.query == "test query"
    assert len(context.supported_findings) >= 1
    assert context.supported_findings[0].text == "Test finding about the topic."
    assert len(context.display_sources) == 1
    assert context.display_sources[0] == (
        "Test Source",
        "https://example.com/article",
    )


def test_search_tool_history_summary_from_typed_payload() -> None:
    """build_search_tool_history_summary accepts a schema-compliant payload."""
    payload = _build_minimal_payload()
    lines = build_search_tool_history_summary(
        payload=payload,
        query="test query",
        current_date=date(2026, 3, 15),
    )
    assert len(lines) > 0
    assert any("test query" in line.lower() for line in lines)


def test_search_tool_history_summary_round_trips_through_history_parser() -> None:
    """Grounding history formatting remains parseable after refactors."""
    payload = _build_minimal_payload()
    payload["query"] = "who is Casey Hart"
    payload["summary_points"] = [
        "Casey Hart is a robotics engineer.",
        "One profile lists the handle @caseyhart.",
    ]
    payload["display_sources"] = [
        SearchDisplaySourcePayload(
            title="Company Bio",
            url="https://example.com/casey",
        ),
        SearchDisplaySourcePayload(
            title="Community Profile",
            url="https://example.com/casey-community",
        ),
    ]
    payload["synthesized_findings"] = [
        SearchFindingPayload(
            text="Casey Hart is a robotics engineer.",
            score=8.1,
            support_count=2,
            source_titles=["Company Bio", "Community Profile"],
            source_urls=[
                "https://example.com/casey",
                "https://example.com/casey-community",
            ],
        ),
        SearchFindingPayload(
            text="One profile lists the handle @caseyhart.",
            score=4.2,
            support_count=1,
            source_titles=["Community Profile"],
            source_urls=["https://example.com/casey-community"],
        ),
    ]
    payload["results"] = [
        SearchResultSourcePayload(
            title="Company Bio",
            url="https://example.com/casey",
            takeaway="Official biography.",
            depth=0,
            fetched=True,
            evidence_count=1,
            fetch_error=None,
            used_snippet_fallback=False,
            usefulness=8.5,
        ),
        SearchResultSourcePayload(
            title="Community Profile",
            url="https://example.com/casey-community",
            takeaway="One community profile with a social handle mention.",
            depth=0,
            fetched=True,
            evidence_count=1,
            fetch_error=None,
            used_snippet_fallback=False,
            usefulness=4.0,
        ),
    ]
    payload["evidence"] = [
        SearchEvidencePayload(
            text="Casey Hart was born on 1990-07-12.",
            url="https://example.com/casey",
            source_title="Company Bio",
            score=7.2,
            depth=0,
            query_relevance=6.0,
            evidence_quality=5.0,
            novelty=1.0,
            supporting_urls=["https://example.com/casey"],
            supporting_titles=["Company Bio"],
        ),
    ]

    lines = build_search_tool_history_summary(
        payload=payload,
        query="who is Casey Hart",
        current_date=date(2026, 3, 15),
    )
    content = "\n".join(("Tool: search_web", "Outcome: success", "", *lines))

    parsed = parse_search_tool_history(content)

    assert parsed is not None
    assert parsed.query == "who is Casey Hart"
    assert parsed.current_date == date(2026, 3, 15)
    assert parsed.birth_date == date(1990, 7, 12)
    assert tuple(finding.text for finding in parsed.supported_findings) == (
        "Casey Hart is a robotics engineer.",
    )
    assert tuple(finding.text for finding in parsed.uncertain_findings) == (
        "One profile lists the handle @caseyhart.",
    )
    assert parsed.display_sources == (
        ("Company Bio", "https://example.com/casey"),
        ("Community Profile", "https://example.com/casey-community"),
    )


def test_shape_search_backed_reply_from_typed_payload() -> None:
    """shape_search_backed_reply accepts a schema-compliant payload."""
    payload = _build_minimal_payload()
    result = shape_search_backed_reply(
        "Test finding about the topic.",
        payload=payload,
        query="test query",
        current_date=date(2026, 3, 15),
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_append_search_sources_section_from_typed_payload() -> None:
    """append_search_sources_section accepts a schema-compliant payload."""
    payload = _build_minimal_payload()
    result = append_search_sources_section(
        "Here is the answer.",
        payload=payload,
    )
    assert "Sources:" in result
    assert "https://example.com/article" in result


def test_build_tool_history_content_from_typed_payload() -> None:
    """build_tool_history_content works with a schema-compliant payload."""
    payload = _build_minimal_payload()
    tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text="Search results text.",
        payload=payload,
    )
    content = build_tool_history_content(tool_result)
    assert "search_web" in content
    assert "success" in content


def test_nested_typed_dicts_are_plain_dicts_at_runtime() -> None:
    """TypedDicts remain plain dicts at runtime — no behavior change."""
    payload = _build_minimal_payload()
    assert isinstance(payload, dict)
    assert isinstance(payload["display_sources"][0], dict)
    assert isinstance(payload["synthesized_findings"][0], dict)
    assert isinstance(payload["results"][0], dict)
    assert isinstance(payload["evidence"][0], dict)
