"""Focused quality tests for search query discipline and source hygiene."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from unclaw.tools.web_retrieval import _score_fetched_page
from unclaw.tools.web_search import (
    _build_search_query,
    _build_staged_search_queries,
    _rank_search_results,
)
from unclaw.tools.web_tools import _run_bounded_staged_search


def test_first_pass_preserves_exact_entity_surface() -> None:
    queries = _build_staged_search_queries(
        "biographie de Marine Leleu",
        fast_mode=False,
    )

    assert queries[0] == '"Marine Leleu"'
    assert all("le pen" not in query.casefold() for query in queries)


def test_staged_queries_do_not_contaminate_next_lookup() -> None:
    first_queries = _build_staged_search_queries("Marine Leleu", fast_mode=False)
    second_queries = _build_staged_search_queries("Ada Lovelace", fast_mode=False)

    assert any("marine leleu" in query.casefold() for query in first_queries)
    assert all("marine" not in query.casefold() for query in second_queries)
    assert any("ada lovelace" in query.casefold() for query in second_queries)


def test_entity_ranking_penalizes_partial_name_substitution() -> None:
    search_query = _build_search_query("Marine Leleu")
    ranked = _rank_search_results(
        [
            {
                "title": "Marine Le Pen",
                "url": "https://example.com/marine-le-pen",
                "snippet": "French politician and party leader.",
            },
            {
                "title": "Marine Leleu",
                "url": "https://example.com/marine-leleu",
                "snippet": "French endurance athlete and content creator.",
            },
        ],
        query=search_query,
    )

    assert ranked[0]["title"] == "Marine Leleu"


@patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
@patch("unclaw.tools.web_tools._search_public_web")
def test_staged_broadening_waits_until_first_pass_is_weak(
    mock_search: MagicMock,
    mock_parse: MagicMock,
) -> None:
    mock_search.side_effect = ["<html>weak</html>", "<html>strong</html>"]
    mock_parse.side_effect = [
        [
            {
                "title": "Marine Le Pen",
                "url": "https://example.com/marine-le-pen",
                "snippet": "French politician.",
            },
        ],
        [
            {
                "title": "Marine Leleu",
                "url": "https://example.com/marine-leleu",
                "snippet": "French endurance athlete.",
            },
        ],
    ]

    _search_query, ranked_results = _run_bounded_staged_search(
        query="Marine Leleu",
        max_results=5,
        timeout_seconds=5.0,
        fast_mode=False,
    )

    searched_queries = [
        call.kwargs["query"]
        for call in mock_search.call_args_list
    ]
    assert searched_queries == ['"Marine Leleu"', "Marine Leleu"]
    assert ranked_results[0]["title"] == "Marine Leleu"


@patch("unclaw.tools.web_tools._parse_duckduckgo_html_results")
@patch("unclaw.tools.web_tools._search_public_web")
def test_staged_broadening_stops_after_strong_exact_match(
    mock_search: MagicMock,
    mock_parse: MagicMock,
) -> None:
    mock_search.return_value = "<html>strong</html>"
    mock_parse.return_value = [
        {
            "title": "Marine Leleu",
            "url": "https://example.com/marine-leleu",
            "snippet": "French endurance athlete and content creator.",
        },
    ]

    _run_bounded_staged_search(
        query="Marine Leleu",
        max_results=5,
        timeout_seconds=5.0,
        fast_mode=False,
    )

    assert mock_search.call_count == 1
    assert mock_search.call_args.kwargs["query"] == '"Marine Leleu"'


def test_source_hygiene_penalty_pushes_noisy_pages_down() -> None:
    search_query = _build_search_query("Marine Leleu")
    clean_page = SimpleNamespace(
        text=(
            "Marine Leleu is a French endurance athlete. "
            "She completes long-distance challenges and shares training content."
        ),
        links=[],
        resolved_url="https://example.com/profile",
        title="Marine Leleu",
    )
    noisy_page = SimpleNamespace(
        text=(
            "Marine Leleu biography age height husband net worth wiki\n"
            "Read more\nRead more\nRead more\nRead more"
        ),
        links=[SimpleNamespace(url="/more", text="Read more") for _ in range(8)],
        resolved_url="https://example.com/marine-leleu-biography",
        title="Marine Leleu Biography Age Height Husband Net Worth Wiki",
    )

    clean_scores = _score_fetched_page(
        page=clean_page,
        title=clean_page.title,
        query=search_query,
        evidence_candidates=[],
        existing_evidence=[],
    )
    noisy_scores = _score_fetched_page(
        page=noisy_page,
        title=noisy_page.title,
        query=search_query,
        evidence_candidates=[],
        existing_evidence=[],
    )

    assert noisy_scores.hygiene_penalty > clean_scores.hygiene_penalty
    assert noisy_scores.usefulness < clean_scores.usefulness
