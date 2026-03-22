"""Typed payload contracts for the local_text_search skill."""

from __future__ import annotations

from typing import TypedDict


class LocalTextMatchPayload(TypedDict):
    """One match entry returned by search_local_text."""

    file_path: str
    line_number: int
    line_text: str
    snippet: str


class LocalTextSearchPayload(TypedDict):
    """Structured result from search_local_text."""

    query: str
    root: str
    extensions_filter: list[str] | None
    max_results: int
    total_matches_found: int
    truncated: bool
    matches: list[LocalTextMatchPayload]


__all__ = [
    "LocalTextMatchPayload",
    "LocalTextSearchPayload",
]
