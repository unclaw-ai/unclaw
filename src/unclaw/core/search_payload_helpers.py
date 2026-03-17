"""Shared helpers for normalized search payload fields and compact sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unclaw.constants import EMPTY_RESPONSE_REPLY, RUNTIME_ERROR_REPLY


def read_search_string_items(value: Any) -> tuple[str, ...]:
    """Return trimmed non-empty string items from a search payload field."""
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )


def read_search_display_sources(
    payload: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    """Read compact display sources from ``display_sources`` or ``results``."""
    if not isinstance(payload, Mapping):
        return ()

    raw_sources = payload.get("display_sources")
    if not isinstance(raw_sources, list):
        raw_sources = payload.get("results")
        if not isinstance(raw_sources, list):
            return ()

    sources: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        title = entry.get("title")
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        normalized_url = url.strip()
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        sources.append(
            (
                title.strip() if isinstance(title, str) else "",
                normalized_url,
            )
        )
    return tuple(sources)


def append_compact_search_sources(
    reply_text: str,
    *,
    sources: tuple[tuple[str, str], ...],
) -> str:
    """Append a compact sources section when the reply body can accept it."""
    if reply_text in {RUNTIME_ERROR_REPLY, EMPTY_RESPONSE_REPLY}:
        return reply_text
    if not sources:
        return reply_text

    lines = [reply_text.rstrip(), "", "Sources:"]
    for title, url in sources:
        if title:
            lines.append(f"- {title}: {url}")
        else:
            lines.append(f"- {url}")
    return "\n".join(lines)
