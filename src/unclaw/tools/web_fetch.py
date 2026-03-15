"""HTTP fetching, content decoding, and text document extraction."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.request import Request

from unclaw.tools.web_html import _HTMLLink, _extract_page_content
from unclaw.tools.web_safety import (
    _ensure_fetch_target_allowed,
    _open_request,
)

_DEFAULT_MAX_FETCH_CHARS = 8_000
_MAX_FETCH_BYTES = 1_000_000
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_SEARCH_FETCH_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class _RawFetchedDocument:
    """Decoded network response body and basic metadata."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    decoded_text: str


@dataclass(slots=True)
class _FetchedTextDocument:
    """Compact extracted text payload for one fetched public URL."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    text_excerpt: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class _FetchedSearchPage:
    """Fetched page content enriched with extracted links for retrieval."""

    requested_url: str
    resolved_url: str
    status_code: int | None
    content_type: str
    title: str
    text: str
    truncated: bool
    links: tuple[_HTMLLink, ...]


def _decode_content(raw_content: bytes, charset: str) -> str:
    try:
        return raw_content.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_content.decode("utf-8", errors="replace")


def _fetch_raw_document(
    url: str,
    *,
    timeout_seconds: float,
    allow_private_networks: bool,
    accept_header: str,
) -> _RawFetchedDocument:
    _ensure_fetch_target_allowed(
        url,
        allow_private_networks=allow_private_networks,
    )

    request = Request(
        url,
        headers={
            "User-Agent": "unclaw/0.1 (+https://local-first.invalid)",
            "Accept": accept_header,
        },
    )

    with _open_request(
        request,
        timeout_seconds=timeout_seconds,
        allow_private_networks=allow_private_networks,
    ) as response:
        content_type = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        status_code = getattr(response, "status", None)
        resolved_url = response.geturl()
        raw_content = response.read(_MAX_FETCH_BYTES + 1)

    if len(raw_content) > _MAX_FETCH_BYTES:
        raw_content = raw_content[:_MAX_FETCH_BYTES]

    return _RawFetchedDocument(
        requested_url=url,
        resolved_url=resolved_url,
        status_code=status_code,
        content_type=content_type,
        decoded_text=_decode_content(raw_content, charset),
    )


def _fetch_text_document(
    url: str,
    *,
    max_chars: int,
    timeout_seconds: float,
    allow_private_networks: bool,
    accept_header: str,
) -> _FetchedTextDocument:
    raw_document = _fetch_raw_document(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=allow_private_networks,
        accept_header=accept_header,
    )
    extracted_content = _extract_page_content(
        raw_document.decoded_text,
        raw_document.content_type,
    )
    if extracted_content is None:
        raise ValueError(
            f"Unsupported content type for text extraction: {raw_document.content_type}"
        )

    _title, extracted_text, _links = extracted_content
    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    text_excerpt = extracted_text[:max_chars].rstrip() if truncated else extracted_text
    return _FetchedTextDocument(
        requested_url=url,
        resolved_url=raw_document.resolved_url,
        status_code=raw_document.status_code,
        content_type=raw_document.content_type,
        text_excerpt=text_excerpt,
        truncated=truncated,
    )


def _fetch_search_page(
    url: str,
    *,
    max_chars: int,
    timeout_seconds: float,
) -> _FetchedSearchPage:
    raw_document = _fetch_raw_document(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=False,
        accept_header="text/plain, text/html, application/json;q=0.9, */*;q=0.1",
    )
    extracted_content = _extract_page_content(
        raw_document.decoded_text,
        raw_document.content_type,
    )
    if extracted_content is None:
        raise ValueError(
            f"Unsupported content type for text extraction: {raw_document.content_type}"
        )

    page_title, extracted_text, links = extracted_content
    if not extracted_text:
        extracted_text = "[empty response body]"

    truncated = len(extracted_text) > max_chars
    page_text = extracted_text[:max_chars].rstrip() if truncated else extracted_text
    return _FetchedSearchPage(
        requested_url=url,
        resolved_url=raw_document.resolved_url,
        status_code=raw_document.status_code,
        content_type=raw_document.content_type,
        title=page_title,
        text=page_text,
        truncated=truncated,
        links=links,
    )


def _format_text_excerpt(text: str, *, truncated: bool) -> str:
    if not truncated:
        return text
    return f"{text}\n\n[truncated]"
