"""DuckDuckGo integration, result parsing, URL classification, and ranking."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse, urlunparse

from unclaw.tools.web_fetch import _fetch_raw_document
from unclaw.tools.web_safety import _is_supported_url
from unclaw.tools.web_text import (
    _QUERY_STOPWORDS,
    _fold_for_match,
    _keyword_overlap_score,
    _normalize_text,
    _passage_has_noise_signals,
    _text_tokens,
)

_DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
_SEARCH_PROVIDER_NAME = "DuckDuckGo HTML"
_DEFAULT_MAX_SEARCH_RESULTS = 20
_MAX_SEARCH_RESULTS = 20
_LOW_VALUE_RESULT_TITLES = {"accueil", "home", "homepage", "index"}
_ARTICLE_PATH_CUES = frozenset(
    {
        "analysis",
        "article",
        "articles",
        "blog",
        "blogs",
        "entry",
        "feature",
        "features",
        "post",
        "posts",
        "recap",
        "report",
        "reports",
        "story",
        "stories",
        "update",
        "updates",
    }
)
_LIVE_STREAMING_PATH_CUES = frozenset(
    {
        "direct",
        "directs",
        "emission",
        "emissions",
        "en-direct",
        "live",
        "player",
        "programme",
        "programmes",
        "regarder",
        "replay",
        "stream",
        "streaming",
        "tv",
        "watch",
    }
)
_HUB_PATH_CUES = frozenset(
    {
        "archive",
        "archives",
        "category",
        "categories",
        "index",
        "latest",
        "listing",
        "live",
        "page",
        "section",
        "tag",
        "tags",
        "topics",
        "updates",
    }
)
_LOW_VALUE_PATH_CUES = frozenset(
    {
        "about",
        "account",
        "contact",
        "donate",
        "help",
        "join",
        "legal",
        "login",
        "logout",
        "privacy",
        "register",
        "settings",
        "share",
        "signin",
        "signup",
        "subscribe",
        "support",
        "terms",
    }
)
_LOW_VALUE_EXTENSIONS = (
    ".css",
    ".ico",
    ".jpg",
    ".jpeg",
    ".js",
    ".json",
    ".pdf",
    ".png",
    ".rss",
    ".svg",
    ".xml",
    ".zip",
)


@dataclass(slots=True)
class _SearchResultBuilder:
    """Collect one parsed DuckDuckGo HTML result block."""

    url: str
    title_parts: list[str] = field(default_factory=list)
    snippet_parts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _SearchQuery:
    """Normalized query tokens for generic retrieval scoring."""

    raw_query: str
    normalized_query: str
    keyword_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SearchCandidate:
    """One URL candidate waiting to be fetched."""

    url: str
    title: str
    snippet: str
    depth: int
    priority: float
    order: int
    parent_url: str | None = None
    anchor_text: str = ""


class _DuckDuckGoHTMLSearchParser(HTMLParser):
    """Parse compact search results from DuckDuckGo's HTML endpoint."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._current_result: _SearchResultBuilder | None = None
        self._capture_target: tuple[str, str] | None = None
        self._results: list[dict[str, str]] = []

    @property
    def results(self) -> list[dict[str, str]]:
        return self._results

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag == "a" and "result__a" in classes:
            self._finalize_current_result()
            self._current_result = _SearchResultBuilder(
                url=_normalize_search_result_url(attributes.get("href"))
            )
            self._capture_target = ("title", tag)
            return

        if self._current_result is not None and "result__snippet" in classes:
            self._capture_target = ("snippet", tag)

    def handle_endtag(self, tag: str) -> None:
        if self._capture_target is None:
            return
        _kind, captured_tag = self._capture_target
        if tag == captured_tag:
            self._capture_target = None

    def handle_data(self, data: str) -> None:
        if self._current_result is None or self._capture_target is None:
            return

        text = data.strip()
        if not text:
            return

        kind, _tag = self._capture_target
        if kind == "title":
            self._current_result.title_parts.append(text)
            return

        self._current_result.snippet_parts.append(text)

    def close(self) -> None:
        super().close()
        self._finalize_current_result()

    def _finalize_current_result(self) -> None:
        if self._current_result is None:
            return

        title = _normalize_text(" ".join(self._current_result.title_parts))
        url = self._current_result.url.strip()
        snippet = _normalize_text(" ".join(self._current_result.snippet_parts))

        if title and url:
            self._results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        self._capture_target = None
        self._current_result = None


def _search_public_web(*, query: str, timeout_seconds: float) -> str:
    request_url = f"{_DUCKDUCKGO_HTML_SEARCH_URL}?{urlencode({'q': query})}"
    raw_doc = _fetch_raw_document(
        request_url,
        timeout_seconds=timeout_seconds,
        allow_private_networks=False,
        accept_header="text/html, application/xhtml+xml;q=0.9, */*;q=0.1",
    )

    if raw_doc.content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(
            f"Search provider returned unsupported content type: {raw_doc.content_type}"
        )

    return raw_doc.decoded_text


def _parse_duckduckgo_html_results(
    response_text: str,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    parser = _DuckDuckGoHTMLSearchParser()
    parser.feed(response_text)
    parser.close()
    return parser.results[:max_results]


def _build_search_query(query: str) -> _SearchQuery:
    keyword_tokens = tuple(
        token
        for token in _text_tokens(query)
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    )
    if not keyword_tokens:
        keyword_tokens = tuple(token for token in _text_tokens(query) if len(token) >= 2)
    return _SearchQuery(
        raw_query=query,
        normalized_query=_fold_for_match(query),
        keyword_tokens=keyword_tokens,
    )


def _deduplicate_search_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduplicated: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for result in results:
        url = result["url"].strip()
        canonical_url = _canonicalize_url(url)
        if not url or not canonical_url or canonical_url in seen_urls:
            continue
        deduplicated.append(dict(result))
        seen_urls.add(canonical_url)

    return deduplicated


def _rank_search_results(
    results: list[dict[str, str]],
    *,
    query: _SearchQuery,
) -> list[dict[str, str]]:
    scored_results: list[tuple[float, int, dict[str, str]]] = []

    for index, result in enumerate(results):
        score = _score_search_result(result, query=query)
        scored_results.append((score, index, dict(result)))

    scored_results.sort(key=lambda item: (-item[0], item[1]))
    return [result for _score, _index, result in scored_results]


def _score_search_result(
    result: Mapping[str, str],
    *,
    query: _SearchQuery,
) -> float:
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    url = result.get("url", "")
    parsed_url = urlparse(url)
    folded_metadata = _fold_for_match(f"{title} {snippet} {url}")
    path_segments = _url_path_segments(url)
    score = 0.0

    score += _keyword_overlap_score(folded_metadata, query.keyword_tokens) * 3.0
    score += min(len(snippet.split()) / 12.0, 2.0)
    score += min(len(path_segments), 3)

    if _url_looks_article_like(url):
        score += 4.0
    if _url_looks_archive_like(url):
        score += 1.0
    if _url_looks_homepage_like(url):
        score -= 6.0
    if _url_looks_live_or_streaming(url):
        score -= 5.0
    if _looks_generic_result_title(title=title, hostname=parsed_url.hostname or ""):
        score -= 3.0
    if _passage_has_noise_signals(snippet):
        score -= 2.0

    return score


# --- URL classification and utilities ---


def _canonicalize_url(url: str) -> str:
    if not _is_supported_url(url):
        return ""
    parsed = urlparse(url)
    normalized_path = parsed.path or "/"
    if normalized_path != "/" and normalized_path.endswith("/"):
        normalized_path = normalized_path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            normalized_path,
            "",
            parsed.query,
            "",
        )
    )


def _normalize_search_result_url(raw_url: str | None) -> str:
    if raw_url is None or not raw_url.strip():
        return ""

    resolved_url = urljoin(_DUCKDUCKGO_HTML_SEARCH_URL, raw_url.strip())
    parsed = urlparse(resolved_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.rstrip("/") == "/l":
        redirect_targets = parse_qs(parsed.query).get("uddg", ())
        if redirect_targets:
            target_url = unquote(redirect_targets[0]).strip()
            if target_url:
                return target_url

    return resolved_url


def _url_path_segments(url: str) -> tuple[str, ...]:
    parsed = urlparse(url)
    return tuple(segment for segment in parsed.path.split("/") if segment)


def _url_looks_homepage_like(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path in {"", "/"}:
        return True
    return path.casefold() in {"/accueil", "/home", "/index.html"}


def _url_looks_article_like(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path
    path_segments = _url_path_segments(url)
    if re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", path):
        return True
    if re.search(r"/20\d{2}-\d{2}-\d{2}", path):
        return True
    if any(segment.casefold() in _ARTICLE_PATH_CUES for segment in path_segments):
        return True
    if path_segments and path_segments[-1].count("-") >= 3:
        return True
    return False


def _url_looks_archive_like(url: str) -> bool:
    path_segments = {segment.casefold() for segment in _url_path_segments(url)}
    return bool(path_segments & _HUB_PATH_CUES)


def _url_looks_live_or_streaming(url: str) -> bool:
    path_segments = {segment.casefold() for segment in _url_path_segments(url)}
    return bool(path_segments & _LIVE_STREAMING_PATH_CUES)


def _url_looks_low_value(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.fragment:
        return True
    path = parsed.path.casefold()
    if any(path.endswith(extension) for extension in _LOW_VALUE_EXTENSIONS):
        return True
    path_segments = {segment.casefold() for segment in _url_path_segments(url)}
    if path_segments & _LOW_VALUE_PATH_CUES:
        return True
    return False


def _looks_generic_result_title(*, title: str, hostname: str) -> bool:
    folded_title = _fold_for_match(title)
    if folded_title in _LOW_VALUE_RESULT_TITLES:
        return True

    hostname_tokens = [
        token
        for token in re.split(r"[.\-]+", hostname.casefold())
        if token and token not in {"com", "fr", "net", "org", "www"}
    ]
    if hostname_tokens and folded_title == " ".join(hostname_tokens):
        return True
    return False


def _registered_domain(hostname: str) -> str:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname.startswith("www."):
        normalized_hostname = normalized_hostname[4:]
    parts = normalized_hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return normalized_hostname


def _is_internal_link(*, parent_url: str, child_url: str) -> bool:
    parent_hostname = urlparse(parent_url).hostname
    child_hostname = urlparse(child_url).hostname
    if parent_hostname is None or child_hostname is None:
        return False
    return _registered_domain(parent_hostname) == _registered_domain(child_hostname)


def _source_title_looks_weak(title: str) -> bool:
    folded_title = _fold_for_match(title)
    return any(
        token in folded_title
        for token in ("podcast", "episode", "newsletter", "substack", "roundup")
    )
