"""DuckDuckGo integration, query discipline, URL classification, and ranking."""

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
_NAME_PARTICLES = frozenset(
    {
        "al",
        "bin",
        "da",
        "de",
        "del",
        "der",
        "des",
        "di",
        "dos",
        "du",
        "el",
        "la",
        "le",
        "van",
        "von",
    }
)
_QUERY_ENTITY_PATTERNS = (
    re.compile(
        r"^\s*(?:who\s+is|who['’]s|tell\s+me\s+about|profile\s+of|bio(?:graphy)?\s+of|news\s+about|latest\s+news\s+about)\s+(.+?)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:qui\s+est|c['’]?est\s+qui|biographie\s+de|bio\s+de|profil\s+de|actualit(?:e|é)s\s+sur|nouvelles\s+sur)\s+(.+?)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:quien\s+es|biograf(?:i|í)a\s+de|perfil\s+de|noticias\s+sobre)\s+(.+?)\s*$",
        re.IGNORECASE,
    ),
)
_ENTITY_CONTEXT_MODIFIERS = frozenset(
    {
        "actualite",
        "actualites",
        "biografia",
        "biographie",
        "bio",
        "career",
        "dernieres",
        "history",
        "latest",
        "news",
        "nouvelles",
        "official",
        "profil",
        "profile",
        "recent",
        "recente",
        "recientes",
        "wiki",
        "wikipedia",
    }
)
_SEO_PROFILE_CUES = frozenset(
    {
        "age",
        "biography",
        "dating",
        "family",
        "height",
        "husband",
        "married",
        "net worth",
        "salary",
        "wife",
        "wiki",
    }
)
# Encyclopedic/reference domains with strong biographical authority.
_REFERENCE_DOMAINS = frozenset(
    {
        "wikipedia.org",
        "wikidata.org",
        "britannica.com",
        "larousse.fr",
        "universalis.fr",
        "enciclopedia.cat",
    }
)
# Social network domains where a short path means a profile landing page
# with almost no extractable biographical text.
_SOCIAL_PROFILE_SHELL_DOMAINS = frozenset(
    {
        "instagram.com",
        "tiktok.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "youtube.com",
        "threads.net",
        "snapchat.com",
        "pinterest.com",
    }
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
    entity_surface: str = ""
    normalized_entity: str = ""
    entity_tokens: tuple[str, ...] = ()
    context_tokens: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _QueryDiscipline:
    """Conservative query analysis used to preserve user-supplied entities."""

    raw_query: str
    conservative_query: str
    entity_surface: str
    normalized_entity: str
    entity_tokens: tuple[str, ...]
    context_tokens: tuple[str, ...]


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


@dataclass(frozen=True, slots=True)
class _StagedSearchOutcome:
    """Result of bounded staged search broadening."""

    search_query: _SearchQuery
    ranked_results: list[dict[str, str]]
    executed_queries: tuple[str, ...]


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


def _normalize_query_surface(query: str) -> str:
    return " ".join(query.split()).strip()


def _strip_wrapping_punctuation(text: str) -> str:
    return text.strip().strip("()[]{}<>'\"“”‘’.,;:!?")


def _extract_quoted_entity(query: str) -> str:
    match = re.search(r"['\"“”‘’]([^'\"“”‘’]+)['\"“”‘’]", query)
    if match is None:
        return ""
    return _strip_wrapping_punctuation(match.group(1))


def _extract_prefixed_entity(query: str) -> str:
    for pattern in _QUERY_ENTITY_PATTERNS:
        match = pattern.match(query)
        if match is None:
            continue
        return _strip_wrapping_punctuation(match.group(1))
    return ""


def _query_token_surfaces(query: str) -> tuple[str, ...]:
    return tuple(
        _strip_wrapping_punctuation(token)
        for token in re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ'’.-]+", query)
        if _strip_wrapping_punctuation(token)
    )


def _token_looks_entity_like(token: str, *, position: int, span_started: bool) -> bool:
    folded = _fold_for_match(token)
    if not folded:
        return False
    if span_started and folded in _NAME_PARTICLES:
        return True
    if folded in _ENTITY_CONTEXT_MODIFIERS:
        return False
    if token[0].isupper():
        return True
    return position == 0 and (token.isupper() or any(character.isdigit() for character in token))


def _extract_leading_entity_span(query: str) -> str:
    tokens = _query_token_surfaces(query)
    if not tokens:
        return ""

    span: list[str] = []
    for index, token in enumerate(tokens):
        folded = _fold_for_match(token)
        if folded in _ENTITY_CONTEXT_MODIFIERS:
            break
        if _token_looks_entity_like(token, position=index, span_started=bool(span)):
            span.append(token)
            continue
        if span:
            break

    if not span or len(span) > 4:
        return ""

    remaining = tokens[len(span) :]
    if remaining and not all(
        _fold_for_match(token) in _ENTITY_CONTEXT_MODIFIERS
        or _fold_for_match(token) in _QUERY_STOPWORDS
        for token in remaining
    ):
        return ""
    return " ".join(span)


def _extract_entity_surface(query: str) -> str:
    normalized_query = _normalize_query_surface(query)
    if not normalized_query:
        return ""

    for candidate in (
        _extract_quoted_entity(normalized_query),
        _extract_prefixed_entity(normalized_query),
        _extract_leading_entity_span(normalized_query),
    ):
        if candidate:
            return candidate

    token_surfaces = _query_token_surfaces(normalized_query)
    if 1 <= len(token_surfaces) <= 4:
        return " ".join(token_surfaces)
    return ""


def _build_context_tokens(query: str, entity_surface: str) -> tuple[str, ...]:
    full_tokens = list(_text_tokens(query))
    entity_tokens = list(_text_tokens(entity_surface))
    if entity_tokens:
        for index in range(len(full_tokens) - len(entity_tokens) + 1):
            if full_tokens[index : index + len(entity_tokens)] == entity_tokens:
                del full_tokens[index : index + len(entity_tokens)]
                break

    return tuple(
        token
        for token in full_tokens
        if len(token) >= 3
        and token not in _QUERY_STOPWORDS
        and token not in entity_tokens
    )[:4]


def _analyze_query_discipline(query: str) -> _QueryDiscipline:
    normalized_query = _normalize_query_surface(query)
    entity_surface = _extract_entity_surface(normalized_query)
    normalized_entity = _fold_for_match(entity_surface)
    context_tokens = _build_context_tokens(normalized_query, entity_surface)
    conservative_query = entity_surface or normalized_query
    return _QueryDiscipline(
        raw_query=normalized_query,
        conservative_query=conservative_query,
        entity_surface=entity_surface,
        normalized_entity=normalized_entity,
        entity_tokens=_text_tokens(entity_surface),
        context_tokens=context_tokens,
    )


def _build_staged_search_queries(
    query: str,
    *,
    fast_mode: bool,
) -> tuple[str, ...]:
    discipline = _analyze_query_discipline(query)
    max_passes = 2 if fast_mode else 3
    planned_queries: list[str] = []
    seen_queries: set[str] = set()

    def _add(candidate: str) -> None:
        normalized_candidate = _normalize_query_surface(candidate)
        if not normalized_candidate:
            return
        key = normalized_candidate.casefold()
        if key in seen_queries:
            return
        planned_queries.append(normalized_candidate)
        seen_queries.add(key)

    if discipline.entity_surface:
        quoted_entity = f"\"{discipline.entity_surface}\""
        _add(quoted_entity)
        if discipline.context_tokens:
            _add(" ".join((quoted_entity, *discipline.context_tokens[:2])))
            _add(" ".join((discipline.entity_surface, *discipline.context_tokens[:4])))
        else:
            _add(discipline.entity_surface)
    else:
        _add(discipline.conservative_query)
        keyword_only = " ".join(
            token
            for token in _text_tokens(discipline.raw_query)
            if token not in _QUERY_STOPWORDS
        )
        _add(keyword_only)

    return tuple(planned_queries[:max_passes])


def _read_search_pass_number(result: Mapping[str, str]) -> int:
    raw_value = result.get("_pass_number", "1")
    try:
        return max(int(raw_value), 1)
    except ValueError:
        return 1


def _result_folded_metadata(result: Mapping[str, str]) -> str:
    return _fold_for_match(
        f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}"
    )


def _result_entity_token_hits(
    result: Mapping[str, str],
    *,
    query: _SearchQuery,
) -> int:
    if not query.entity_tokens:
        return 0
    metadata_tokens = set(_result_folded_metadata(result).split())
    return sum(1 for token in query.entity_tokens if token in metadata_tokens)


def _result_exact_entity_match(
    result: Mapping[str, str],
    *,
    query: _SearchQuery,
) -> bool:
    return bool(
        query.normalized_entity
        and query.normalized_entity in _result_folded_metadata(result)
    )


def _text_looks_malformed(text: str) -> bool:
    if not text:
        return False
    if "\ufffd" in text:
        return True
    symbol_count = sum(
        1
        for character in text
        if not (character.isalnum() or character.isspace() or character in ".,;:!?%()[]{}'\"/-_")
    )
    return symbol_count / max(len(text), 1) > 0.18


def _result_hygiene_penalty(
    *,
    title: str,
    snippet: str,
    url: str,
) -> float:
    folded_metadata = _fold_for_match(f"{title} {snippet} {url}")
    cue_hits = sum(1 for cue in _SEO_PROFILE_CUES if cue in folded_metadata)
    penalty = 0.0
    if cue_hits >= 4:
        penalty += 4.0
    elif cue_hits >= 3:
        penalty += 2.5
    elif cue_hits >= 2 and _passage_has_noise_signals(snippet):
        penalty += 1.0
    if _text_looks_malformed(title) or _text_looks_malformed(snippet):
        penalty += 1.5
    return penalty


def _search_results_look_weak(
    results: list[dict[str, str]],
    *,
    query: _SearchQuery,
) -> bool:
    if not results:
        return True

    top_results = results[:3]
    scored_results = [_score_search_result(result, query=query) for result in top_results]
    if query.entity_tokens:
        exact_matches = sum(
            1 for result in top_results if _result_exact_entity_match(result, query=query)
        )
        aligned_matches = sum(
            1
            for result in top_results
            if _result_entity_token_hits(result, query=query) >= len(query.entity_tokens)
        )
        return not (
            exact_matches >= 1
            or aligned_matches >= 1 and max(scored_results, default=0.0) >= 6.0
            or aligned_matches >= 2
        )

    return max(scored_results, default=0.0) < 5.0


def _run_staged_public_search(
    *,
    query: str,
    max_results: int,
    timeout_seconds: float,
    fast_mode: bool,
) -> _StagedSearchOutcome:
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
    return _StagedSearchOutcome(
        search_query=search_query,
        ranked_results=ranked_results[:max_results],
        executed_queries=tuple(executed_queries),
    )


def _build_search_query(query: str) -> _SearchQuery:
    discipline = _analyze_query_discipline(query)
    keyword_tokens = tuple(
        token
        for token in _text_tokens(query)
        if len(token) >= 3 and token not in _QUERY_STOPWORDS
    )
    if not keyword_tokens:
        keyword_tokens = tuple(token for token in _text_tokens(query) if len(token) >= 2)
    return _SearchQuery(
        raw_query=discipline.raw_query,
        normalized_query=_fold_for_match(discipline.raw_query),
        keyword_tokens=keyword_tokens,
        entity_surface=discipline.entity_surface,
        normalized_entity=discipline.normalized_entity,
        entity_tokens=discipline.entity_tokens,
        context_tokens=discipline.context_tokens,
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
    score += max(1.5 - (_read_search_pass_number(result) - 1) * 0.6, 0.0)

    if query.entity_tokens:
        entity_hits = _result_entity_token_hits(result, query=query)
        if entity_hits >= len(query.entity_tokens):
            score += 4.5
            if _result_exact_entity_match(result, query=query):
                score += 2.0
        elif 0 < entity_hits < len(query.entity_tokens):
            score -= 4.5
        else:
            score -= 1.0

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
    score -= _result_hygiene_penalty(title=title, snippet=snippet, url=url)

    if _url_looks_reference_like(url):
        score += 3.0
    if _url_looks_social_profile_shell(url):
        score -= 3.5

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


def _url_looks_reference_like(url: str) -> bool:
    """True for encyclopedic/reference sources with strong biographical authority."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if any(hostname == domain or hostname.endswith("." + domain) for domain in _REFERENCE_DOMAINS):
        return True
    # /wiki/ path pattern covers most Wikipedia language variants.
    if "/wiki/" in parsed.path.lower():
        return True
    return False


def _url_looks_social_profile_shell(url: str) -> bool:
    """True for social network profile landing pages with little extractable biography text."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    # Strip leading www.
    if hostname.startswith("www."):
        hostname = hostname[4:]
    if hostname not in _SOCIAL_PROFILE_SHELL_DOMAINS:
        return False
    # Profile shell: root or single user-slug segment (e.g. /username or /username/).
    path_segments = _url_path_segments(url)
    return len(path_segments) <= 1


def _source_title_looks_weak(title: str) -> bool:
    folded_title = _fold_for_match(title)
    return any(
        token in folded_title
        for token in ("podcast", "episode", "newsletter", "substack", "roundup")
    )
