"""Grounding helpers for search-backed answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
import re
import unicodedata
from typing import Any

from unclaw.core.search_payload_helpers import (
    read_search_display_sources,
    read_search_string_items,
)
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.tools.contracts import SearchWebPayload

_MAX_COMPOSED_FACTS = 3
_SEARCH_TOOL_PREFIX = "Tool: search_web\n"
_STALE_AS_OF_PATTERN = re.compile(
    r"\bas of\s+(?:"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{4}"
    r"|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},\s+\d{4}"
    r")\b",
    flags=re.IGNORECASE,
)
_AGE_QUERY_PATTERNS = (
    re.compile(r"\bhow old\b"),
    re.compile(r"\bage\b"),
    re.compile(r"\bdate of birth\b"),
    re.compile(r"\bbirth date\b"),
    re.compile(r"\bborn\b"),
    re.compile(r"\bquel age\b"),
    re.compile(r"\bage de\b"),
    re.compile(r"\bdate de naissance\b"),
)
_PERSON_QUERY_PATTERNS = (
    re.compile(r"\bwho is\b"),
    re.compile(r"\bwho[' ]?s\b"),
    re.compile(r"\btell me(?: everything)? you know about\b"),
    re.compile(r"\btell me about\b"),
    re.compile(r"\bwhat has\b"),
    re.compile(r"\bwhat did\b"),
    re.compile(r"\bprofile\b"),
    re.compile(r"\bqui est\b"),
    re.compile(r"\bparle moi de\b"),
    re.compile(r"\bfais moi un resume\b"),
    re.compile(r"\bresume sur\b"),
)
_LOW_VALUE_DETAIL_PATTERNS = (
    re.compile(r"\bjourneys?\b"),
    re.compile(r"\binspiring\b"),
    re.compile(r"\bpassion(?:ate)?\b"),
    re.compile(r"\bit seems\b"),
    re.compile(r"\bseems to\b"),
    re.compile(r"\bappears to\b"),
    re.compile(r"\bprobably\b"),
    re.compile(r"\boften\b"),
    re.compile(r"\bpodcasts?\b"),
    re.compile(r"\bepisodes?\b"),
    re.compile(r"\bnewsletters?\b"),
    re.compile(r"\bblogs?\b"),
    re.compile(r"\bsubstack\b"),
)
_HANDLE_HINT_PATTERN = re.compile(
    r"(?<!\w)@[a-z0-9_.]{2,32}\b",
    flags=re.IGNORECASE,
)
_HANDLE_TERM_PATTERN = re.compile(
    r"\b(?:username|handle|instagram|twitter|x\.com|tiktok|youtube|linkedin|threads)\b",
    flags=re.IGNORECASE,
)
_AGE_CLAIM_PATTERN = re.compile(
    r"\b\d{1,3}\s+years?\s+old\b|\baged\s+\d{1,3}\b",
    flags=re.IGNORECASE,
)
_SOURCE_BULLET_PATTERN = re.compile(r"^- (?P<title>.*?): (?P<url>https?://\S+)$")
_FINDING_BULLET_PATTERN = re.compile(
    r"^- \[(?P<label>[a-z]+); (?P<support>\d+) source(?:s)?\] (?P<text>.+)$",
    flags=re.IGNORECASE,
)
_FOLLOW_UP_QUERY_PATTERN = re.compile(
    r"\b(?:that|this|those|these|it|them|shorter|briefly|recap|summari[sz]e|"
    r"clarify|expand|verify|confirm|source|sources|what about|and what)\b",
    flags=re.IGNORECASE,
)
_MONTH_DATE_PATTERNS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


@dataclass(frozen=True, slots=True)
class SearchGroundingFinding:
    """One grounded fact extracted from retrieved search context."""

    text: str
    support_count: int
    score: float
    source_titles: tuple[str, ...]
    source_urls: tuple[str, ...]
    confidence: str


@dataclass(frozen=True, slots=True)
class SearchGroundingContext:
    """Normalized search grounding data used for prompt shaping and rewrites."""

    query: str
    current_date: date
    supported_findings: tuple[SearchGroundingFinding, ...]
    uncertain_findings: tuple[SearchGroundingFinding, ...]
    display_sources: tuple[tuple[str, str], ...]
    birth_date: date | None = None


@dataclass(frozen=True, slots=True)
class _ParsedSearchPayload:
    query: str
    raw_findings: tuple[SearchGroundingFinding, ...]
    display_sources: tuple[tuple[str, str], ...]
    source_quality_by_url: Mapping[str, float]
    birth_date: date | None


@dataclass(slots=True)
class _SearchToolHistoryState:
    current_date: date
    query: str = ""
    supported_findings: list[SearchGroundingFinding] = field(default_factory=list)
    uncertain_findings: list[SearchGroundingFinding] = field(default_factory=list)
    display_sources: list[tuple[str, str]] = field(default_factory=list)
    birth_date: date | None = None
    section: str = ""


def build_search_answer_contract(*, current_date: date | None = None) -> str:
    """Build the system note for grounded search-backed answering."""
    resolved_date = current_date or date.today()
    return "\n".join(
        (
            "Search-backed answer contract:",
            f"- Current date: {resolved_date.isoformat()}",
            (
                "- If a `search_web` tool message is present, answer only from the "
                "facts it supports."
            ),
            "- Prefer directly supported facts and omit decorative filler.",
            (
                "- If a detail is weakly supported or unconfirmed, leave it out or "
                "label it clearly as unconfirmed."
            ),
            (
                "- Do not invent relative date framing such as `as of May 2024` or "
                "similar stale anchors."
            ),
            (
                "- Give age only when you can compute it from a retrieved birth date "
                "and the current date above. Otherwise give the birth date only, or "
                "say the age is not confirmed."
            ),
            (
                "- For person summaries, prioritize identity, role or domain, best-"
                "supported achievements, and notable public activity."
            ),
            (
                "- Skip low-value blog, podcast, or narrative filler unless it adds "
                "a concrete fact supported by the retrieved sources."
            ),
            "- Keep the answer natural prose. Do not dump raw tool output.",
        )
    )


def has_search_grounding_context(history: Sequence[ChatMessage]) -> bool:
    """Return whether the recent history already contains search-backed grounding."""
    return any(_is_search_tool_message(message) for message in history)


def build_search_grounding_context(
    payload: SearchWebPayload | Mapping[str, Any] | None,
    *,
    query: str = "",
    current_date: date | None = None,
) -> SearchGroundingContext | None:
    """Normalize a search tool payload into grounded findings and sources."""
    if not isinstance(payload, Mapping):
        return None

    parsed_payload = _parse_search_payload(payload, fallback_query=query)
    if parsed_payload is None:
        return None

    normalized_findings = _classify_findings(
        parsed_payload.raw_findings,
        source_quality_by_url=parsed_payload.source_quality_by_url,
        birth_date=parsed_payload.birth_date,
    )
    supported_findings, uncertain_findings = _partition_findings_by_confidence(
        normalized_findings
    )

    return SearchGroundingContext(
        query=parsed_payload.query,
        current_date=current_date or date.today(),
        supported_findings=supported_findings,
        uncertain_findings=uncertain_findings,
        display_sources=parsed_payload.display_sources,
        birth_date=parsed_payload.birth_date,
    )


def build_search_tool_history_summary(
    *,
    payload: SearchWebPayload | Mapping[str, Any] | None,
    query: str = "",
    current_date: date | None = None,
) -> tuple[str, ...]:
    """Build compact, parseable search grounding lines for tool history."""
    grounding = build_search_grounding_context(
        payload,
        query=query,
        current_date=current_date,
    )
    if grounding is None:
        return ()

    lines = _build_tool_history_header_lines(grounding)
    lines.extend(_build_supported_history_lines(grounding))
    lines.extend(_build_uncertain_history_lines(grounding))
    lines.extend(_build_birth_date_history_lines(grounding))
    lines.extend(_build_source_history_lines(grounding))
    return tuple(lines)


def shape_search_backed_reply(
    reply_text: str,
    *,
    payload: SearchWebPayload | Mapping[str, Any] | None,
    query: str,
    current_date: date | None = None,
) -> str:
    """Rewrite risky search-backed replies into grounded, compact prose."""
    grounding = build_search_grounding_context(
        payload,
        query=query,
        current_date=current_date,
    )
    return shape_reply_with_grounding(reply_text, grounding=grounding, query=query)


def shape_reply_with_grounding(
    reply_text: str,
    *,
    grounding: SearchGroundingContext | None,
    query: str,
) -> str:
    """Rewrite risky search-backed replies using a pre-parsed grounding context."""
    stripped_reply = reply_text.strip()
    if grounding is None or not stripped_reply:
        return stripped_reply

    if not _reply_needs_rewrite(stripped_reply, grounding=grounding, query=query):
        return _sanitize_reply(stripped_reply)

    return _compose_grounded_answer(query=query, grounding=grounding)


def parse_search_tool_history(content: str) -> SearchGroundingContext | None:
    """Parse search grounding details back out of stored tool history text."""
    if not content.startswith(_SEARCH_TOOL_PREFIX):
        return None

    state = _SearchToolHistoryState(current_date=date.today())
    for line in content.splitlines():
        _parse_search_tool_history_line(line.strip(), state)

    return SearchGroundingContext(
        query=state.query,
        current_date=state.current_date,
        supported_findings=tuple(state.supported_findings),
        uncertain_findings=tuple(state.uncertain_findings),
        display_sources=tuple(state.display_sources),
        birth_date=state.birth_date,
    )


def should_apply_search_grounding(
    *,
    query: str,
    grounding: SearchGroundingContext | None,
) -> bool:
    """Return whether the current user turn likely refers to recent search context."""
    if grounding is None:
        return False

    normalized_query = _fold_for_match(query)
    if not normalized_query:
        return False

    if grounding.query and normalized_query == _fold_for_match(grounding.query):
        return True

    fact_tokens = {
        token
        for finding in (*grounding.supported_findings, *grounding.uncertain_findings)
        for token in _content_tokens(finding.text)
    }
    query_tokens = set(_content_tokens(query))
    if fact_tokens and query_tokens & fact_tokens:
        return True

    return _FOLLOW_UP_QUERY_PATTERN.search(query) is not None


def _read_query(payload: Mapping[str, Any], *, fallback: str) -> str:
    value = payload.get("query")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback.strip()


def _parse_search_payload(
    payload: Mapping[str, Any],
    *,
    fallback_query: str,
) -> _ParsedSearchPayload | None:
    resolved_query = _read_query(payload, fallback=fallback_query)
    raw_findings = _read_findings(payload)
    display_sources = read_search_display_sources(payload)
    if not raw_findings and not resolved_query and not display_sources:
        return None

    return _ParsedSearchPayload(
        query=resolved_query,
        raw_findings=raw_findings,
        display_sources=display_sources,
        source_quality_by_url=_build_source_quality_index(payload),
        birth_date=_extract_birth_date(payload, findings=raw_findings),
    )


def _read_findings(payload: Mapping[str, Any]) -> tuple[SearchGroundingFinding, ...]:
    raw_findings = payload.get("synthesized_findings")
    findings: list[SearchGroundingFinding] = []
    if isinstance(raw_findings, list):
        for entry in raw_findings:
            if not isinstance(entry, Mapping):
                continue
            text = entry.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            support_count = entry.get("support_count")
            score = entry.get("score")
            findings.append(
                SearchGroundingFinding(
                    text=text.strip(),
                    support_count=(
                        support_count if isinstance(support_count, int) and support_count > 0 else 1
                    ),
                    score=float(score) if isinstance(score, int | float) else 0.0,
                    source_titles=read_search_string_items(entry.get("source_titles")),
                    source_urls=read_search_string_items(entry.get("source_urls")),
                    confidence="supported",
                )
            )
    if findings:
        return tuple(findings)

    summary_points = read_search_string_items(payload.get("summary_points"))
    if not summary_points:
        return ()

    fallback_findings = [
        SearchGroundingFinding(
            text=item,
            support_count=1,
            score=6.0,
            source_titles=(),
            source_urls=(),
            confidence="supported",
        )
        for item in summary_points
    ]
    return tuple(fallback_findings)


def _classify_findings(
    findings: Sequence[SearchGroundingFinding],
    *,
    source_quality_by_url: Mapping[str, float],
    birth_date: date | None,
) -> tuple[SearchGroundingFinding, ...]:
    return tuple(
        _classify_finding(
            finding,
            source_quality_by_url=source_quality_by_url,
            birth_date=birth_date,
        )
        for finding in findings
    )


def _partition_findings_by_confidence(
    findings: Sequence[SearchGroundingFinding],
) -> tuple[tuple[SearchGroundingFinding, ...], tuple[SearchGroundingFinding, ...]]:
    supported_findings = tuple(
        finding
        for finding in findings
        if finding.confidence in {"strong", "supported"}
    )
    uncertain_findings = tuple(
        finding for finding in findings if finding.confidence == "uncertain"
    )
    return supported_findings, uncertain_findings


def _build_source_quality_index(payload: Mapping[str, Any]) -> dict[str, float]:
    raw_sources = payload.get("results")
    if not isinstance(raw_sources, list):
        return {}

    quality_by_url: dict[str, float] = {}
    for entry in raw_sources:
        if not isinstance(entry, Mapping):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        usefulness = entry.get("usefulness")
        evidence_count = entry.get("evidence_count")
        fetched = entry.get("fetched")
        used_snippet_fallback = entry.get("used_snippet_fallback")
        quality = float(usefulness) if isinstance(usefulness, int | float) else 0.0
        if isinstance(evidence_count, int):
            quality += min(evidence_count, 3) * 1.25
        if fetched is True:
            quality += 0.75
        if used_snippet_fallback is True:
            quality -= 0.75
        quality_by_url[url.strip()] = quality
    return quality_by_url


def _classify_finding(
    finding: SearchGroundingFinding,
    *,
    source_quality_by_url: Mapping[str, float],
    birth_date: date | None,
) -> SearchGroundingFinding:
    source_quality = max(
        (source_quality_by_url.get(url, 0.0) for url in finding.source_urls),
        default=0.0,
    )
    confidence = "uncertain"
    if finding.support_count >= 2:
        confidence = "strong"
    elif finding.support_count == 1 and (finding.score >= 6.0 or source_quality >= 6.0):
        confidence = "supported"

    if _finding_requires_extra_caution(finding.text, birth_date=birth_date):
        confidence = "uncertain"
    elif confidence == "uncertain" and finding.support_count == 1 and finding.score >= 4.0:
        confidence = "supported"

    return SearchGroundingFinding(
        text=finding.text,
        support_count=finding.support_count,
        score=finding.score,
        source_titles=finding.source_titles,
        source_urls=finding.source_urls,
        confidence=confidence,
    )


def _finding_requires_extra_caution(text: str, *, birth_date: date | None) -> bool:
    if _HANDLE_HINT_PATTERN.search(text) or _HANDLE_TERM_PATTERN.search(text):
        return True
    if _STALE_AS_OF_PATTERN.search(text):
        return True
    if _AGE_CLAIM_PATTERN.search(text):
        return birth_date is None
    if any(pattern.search(text) for pattern in _LOW_VALUE_DETAIL_PATTERNS):
        return True
    return False


def _extract_birth_date(
    payload: Mapping[str, Any],
    *,
    findings: Sequence[SearchGroundingFinding] | None = None,
) -> date | None:
    candidate_texts = _collect_birth_date_candidate_texts(
        payload,
        findings=findings,
    )
    for text in candidate_texts:
        parsed_date = _extract_birth_date_from_text(text)
        if parsed_date is not None:
            return parsed_date
    return None


def _collect_birth_date_candidate_texts(
    payload: Mapping[str, Any],
    *,
    findings: Sequence[SearchGroundingFinding] | None = None,
) -> tuple[str, ...]:
    candidate_texts = [finding.text for finding in findings or _read_findings(payload)]
    raw_evidence = payload.get("evidence")
    if isinstance(raw_evidence, list):
        for entry in raw_evidence:
            if not isinstance(entry, Mapping):
                continue
            text = entry.get("text")
            if isinstance(text, str) and text.strip():
                candidate_texts.append(text.strip())
    return tuple(candidate_texts)


def _extract_birth_date_from_text(text: str) -> date | None:
    match = re.search(
        r"\bborn(?: on)? (?P<date>[A-Z][a-z]+ \d{1,2}, \d{4}|"
        r"\d{4}-\d{2}-\d{2}|"
        r"\d{1,2} [A-Z][a-z]+ \d{4})\b",
        text,
    )
    if match is None:
        return None
    return _parse_human_date(match.group("date"))


def _parse_human_date(value: str) -> date | None:
    normalized_value = value.strip()
    iso_date = _parse_iso_date(normalized_value)
    if iso_date is not None:
        return iso_date
    for date_pattern in _MONTH_DATE_PATTERNS:
        try:
            return datetime.strptime(normalized_value, date_pattern).date()
        except ValueError:
            continue
    return None


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _reply_needs_rewrite(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
    query: str,
) -> bool:
    return any(
        (
            _STALE_AS_OF_PATTERN.search(reply_text) is not None,
            _contains_unconfirmed_handle_claim(reply_text, grounding=grounding),
            _contains_unsupported_age_claim(reply_text, grounding=grounding),
            _contains_unqualified_uncertain_finding(reply_text, grounding=grounding),
            _query_is_person_profile(query) and _contains_low_value_filler(reply_text),
            _contains_unnecessary_hedging(reply_text),
        )
    )


def _contains_unconfirmed_handle_claim(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
) -> bool:
    if not (_HANDLE_HINT_PATTERN.search(reply_text) or _HANDLE_TERM_PATTERN.search(reply_text)):
        return False
    return not _supported_findings_include_handle_details(grounding.supported_findings)


def _contains_unsupported_age_claim(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
) -> bool:
    if not _AGE_CLAIM_PATTERN.search(reply_text):
        return False
    return grounding.birth_date is None


def _contains_low_value_filler(reply_text: str) -> bool:
    return any(pattern.search(reply_text) for pattern in _LOW_VALUE_DETAIL_PATTERNS)


def _contains_unqualified_uncertain_finding(
    reply_text: str,
    *,
    grounding: SearchGroundingContext,
) -> bool:
    if _has_uncertainty_language(reply_text):
        return False

    reply_tokens = set(_content_tokens(reply_text))
    if not reply_tokens:
        return False

    for finding in grounding.uncertain_findings:
        finding_tokens = set(_content_tokens(finding.text))
        if len(finding_tokens) < 2:
            continue
        if _overlap_ratio(finding_tokens, reply_tokens) >= 0.6:
            return True
    return False


def _contains_unnecessary_hedging(reply_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:probably|it seems|seems to|appears to|often)\b",
            reply_text,
            flags=re.IGNORECASE,
        )
    )


def _has_uncertainty_language(reply_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:unconfirmed|not confirmed|not consistently confirmed|"
            r"one source|single source|some sources|reportedly|reported|"
            r"could not confirm|couldn't confirm|uncertain)\b",
            reply_text,
            flags=re.IGNORECASE,
        )
    )


def _sanitize_reply(reply_text: str) -> str:
    text = _STALE_AS_OF_PATTERN.sub("", reply_text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip(" \n")


def _compose_grounded_answer(
    *,
    query: str,
    grounding: SearchGroundingContext,
) -> str:
    if _query_is_age_request(query):
        age_answer = _compose_age_answer(grounding)
        if age_answer:
            return age_answer

    supported_facts = _select_supported_facts(grounding)
    if supported_facts:
        return _compose_supported_answer(
            query=query,
            supported_facts=supported_facts,
            grounding=grounding,
        )
    if grounding.uncertain_findings:
        return _compose_uncertain_only_answer(grounding)
    return "I couldn't confirm any strong details from the retrieved sources."


def _compose_age_answer(grounding: SearchGroundingContext) -> str | None:
    if grounding.birth_date is None:
        if grounding.uncertain_findings:
            return (
                "I found some biographical details, but I could not confirm a birth "
                "date strongly enough to compute a current age."
            )
        return None

    age = _compute_age(grounding.birth_date, grounding.current_date)
    return (
        f"I found a birth date of {grounding.birth_date.isoformat()}. "
        f"On {grounding.current_date.isoformat()}, that makes them {age} years old."
    )


def _compose_person_answer(
    *,
    supported_facts: Sequence[SearchGroundingFinding],
    grounding: SearchGroundingContext,
) -> str:
    return _compose_fact_answer(
        facts=supported_facts[:_MAX_COMPOSED_FACTS],
        grounding=grounding,
    )


def _compose_uncertain_only_answer(grounding: SearchGroundingContext) -> str:
    if _findings_include_handle_details(grounding.uncertain_findings):
        return (
            "I found some biographical details, but I did not see consistent support "
            "for specific social handles or usernames."
        )
    return (
        "I found a few mentions, but the details were not consistently confirmed "
        "across the retrieved sources."
    )


def _compose_uncertainty_note(grounding: SearchGroundingContext) -> str:
    if _findings_include_handle_details(grounding.uncertain_findings):
        return (
            "Some profile details, including possible social handles, were not "
            "consistently confirmed."
        )
    return "Some lower-confidence details were omitted because the sources did not confirm them consistently."


def _select_supported_facts(
    grounding: SearchGroundingContext,
) -> tuple[SearchGroundingFinding, ...]:
    supported = list(grounding.supported_findings)
    supported.sort(
        key=lambda finding: (
            0 if finding.confidence == "strong" else 1,
            -finding.support_count,
            -finding.score,
            finding.text.casefold(),
        )
    )

    selected: list[SearchGroundingFinding] = []
    selected_tokens: list[set[str]] = []
    identity_facts = [fact for fact in supported if _looks_like_identity_fact(fact.text)]
    other_facts = [fact for fact in supported if fact not in identity_facts]
    ordered_facts = identity_facts + other_facts

    for fact in ordered_facts:
        fact_tokens = set(_content_tokens(fact.text))
        if fact_tokens and any(
            _overlap_ratio(fact_tokens, existing_tokens) >= 0.7
            for existing_tokens in selected_tokens
        ):
            continue
        selected.append(fact)
        selected_tokens.append(fact_tokens)
        if len(selected) >= _MAX_COMPOSED_FACTS:
            break

    return tuple(selected)


def _build_tool_history_header_lines(
    grounding: SearchGroundingContext,
) -> list[str]:
    lines: list[str] = []
    if grounding.query:
        lines.append(f"Search request: {grounding.query}")
    lines.extend(
        (
            f"Grounding date: {grounding.current_date.isoformat()}",
            "",
            "Grounding rules:",
            "- Treat Supported facts as the evidence-backed facts for this search.",
            "- If a detail appears only under Uncertain details, it is not confirmed.",
            (
                "- Do not invent relative dates. Give age only if it can be computed "
                "from a retrieved birth date and the grounding date above."
            ),
            "",
            "Supported facts:",
        )
    )
    return lines


def _build_supported_history_lines(grounding: SearchGroundingContext) -> list[str]:
    if not grounding.supported_findings:
        return ["- No firm details were confirmed across the retrieved sources."]
    return [
        _format_history_finding_bullet(finding)
        for finding in grounding.supported_findings[:_MAX_COMPOSED_FACTS + 2]
    ]


def _build_uncertain_history_lines(grounding: SearchGroundingContext) -> list[str]:
    if not grounding.uncertain_findings:
        return []
    return [
        "",
        "Uncertain details:",
        *(
            _format_history_finding_bullet(finding, confidence="uncertain")
            for finding in grounding.uncertain_findings[:3]
        ),
    ]


def _build_birth_date_history_lines(grounding: SearchGroundingContext) -> list[str]:
    if grounding.birth_date is None:
        return []
    return ["", f"Retrieved birth date: {grounding.birth_date.isoformat()}"]


def _build_source_history_lines(grounding: SearchGroundingContext) -> list[str]:
    if not grounding.display_sources:
        return []
    return [
        "",
        "Sources:",
        *(_format_source_bullet(title, url) for title, url in grounding.display_sources),
    ]


def _format_history_finding_bullet(
    finding: SearchGroundingFinding,
    *,
    confidence: str | None = None,
) -> str:
    resolved_confidence = confidence or finding.confidence
    return (
        f"- [{resolved_confidence}; {finding.support_count} source"
        f"{'' if finding.support_count == 1 else 's'}] {finding.text}"
    )


def _format_source_bullet(title: str, url: str) -> str:
    if title:
        return f"- {title}: {url}"
    return f"- {url}"


def _parse_search_tool_history_line(
    stripped: str,
    state: _SearchToolHistoryState,
) -> None:
    if not stripped:
        return
    if _parse_search_tool_history_metadata(stripped, state):
        return
    if _parse_search_tool_history_section(stripped, state):
        return
    if state.section in {"supported", "uncertain"}:
        finding = _parse_history_finding_line(stripped)
        if finding is None:
            return
        if state.section == "supported":
            state.supported_findings.append(finding)
        else:
            state.uncertain_findings.append(finding)
        return
    if state.section == "sources":
        source = _parse_history_source_line(stripped)
        if source is not None:
            state.display_sources.append(source)


def _parse_search_tool_history_metadata(
    stripped: str,
    state: _SearchToolHistoryState,
) -> bool:
    if stripped.startswith("Search request:"):
        state.query = stripped.partition(":")[2].strip()
        return True
    if stripped.startswith("Grounding date:"):
        parsed_date = _parse_iso_date(stripped.partition(":")[2].strip())
        if parsed_date is not None:
            state.current_date = parsed_date
        return True
    if stripped.startswith("Retrieved birth date:"):
        state.birth_date = _parse_iso_date(stripped.partition(":")[2].strip())
        return True
    return False


def _parse_search_tool_history_section(
    stripped: str,
    state: _SearchToolHistoryState,
) -> bool:
    if stripped == "Supported facts:":
        state.section = "supported"
        return True
    if stripped == "Uncertain details:":
        state.section = "uncertain"
        return True
    if stripped == "Sources:":
        state.section = "sources"
        return True
    return False


def _parse_history_finding_line(stripped: str) -> SearchGroundingFinding | None:
    match = _FINDING_BULLET_PATTERN.match(stripped)
    if match is None:
        return None
    return SearchGroundingFinding(
        text=match.group("text").strip(),
        support_count=max(int(match.group("support")), 1),
        score=0.0,
        source_titles=(),
        source_urls=(),
        confidence=match.group("label").casefold(),
    )


def _parse_history_source_line(stripped: str) -> tuple[str, str] | None:
    source_match = _SOURCE_BULLET_PATTERN.match(stripped)
    if source_match is None:
        return None
    return (
        source_match.group("title").strip(),
        source_match.group("url").strip(),
    )


def _supported_findings_include_handle_details(
    findings: Sequence[SearchGroundingFinding],
) -> bool:
    return _text_contains_handle_details(" ".join(finding.text for finding in findings))


def _findings_include_handle_details(
    findings: Sequence[SearchGroundingFinding],
) -> bool:
    return any(_text_contains_handle_details(finding.text) for finding in findings)


def _text_contains_handle_details(text: str) -> bool:
    return bool(
        _HANDLE_HINT_PATTERN.search(text) or _HANDLE_TERM_PATTERN.search(text)
    )


def _compose_supported_answer(
    *,
    query: str,
    supported_facts: Sequence[SearchGroundingFinding],
    grounding: SearchGroundingContext,
) -> str:
    if _query_is_person_profile(query):
        return _compose_person_answer(
            supported_facts=supported_facts,
            grounding=grounding,
        )
    return _compose_fact_answer(
        facts=supported_facts[:_MAX_COMPOSED_FACTS],
        grounding=grounding,
    )


def _compose_fact_answer(
    *,
    facts: Sequence[SearchGroundingFinding],
    grounding: SearchGroundingContext,
) -> str:
    sentences = _finding_sentences(facts)
    if grounding.uncertain_findings:
        sentences.append(_compose_uncertainty_note(grounding))
    return " ".join(sentences).strip()


def _finding_sentences(
    findings: Sequence[SearchGroundingFinding],
) -> list[str]:
    return [finding.text.rstrip(".") + "." for finding in findings]


def _looks_like_identity_fact(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:is|was|est|etait|était)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _query_is_person_profile(query: str) -> bool:
    folded_query = _fold_for_match(query)
    return any(pattern.search(folded_query) for pattern in _PERSON_QUERY_PATTERNS)


def _query_is_age_request(query: str) -> bool:
    folded_query = _fold_for_match(query)
    return any(pattern.search(folded_query) for pattern in _AGE_QUERY_PATTERNS)


def _compute_age(birth_date: date, current_date: date) -> int:
    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def _fold_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_accents))


def _content_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token for token in _fold_for_match(text).split() if len(token) > 2 or token.isdigit()
    )


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / min(len(left), len(right))


def _is_search_tool_message(message: ChatMessage) -> bool:
    return (
        message.role is MessageRole.TOOL
        and message.content.startswith(_SEARCH_TOOL_PREFIX)
    )


__all__ = [
    "SearchGroundingContext",
    "SearchGroundingFinding",
    "build_search_answer_contract",
    "build_search_grounding_context",
    "build_search_tool_history_summary",
    "has_search_grounding_context",
    "parse_search_tool_history",
    "shape_reply_with_grounding",
    "shape_search_backed_reply",
    "should_apply_search_grounding",
]
