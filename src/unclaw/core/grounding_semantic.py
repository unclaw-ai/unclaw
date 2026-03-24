"""Semantic/model-assisted helpers for search grounding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re
from typing import TYPE_CHECKING, Any

from unclaw.llm.base import LLMMessage, LLMRole
from unclaw.settings import Settings

if TYPE_CHECKING:
    from unclaw.core.search_grounding import (
        SearchGroundingContext,
        SearchGroundingFinding,
    )

_SEARCH_GROUNDING_QUERY_ANALYZER_TIMEOUT_SECONDS = 8.0
_SEARCH_GROUNDING_REPLY_REVIEW_TIMEOUT_SECONDS = 8.0
_SEMANTIC_QUERY_KINDS = frozenset({"age_request", "person_profile", "general"})
_SEARCH_GROUNDING_QUERY_ANALYZER_SYSTEM_PROMPT = (
    "Decide whether the user's latest turn should stay grounded in the most "
    "recent search grounding context. Work semantically across languages. "
    "Return JSON only with keys applies_to_grounding, query_kind, and "
    "is_follow_up. applies_to_grounding and is_follow_up must be booleans. "
    "query_kind must be one of: age_request, person_profile, general. "
    "Set applies_to_grounding true when the current turn is still about the "
    "same entity/topic or clearly refers back to the grounded search context."
)
_SEARCH_GROUNDING_REPLY_REVIEW_SYSTEM_PROMPT = (
    "Review a candidate answer against grounded search evidence. Work "
    "semantically across languages. Return JSON only with keys "
    "rewrite_required, query_kind, safe_answer, and issues. "
    "rewrite_required must be a boolean. query_kind must be one of: "
    "age_request, person_profile, general. safe_answer must be a short "
    "natural-language string and should be empty when no rewrite is needed. "
    "issues must be a JSON array of short lowercase codes. If the candidate "
    "answer is unsupported, stale, or overconfident, set rewrite_required "
    "true and write a safe_answer in the same language as the user's query. "
    "For age questions, give age only when a birth date is available and "
    "compute it from grounding_date and birth_date. Otherwise give the birth "
    "date only or say the age is not confirmed. Never confirm social handles, "
    "usernames, or weak profile details unless the grounding clearly supports them."
)


@dataclass(frozen=True, slots=True)
class _SemanticQueryAnalysis:
    applies_to_grounding: bool
    query_kind: str
    is_follow_up: bool


@dataclass(frozen=True, slots=True)
class _SemanticReplyReview:
    rewrite_required: bool
    query_kind: str
    safe_answer: str
    issues: tuple[str, ...]


def _analyze_query_semantically(
    *,
    query: str,
    grounding: SearchGroundingContext,
    settings: Settings,
    model_profile_name: str,
) -> _SemanticQueryAnalysis | None:
    content = _run_semantic_grounding_request(
        settings=settings,
        model_profile_name=model_profile_name,
        messages=_build_query_analysis_messages(query=query, grounding=grounding),
        timeout_seconds=_SEARCH_GROUNDING_QUERY_ANALYZER_TIMEOUT_SECONDS,
    )
    if content is None:
        return None
    return _parse_semantic_query_analysis(content)


def _review_reply_semantically(
    *,
    query: str,
    grounding: SearchGroundingContext,
    reply_text: str,
    settings: Settings,
    model_profile_name: str,
) -> _SemanticReplyReview | None:
    content = _run_semantic_grounding_request(
        settings=settings,
        model_profile_name=model_profile_name,
        messages=_build_reply_review_messages(
            query=query,
            grounding=grounding,
            reply_text=reply_text,
        ),
        timeout_seconds=_SEARCH_GROUNDING_REPLY_REVIEW_TIMEOUT_SECONDS,
    )
    if content is None:
        return None
    return _parse_semantic_reply_review(content)


def _run_semantic_grounding_request(
    *,
    settings: Settings,
    model_profile_name: str,
    messages: Sequence[LLMMessage],
    timeout_seconds: float,
) -> str | None:
    from unclaw.core.grounding_model_call import run_grounding_model_call

    return run_grounding_model_call(
        settings=settings,
        model_profile_name=model_profile_name,
        messages=messages,
        timeout_seconds=timeout_seconds,
    )


def _build_query_analysis_messages(
    *,
    query: str,
    grounding: SearchGroundingContext,
) -> tuple[LLMMessage, LLMMessage]:
    return (
        LLMMessage(
            role=LLMRole.SYSTEM,
            content=_SEARCH_GROUNDING_QUERY_ANALYZER_SYSTEM_PROMPT,
        ),
        LLMMessage(
            role=LLMRole.USER,
            content=_build_query_analysis_payload(query=query, grounding=grounding),
        ),
    )


def _build_query_analysis_payload(
    *,
    query: str,
    grounding: SearchGroundingContext,
) -> str:
    lines = [
        f"Grounding date: {grounding.current_date.isoformat()}",
        f"Current user turn: {query}",
    ]
    if grounding.query:
        lines.append(f"Most recent grounded search request: {grounding.query}")
    lines.extend(_build_semantic_grounding_fact_lines(grounding))
    lines.append("Return JSON only.")
    return "\n".join(lines)


def _build_reply_review_messages(
    *,
    query: str,
    grounding: SearchGroundingContext,
    reply_text: str,
) -> tuple[LLMMessage, LLMMessage]:
    return (
        LLMMessage(
            role=LLMRole.SYSTEM,
            content=_SEARCH_GROUNDING_REPLY_REVIEW_SYSTEM_PROMPT,
        ),
        LLMMessage(
            role=LLMRole.USER,
            content=_build_reply_review_payload(
                query=query,
                grounding=grounding,
                reply_text=reply_text,
            ),
        ),
    )


def _build_reply_review_payload(
    *,
    query: str,
    grounding: SearchGroundingContext,
    reply_text: str,
) -> str:
    lines = [
        f"Grounding date: {grounding.current_date.isoformat()}",
        f"User query: {query}",
        f"Candidate answer: {reply_text}",
    ]
    if grounding.birth_date is not None:
        lines.append(f"Retrieved birth date: {grounding.birth_date.isoformat()}")
    lines.extend(_build_semantic_grounding_fact_lines(grounding))
    lines.append("Return JSON only.")
    return "\n".join(lines)


def _build_semantic_grounding_fact_lines(
    grounding: SearchGroundingContext,
) -> list[str]:
    from unclaw.core.search_grounding import _MAX_COMPOSED_FACTS

    lines = ["Supported facts:"]
    if grounding.supported_findings:
        lines.extend(
            _format_semantic_finding_line(finding)
            for finding in grounding.supported_findings[:_MAX_COMPOSED_FACTS + 1]
        )
    else:
        lines.append("- none")

    lines.append("Uncertain details:")
    if grounding.uncertain_findings:
        lines.extend(
            _format_semantic_finding_line(finding)
            for finding in grounding.uncertain_findings[:3]
        )
    else:
        lines.append("- none")
    return lines


def _format_semantic_finding_line(finding: SearchGroundingFinding) -> str:
    return (
        f"- [{finding.confidence}; {finding.support_count} source"
        f"{'' if finding.support_count == 1 else 's'}] {finding.text}"
    )


def _parse_semantic_query_analysis(content: str) -> _SemanticQueryAnalysis | None:
    payload = _parse_semantic_json_object(content)
    if payload is None:
        return None

    applies_to_grounding = payload.get("applies_to_grounding")
    query_kind = payload.get("query_kind")
    is_follow_up = payload.get("is_follow_up")
    if not isinstance(applies_to_grounding, bool):
        return None
    if query_kind not in _SEMANTIC_QUERY_KINDS:
        return None
    if not isinstance(is_follow_up, bool):
        return None

    return _SemanticQueryAnalysis(
        applies_to_grounding=applies_to_grounding,
        query_kind=query_kind,
        is_follow_up=is_follow_up,
    )


def _parse_semantic_reply_review(content: str) -> _SemanticReplyReview | None:
    payload = _parse_semantic_json_object(content)
    if payload is None:
        return None

    rewrite_required = payload.get("rewrite_required")
    query_kind = payload.get("query_kind")
    safe_answer = payload.get("safe_answer")
    if not isinstance(rewrite_required, bool):
        return None
    if query_kind not in _SEMANTIC_QUERY_KINDS:
        return None
    if not isinstance(safe_answer, str):
        return None

    raw_issues = payload.get("issues")
    issues: tuple[str, ...]
    if isinstance(raw_issues, list):
        issues = tuple(
            item.strip()
            for item in raw_issues
            if isinstance(item, str) and item.strip()
        )
    else:
        issues = ()

    return _SemanticReplyReview(
        rewrite_required=rewrite_required,
        query_kind=query_kind,
        safe_answer=safe_answer.strip(),
        issues=issues,
    )


def _parse_semantic_json_object(content: str) -> dict[str, Any] | None:
    normalized = content.strip()
    if not normalized:
        return None

    candidates = [normalized]
    if normalized.startswith("```"):
        stripped = re.sub(
            r"^```(?:json)?\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\s*```$", "", stripped)
        candidates.append(stripped.strip())

    start_index = normalized.find("{")
    end_index = normalized.rfind("}")
    if 0 <= start_index < end_index:
        candidates.append(normalized[start_index : end_index + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    return None


__all__ = [
    "_SEARCH_GROUNDING_QUERY_ANALYZER_SYSTEM_PROMPT",
    "_SEARCH_GROUNDING_QUERY_ANALYZER_TIMEOUT_SECONDS",
    "_SEARCH_GROUNDING_REPLY_REVIEW_SYSTEM_PROMPT",
    "_SEARCH_GROUNDING_REPLY_REVIEW_TIMEOUT_SECONDS",
    "_SEMANTIC_QUERY_KINDS",
    "_SemanticQueryAnalysis",
    "_SemanticReplyReview",
    "_analyze_query_semantically",
    "_build_query_analysis_messages",
    "_build_query_analysis_payload",
    "_build_reply_review_messages",
    "_build_reply_review_payload",
    "_build_semantic_grounding_fact_lines",
    "_format_semantic_finding_line",
    "_parse_semantic_json_object",
    "_parse_semantic_query_analysis",
    "_parse_semantic_reply_review",
    "_review_reply_semantically",
    "_run_semantic_grounding_request",
]
