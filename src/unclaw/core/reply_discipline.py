"""Post-tool reply-discipline helpers."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

from unclaw.core.routing import (
    _looks_like_deep_search_request,
    _normalize_runtime_routing_text,
)
from unclaw.tools.contracts import ToolResult

_DEFAULT_TURN_CANCELLED_REPLY = (
    "This request was cancelled before tool work completed."
)
_LIMITATION_ACK_PATTERN = re.compile(
    r"\b(?:could not|couldn't|cannot|can't|did not|didn't|not confirmed|"
    r"unconfirmed|limited|partial|sparse|insufficient|failed|timed out|timeout|"
    r"je n'ai pas pu|je ne peux pas|je ne pouvais pas|non confirme|"
    r"pas confirme|limite|limitee|partiel|partielle|echec|a expire|expire)\b",
    flags=re.IGNORECASE,
)


def _tool_result_payload_dict(tool_result: ToolResult) -> dict[str, Any]:
    return tool_result.payload if isinstance(tool_result.payload, dict) else {}


def _tool_result_timed_out(tool_result: ToolResult) -> bool:
    haystack = " ".join(
        part for part in (tool_result.error, tool_result.output_text) if part
    )
    return "timed out" in haystack.casefold()


def _extract_fast_grounding_points(tool_result: ToolResult) -> tuple[str, ...]:
    grounding_note = _tool_result_payload_dict(tool_result).get("grounding_note")
    note = grounding_note if isinstance(grounding_note, str) else tool_result.output_text
    return tuple(
        line[2:].strip()
        for line in note.splitlines()
        if line.strip().startswith("- ")
    )


def _fast_web_search_result_is_mismatch(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return False
    note = tool_result.output_text.casefold()
    return (
        "different entity" in note
        or "no exact top match" in note
        or "no web results found" in note
    )


def _fast_web_search_result_is_thin(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return True

    payload = _tool_result_payload_dict(tool_result)
    result_count = payload.get("result_count")
    supported_points = _extract_fast_grounding_points(tool_result)
    if _fast_web_search_result_is_mismatch(tool_result):
        return True
    if isinstance(result_count, int) and result_count <= 1:
        return True
    return len(supported_points) <= 1


def _search_web_result_is_thin(tool_result: ToolResult) -> bool:
    if tool_result.success is not True:
        return True

    payload = _tool_result_payload_dict(tool_result)
    evidence_count = payload.get("evidence_count")
    finding_count = payload.get("finding_count")
    display_sources = payload.get("display_sources")

    if isinstance(finding_count, int) and finding_count <= 1:
        return True
    if isinstance(evidence_count, int) and evidence_count <= 1:
        return True
    if isinstance(display_sources, list) and len(display_sources) <= 1:
        return True
    return False


def _reply_acknowledges_limitations(reply: str) -> bool:
    return _LIMITATION_ACK_PATTERN.search(reply) is not None


def _detect_runtime_reply_language(user_input: str) -> str:
    normalized = _normalize_runtime_routing_text(user_input)
    if any(
        token in normalized.split()
        for token in (
            "biographie",
            "de",
            "est",
            "fais",
            "leur",
            "qui",
            "recherche",
            "sur",
            "quoi",
        )
    ):
        return "fr"
    return "en"


def _reply_sentence_count(text: str) -> int:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    return len(sentences) if sentences else (1 if text.strip() else 0)


def _build_all_failed_tool_reply(
    *,
    user_input: str,
    tool_results: Sequence[ToolResult],
) -> str:
    language = _detect_runtime_reply_language(user_input)
    timeout_failed = any(_tool_result_timed_out(tool_result) for tool_result in tool_results)

    if language == "fr":
        if timeout_failed:
            return (
                "L'etape outil a expire, donc je n'ai pas pu confirmer les details "
                "demandes a partir des resultats recuperes."
            )
        return (
            "L'etape outil a echoue, donc je n'ai pas pu confirmer les details "
            "demandes a partir des resultats recuperes."
        )

    if timeout_failed:
        return (
            "The tool step timed out, so I couldn't confirm the requested details "
            "from retrieved tool evidence."
        )
    return (
        "The tool step failed, so I couldn't confirm the requested details "
        "from retrieved tool evidence."
    )


def _build_fast_grounding_guarded_reply(
    *,
    candidate_reply: str,
    user_input: str,
    fast_results: Sequence[ToolResult],
) -> str | None:
    if not fast_results:
        return None

    language = _detect_runtime_reply_language(user_input)
    supported_points = [
        point
        for tool_result in fast_results
        for point in _extract_fast_grounding_points(tool_result)[:1]
    ]
    any_mismatch = any(
        _fast_web_search_result_is_mismatch(tool_result) for tool_result in fast_results
    )
    all_thin = all(_fast_web_search_result_is_thin(tool_result) for tool_result in fast_results)
    full_bio_requested = _looks_like_deep_search_request(user_input)
    reply_is_minimal = _reply_sentence_count(candidate_reply) <= 1

    if len(fast_results) > 1 and (any_mismatch or all_thin):
        if language == "fr":
            return (
                "Les resultats de grounding rapide etaient limites ou melanges, "
                "donc je ne peux confirmer que des fragments partiels pour ces "
                "entites. Je ne peux pas confirmer des biographies completes a partir de ces seules recherches rapides."
            )
        return (
            "The quick grounding results were limited or mixed, so I can only "
            "confirm partial fragments for these entities. I couldn't confirm "
            "full biographies from those probes alone."
        )

    primary_point = supported_points[0] if supported_points else ""
    if any_mismatch and not primary_point:
        if language == "fr":
            return (
                "Le grounding web rapide semblait pointer vers une autre entite, "
                "donc je ne peux pas confirmer les details demandes a partir de "
                "ce resultat seul."
            )
        return (
            "The quick web grounding appeared to match a different entity, so I "
            "couldn't confirm the requested details from that result alone."
        )

    if primary_point and (
        full_bio_requested
        or (
            all_thin
            and not reply_is_minimal
            and primary_point.casefold() not in candidate_reply.casefold()
        )
        or (all_thin and _reply_sentence_count(candidate_reply) > 1)
    ):
        if language == "fr":
            return (
                f"{primary_point} Je ne peux pas confirmer une biographie plus "
                "complete a partir de ce grounding rapide seul."
            )
        return (
            f"{primary_point} I couldn't confirm a fuller biography from that "
            "quick grounding probe alone."
        )

    return None


def _apply_post_tool_reply_discipline(
    *,
    reply: str,
    user_input: str,
    tool_results: Sequence[ToolResult],
    turn_cancelled_reply: str = _DEFAULT_TURN_CANCELLED_REPLY,
) -> str:
    stripped_reply = reply.strip()
    if not stripped_reply or not tool_results:
        return stripped_reply
    if stripped_reply == turn_cancelled_reply:
        return stripped_reply

    if all(tool_result.success is False for tool_result in tool_results):
        if _reply_acknowledges_limitations(stripped_reply):
            return stripped_reply
        return _build_all_failed_tool_reply(
            user_input=user_input,
            tool_results=tool_results,
        )

    successful_search_web = any(
        tool_result.tool_name == "search_web" and tool_result.success
        for tool_result in tool_results
    )
    successful_fast_web = [
        tool_result
        for tool_result in tool_results
        if tool_result.tool_name == "fast_web_search" and tool_result.success
    ]
    if successful_fast_web and not successful_search_web:
        guarded_reply = _build_fast_grounding_guarded_reply(
            candidate_reply=stripped_reply,
            user_input=user_input,
            fast_results=successful_fast_web,
        )
        if guarded_reply is not None:
            # Preserve already-minimal honest replies when they are shorter than
            # our fallback and clearly acknowledge the result limits.
            if _reply_acknowledges_limitations(stripped_reply) and (
                len(stripped_reply) <= len(guarded_reply)
                or _reply_sentence_count(stripped_reply) <= 2
            ):
                return stripped_reply
            return guarded_reply

    return stripped_reply
