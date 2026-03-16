"""Deterministic session summarization for persisted chat messages."""

from __future__ import annotations

from collections.abc import Iterable

from unclaw.core.search_grounding import parse_search_tool_history
from unclaw.schemas.chat import ChatMessage, MessageRole

_MAX_INTENT_COUNT = 3
_MAX_INTENT_LENGTH = 80
_MAX_REPLY_LENGTH = 120
_MAX_RETAINED_FACT_COUNT = 3
_MAX_RETAINED_UNCERTAINTY_COUNT = 2
_MAX_RETAINED_FINDING_LENGTH = 120


def summarize_session_messages(messages: Iterable[ChatMessage]) -> str:
    """Build a compact rule-based summary from persisted session messages."""

    ordered_messages = list(messages)
    if not ordered_messages:
        return "No messages yet."

    user_intents = _collect_recent_user_intents(ordered_messages)
    retained_grounded_facts = _collect_retained_search_findings(
        ordered_messages,
        include_uncertain=False,
        limit=_MAX_RETAINED_FACT_COUNT,
    )
    retained_uncertain_details = _collect_retained_search_findings(
        ordered_messages,
        include_uncertain=True,
        limit=_MAX_RETAINED_UNCERTAINTY_COUNT,
    )
    latest_assistant_reply = _find_latest_reply(
        ordered_messages,
        role=MessageRole.ASSISTANT,
        limit=_MAX_REPLY_LENGTH,
    )

    user_count = sum(1 for message in ordered_messages if message.role == MessageRole.USER)
    assistant_count = sum(
        1 for message in ordered_messages if message.role == MessageRole.ASSISTANT
    )
    tool_count = sum(1 for message in ordered_messages if message.role == MessageRole.TOOL)

    parts: list[str] = []
    if user_intents:
        label = "User intent" if len(user_intents) == 1 else "Recent user intents"
        parts.append(f"{label}: {'; '.join(user_intents)}.")

    if retained_grounded_facts:
        label = (
            "Retained grounded fact"
            if len(retained_grounded_facts) == 1
            else "Retained grounded facts"
        )
        parts.append(f"{label}: {'; '.join(retained_grounded_facts)}.")

    if retained_uncertain_details:
        label = (
            "Retained uncertainty"
            if len(retained_uncertain_details) == 1
            else "Retained uncertainties"
        )
        parts.append(f"{label}: {'; '.join(retained_uncertain_details)}.")

    if latest_assistant_reply is not None:
        parts.append(f"Latest assistant reply: {latest_assistant_reply}.")

    if not parts:
        parts.append(
            "Session has messages but no user, assistant, or grounded tool content yet."
        )

    parts.append(
        "Session size: "
        f"{len(ordered_messages)} messages "
        f"({user_count} user, {assistant_count} assistant, {tool_count} tool)."
    )
    return " ".join(parts)


def _collect_recent_user_intents(messages: list[ChatMessage]) -> list[str]:
    snippets: list[str] = []
    seen_snippets: set[str] = set()

    for message in reversed(messages):
        if message.role != MessageRole.USER:
            continue

        snippet = _summary_fragment(message.content, limit=_MAX_INTENT_LENGTH)
        if snippet is None:
            continue

        normalized_snippet = snippet.casefold()
        if normalized_snippet in seen_snippets:
            continue

        snippets.append(snippet)
        seen_snippets.add(normalized_snippet)
        if len(snippets) >= _MAX_INTENT_COUNT:
            break

    snippets.reverse()
    return snippets


def _collect_retained_search_findings(
    messages: list[ChatMessage],
    *,
    include_uncertain: bool,
    limit: int,
) -> list[str]:
    findings: list[str] = []
    seen_findings: set[str] = set()

    for message in reversed(messages):
        if message.role is not MessageRole.TOOL:
            continue

        grounding = parse_search_tool_history(message.content)
        if grounding is None:
            continue

        query_label = _summary_fragment(grounding.query, limit=_MAX_INTENT_LENGTH)
        source_findings = (
            grounding.uncertain_findings
            if include_uncertain
            else grounding.supported_findings
        )

        for finding in source_findings:
            finding_text = _summary_fragment(
                finding.text,
                limit=_MAX_RETAINED_FINDING_LENGTH,
            )
            if finding_text is None:
                continue

            normalized_finding = finding_text.casefold()
            if normalized_finding in seen_findings:
                continue

            seen_findings.add(normalized_finding)
            if query_label is None:
                findings.append(finding_text)
            else:
                findings.append(f"[{query_label}] {finding_text}")

            if len(findings) >= limit:
                findings.reverse()
                return findings

    findings.reverse()
    return findings


def _find_latest_reply(
    messages: list[ChatMessage],
    *,
    role: MessageRole,
    limit: int,
) -> str | None:
    for message in reversed(messages):
        if message.role != role:
            continue

        snippet = _summary_fragment(message.content, limit=limit)
        if snippet is not None:
            return snippet

    return None


def _message_snippet(content: str, *, limit: int) -> str | None:
    normalized = " ".join(content.split()).strip()
    if not normalized:
        return None
    return _clip_text(normalized, limit=limit)


def _summary_fragment(content: str, *, limit: int) -> str | None:
    snippet = _message_snippet(content, limit=limit)
    if snippet is None:
        return None

    cleaned_snippet = snippet.rstrip(" .!?;:")
    return cleaned_snippet or None


def _clip_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value

    clipped = value[: limit - 3].rstrip(" ,;:.")
    return f"{clipped}..."
