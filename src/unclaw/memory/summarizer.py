"""Deterministic session summarization for persisted chat messages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from typing import Any

from unclaw.constants import (
    SESSION_SUMMARY_FINDING_CHARACTER_LIMIT,
    SESSION_SUMMARY_INTENT_CHARACTER_LIMIT,
    SESSION_SUMMARY_INTENT_LIMIT,
    SESSION_SUMMARY_RETAINED_FACT_LIMIT,
    SESSION_SUMMARY_RETAINED_UNCERTAINTY_LIMIT,
    SESSION_SUMMARY_REPLY_CHARACTER_LIMIT,
)
from unclaw.core.search_grounding import parse_search_tool_history
from unclaw.schemas.chat import ChatMessage, MessageRole

_SESSION_MEMORY_SCHEMA = "unclaw.session_memory"
_SESSION_MEMORY_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionMemoryFinding:
    """One retained memory item extracted from session history."""

    text: str
    query: str | None = None


@dataclass(frozen=True, slots=True)
class SessionMemoryStats:
    """Compact counts captured when one session summary is built."""

    message_count: int
    user_message_count: int
    assistant_message_count: int
    tool_message_count: int

    @classmethod
    def from_messages(cls, messages: Iterable[ChatMessage]) -> "SessionMemoryStats":
        ordered_messages = tuple(messages)
        return cls(
            message_count=len(ordered_messages),
            user_message_count=sum(
                1 for message in ordered_messages if message.role == MessageRole.USER
            ),
            assistant_message_count=sum(
                1
                for message in ordered_messages
                if message.role == MessageRole.ASSISTANT
            ),
            tool_message_count=sum(
                1 for message in ordered_messages if message.role == MessageRole.TOOL
            ),
        )


@dataclass(frozen=True, slots=True)
class StructuredSessionMemory:
    """Explicit persisted session-memory representation."""

    recent_user_intents: tuple[str, ...]
    retained_facts: tuple[SessionMemoryFinding, ...]
    retained_uncertainties: tuple[SessionMemoryFinding, ...]
    latest_assistant_reply: str | None
    stats: SessionMemoryStats
    summary_text: str

    @classmethod
    def from_legacy_text(
        cls,
        summary_text: str,
        *,
        stats: SessionMemoryStats | None = None,
    ) -> "StructuredSessionMemory":
        normalized_summary = summary_text.strip() or "No messages yet."
        return cls(
            recent_user_intents=(),
            retained_facts=(),
            retained_uncertainties=(),
            latest_assistant_reply=None,
            stats=stats or SessionMemoryStats(0, 0, 0, 0),
            summary_text=normalized_summary,
        )


def summarize_session_messages(messages: Iterable[ChatMessage]) -> str:
    """Build a compact rule-based summary from persisted session messages."""
    return build_structured_session_memory(messages).summary_text


def build_structured_session_memory(
    messages: Iterable[ChatMessage],
) -> StructuredSessionMemory:
    """Build a typed session-memory summary from persisted chat messages."""

    ordered_messages = list(messages)
    stats = SessionMemoryStats.from_messages(ordered_messages)
    if not ordered_messages:
        return StructuredSessionMemory(
            recent_user_intents=(),
            retained_facts=(),
            retained_uncertainties=(),
            latest_assistant_reply=None,
            stats=stats,
            summary_text="No messages yet.",
        )

    user_intents = _collect_recent_user_intents(ordered_messages)
    retained_grounded_facts = tuple(
        _collect_retained_search_findings(
            ordered_messages,
            include_uncertain=False,
            limit=SESSION_SUMMARY_RETAINED_FACT_LIMIT,
        )
    )
    retained_uncertain_details = tuple(
        _collect_retained_search_findings(
            ordered_messages,
            include_uncertain=True,
            limit=SESSION_SUMMARY_RETAINED_UNCERTAINTY_LIMIT,
        )
    )
    latest_assistant_reply = _find_latest_reply(
        ordered_messages,
        role=MessageRole.ASSISTANT,
        limit=SESSION_SUMMARY_REPLY_CHARACTER_LIMIT,
    )

    summary_text = _compose_summary_text(
        recent_user_intents=tuple(user_intents),
        retained_facts=retained_grounded_facts,
        retained_uncertainties=retained_uncertain_details,
        latest_assistant_reply=latest_assistant_reply,
        stats=stats,
    )
    return StructuredSessionMemory(
        recent_user_intents=tuple(user_intents),
        retained_facts=retained_grounded_facts,
        retained_uncertainties=retained_uncertain_details,
        latest_assistant_reply=latest_assistant_reply,
        stats=stats,
        summary_text=summary_text,
    )


def render_session_memory_summary(summary: StructuredSessionMemory) -> str:
    """Render one structured session-memory object to summary text."""
    if summary.summary_text.strip():
        return summary.summary_text

    return _compose_summary_text(
        recent_user_intents=summary.recent_user_intents,
        retained_facts=summary.retained_facts,
        retained_uncertainties=summary.retained_uncertainties,
        latest_assistant_reply=summary.latest_assistant_reply,
        stats=summary.stats,
    )


def serialize_structured_session_memory(summary: StructuredSessionMemory) -> str:
    """Serialize one structured summary into the existing summary_text column."""
    payload = {
        "schema": _SESSION_MEMORY_SCHEMA,
        "version": _SESSION_MEMORY_VERSION,
        "summary_text": render_session_memory_summary(summary),
        "recent_user_intents": list(summary.recent_user_intents),
        "retained_facts": [
            _serialize_memory_finding(finding) for finding in summary.retained_facts
        ],
        "retained_uncertainties": [
            _serialize_memory_finding(finding)
            for finding in summary.retained_uncertainties
        ],
        "latest_assistant_reply": summary.latest_assistant_reply,
        "message_stats": {
            "message_count": summary.stats.message_count,
            "user_message_count": summary.stats.user_message_count,
            "assistant_message_count": summary.stats.assistant_message_count,
            "tool_message_count": summary.stats.tool_message_count,
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_persisted_session_memory(
    persisted_summary: str | None,
    *,
    fallback_stats: SessionMemoryStats | None = None,
) -> StructuredSessionMemory | None:
    """Parse a stored summary from either structured or legacy plain text."""
    if persisted_summary is None:
        return None

    normalized_summary = persisted_summary.strip()
    if not normalized_summary:
        return None

    parsed_payload = _load_persisted_summary_payload(normalized_summary)
    if parsed_payload is None:
        return StructuredSessionMemory.from_legacy_text(
            normalized_summary,
            stats=fallback_stats,
        )

    resolved_stats = _parse_memory_stats(
        parsed_payload.get("message_stats"),
        fallback=fallback_stats,
    )
    summary_text = _parse_optional_text(parsed_payload.get("summary_text"))
    if summary_text is None:
        summary_text = _compose_summary_text(
            recent_user_intents=_parse_string_tuple(
                parsed_payload.get("recent_user_intents")
            ),
            retained_facts=_parse_memory_findings(parsed_payload.get("retained_facts")),
            retained_uncertainties=_parse_memory_findings(
                parsed_payload.get("retained_uncertainties")
            ),
            latest_assistant_reply=_parse_optional_text(
                parsed_payload.get("latest_assistant_reply")
            ),
            stats=resolved_stats,
        )

    return StructuredSessionMemory(
        recent_user_intents=_parse_string_tuple(
            parsed_payload.get("recent_user_intents")
        ),
        retained_facts=_parse_memory_findings(parsed_payload.get("retained_facts")),
        retained_uncertainties=_parse_memory_findings(
            parsed_payload.get("retained_uncertainties")
        ),
        latest_assistant_reply=_parse_optional_text(
            parsed_payload.get("latest_assistant_reply")
        ),
        stats=resolved_stats,
        summary_text=summary_text,
    )


def _compose_summary_text(
    *,
    recent_user_intents: tuple[str, ...],
    retained_facts: tuple[SessionMemoryFinding, ...],
    retained_uncertainties: tuple[SessionMemoryFinding, ...],
    latest_assistant_reply: str | None,
    stats: SessionMemoryStats,
) -> str:
    if stats.message_count <= 0:
        return "No messages yet."

    parts: list[str] = []
    if recent_user_intents:
        label = (
            "User intent"
            if len(recent_user_intents) == 1
            else "Recent user intents"
        )
        parts.append(f"{label}: {'; '.join(recent_user_intents)}.")

    if retained_facts:
        label = (
            "Retained grounded fact"
            if len(retained_facts) == 1
            else "Retained grounded facts"
        )
        parts.append(
            f"{label}: {'; '.join(_render_memory_finding(finding) for finding in retained_facts)}."
        )

    if retained_uncertainties:
        label = (
            "Retained uncertainty"
            if len(retained_uncertainties) == 1
            else "Retained uncertainties"
        )
        parts.append(
            (
                f"{label}: "
                f"{'; '.join(_render_memory_finding(finding) for finding in retained_uncertainties)}."
            )
        )

    if latest_assistant_reply is not None:
        parts.append(f"Latest assistant reply: {latest_assistant_reply}.")

    if not parts:
        parts.append(
            "Session has messages but no user, assistant, or grounded tool content yet."
        )

    parts.append(
        "Session size: "
        f"{stats.message_count} messages "
        f"({stats.user_message_count} user, "
        f"{stats.assistant_message_count} assistant, "
        f"{stats.tool_message_count} tool)."
    )
    return " ".join(parts)


def _collect_recent_user_intents(messages: list[ChatMessage]) -> list[str]:
    snippets: list[str] = []
    seen_snippets: set[str] = set()

    for message in reversed(messages):
        if message.role != MessageRole.USER:
            continue

        snippet = _summary_fragment(
            message.content,
            limit=SESSION_SUMMARY_INTENT_CHARACTER_LIMIT,
        )
        if snippet is None:
            continue

        normalized_snippet = snippet.casefold()
        if normalized_snippet in seen_snippets:
            continue

        snippets.append(snippet)
        seen_snippets.add(normalized_snippet)
        if len(snippets) >= SESSION_SUMMARY_INTENT_LIMIT:
            break

    snippets.reverse()
    return snippets


def _collect_retained_search_findings(
    messages: list[ChatMessage],
    *,
    include_uncertain: bool,
    limit: int,
) -> list[SessionMemoryFinding]:
    findings: list[SessionMemoryFinding] = []
    seen_findings: set[str] = set()

    for message in reversed(messages):
        if message.role is not MessageRole.TOOL:
            continue

        grounding = parse_search_tool_history(message.content)
        if grounding is None:
            continue

        query_label = _summary_fragment(
            grounding.query,
            limit=SESSION_SUMMARY_INTENT_CHARACTER_LIMIT,
        )
        source_findings = (
            grounding.uncertain_findings
            if include_uncertain
            else grounding.supported_findings
        )

        for finding in source_findings:
            finding_text = _summary_fragment(
                finding.text,
                limit=SESSION_SUMMARY_FINDING_CHARACTER_LIMIT,
            )
            if finding_text is None:
                continue

            normalized_finding = finding_text.casefold()
            if normalized_finding in seen_findings:
                continue

            seen_findings.add(normalized_finding)
            findings.append(
                SessionMemoryFinding(text=finding_text, query=query_label)
            )

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


def _render_memory_finding(finding: SessionMemoryFinding) -> str:
    if finding.query is None:
        return finding.text
    return f"[{finding.query}] {finding.text}"


def _serialize_memory_finding(finding: SessionMemoryFinding) -> dict[str, str]:
    payload = {"text": finding.text}
    if finding.query is not None:
        payload["query"] = finding.query
    return payload


def _load_persisted_summary_payload(
    persisted_summary: str,
) -> dict[str, Any] | None:
    if not persisted_summary.startswith("{"):
        return None

    try:
        parsed = json.loads(persisted_summary)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    if parsed.get("schema") != _SESSION_MEMORY_SCHEMA:
        if isinstance(parsed.get("summary_text"), str):
            return parsed
        return None

    if parsed.get("version") != _SESSION_MEMORY_VERSION:
        if isinstance(parsed.get("summary_text"), str):
            return parsed
        return None

    return parsed


def _parse_memory_stats(
    payload: object,
    *,
    fallback: SessionMemoryStats | None,
) -> SessionMemoryStats:
    if isinstance(payload, dict):
        return SessionMemoryStats(
            message_count=_parse_non_negative_int(payload.get("message_count")),
            user_message_count=_parse_non_negative_int(
                payload.get("user_message_count")
            ),
            assistant_message_count=_parse_non_negative_int(
                payload.get("assistant_message_count")
            ),
            tool_message_count=_parse_non_negative_int(
                payload.get("tool_message_count")
            ),
        )

    if fallback is not None:
        return fallback
    return SessionMemoryStats(0, 0, 0, 0)


def _parse_memory_findings(payload: object) -> tuple[SessionMemoryFinding, ...]:
    if not isinstance(payload, list):
        return ()

    findings: list[SessionMemoryFinding] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        text = _parse_optional_text(entry.get("text"))
        if text is None:
            continue
        findings.append(
            SessionMemoryFinding(
                text=text,
                query=_parse_optional_text(entry.get("query")),
            )
        )
    return tuple(findings)


def _parse_string_tuple(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, list):
        return ()
    values = [_parse_optional_text(entry) for entry in payload]
    return tuple(value for value in values if value is not None)


def _parse_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _parse_non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0
