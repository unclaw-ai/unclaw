"""Deterministic exact session-recall shortcut for runtime.py.

Narrow, explicit, auditable path for exact conversation-history requests.
Only handles the four clearly defined recall intents listed below.
All other queries return None so normal runtime routing is unchanged.

This is NOT general routing logic. It is a narrow, scoped correctness path
for one feature (session recall), explicitly required by the session-recall
mission spec.

Compliant with mandatory_rules.md rule 5 allowed exception:
  - Explicitly required by the session-recall mission.
  - Clearly scoped to session recall only; does not affect any other route.
  - Not the main architecture — returns None for all non-matching queries.
  - Documented honestly here and in the diff.

Supported intents
-----------------
1. NTH_USER_PROMPT    "première question", "2e prompt", "et la 3eme ?", …
2. COUNT_USER_PROMPTS "combien de prompts", "how many prompts"
3. COUNT_ALL_MESSAGES "combien de messages", "how many messages"
4. LIST_USER_PROMPTS  "liste tous mes prompts dans l'ordre"

Matching rules (explicit — see each section below for the full list)
----------------------------------------------------------------------
NTH:   ordinal word/number  AND  history context word in the same message
       OR  continuation shorthand: the whole message IS almost entirely an
       ordinal (e.g. "et la 3eme ?", "première ?").
COUNT: explicit count keywords with "message" or "prompt" context.
LIST:  explicit list-request keywords with "prompt" or "question" context.

Source of truth: ChatMemoryStore (JSONL). The model is never called on these
paths. The calling code must persist the assistant reply as usual.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

from unclaw.memory.chat_store import ChatMemoryRecord


# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

class ExactRecallKind(Enum):
    NTH_USER_PROMPT = auto()
    COUNT_USER_PROMPTS = auto()
    COUNT_ALL_MESSAGES = auto()
    LIST_USER_PROMPTS = auto()


@dataclass(frozen=True, slots=True)
class ExactRecallIntent:
    kind: ExactRecallKind
    nth: int = field(default=0)  # 1-indexed; meaningful only for NTH_USER_PROMPT


# ---------------------------------------------------------------------------
# Ordinal vocabulary — explicit, auditable, easy to extend/remove
# ---------------------------------------------------------------------------

# Word ordinals → integer  (French + English)
_ORDINAL_WORDS: dict[str, int] = {
    # French
    "premier": 1,
    "première": 1,
    "deuxième": 2,
    "second": 2,
    "seconde": 2,
    "troisième": 3,
    "quatrième": 4,
    "cinquième": 5,
    "sixième": 6,
    "septième": 7,
    "huitième": 8,
    "neuvième": 9,
    "dixième": 10,
    # English
    "first": 1,
    # "second" already covered above
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

# Numeric ordinal suffix pattern  "1er", "2e", "3ème", "4th", etc.
# Alternatives ordered longest-first to avoid partial matches (ème before eme before e).
_NUMERIC_ORDINAL_RE = re.compile(
    r"\b(\d+)\s*(?:i?ère|er|i?ème|eme|e|st|nd|rd|th)\b",
    re.IGNORECASE | re.UNICODE,
)

# Words that signal the message is about conversation/prompt history
_HISTORY_CONTEXT_WORDS: frozenset[str] = frozenset({
    "question",
    "questions",
    "prompt",
    "prompts",
    "message",
    "messages",
    "demande",
    "demandes",
})

# ---------------------------------------------------------------------------
# Continuation shorthand pattern
#
# Matches short messages whose entire content is (optional preamble) + ordinal.
# Examples that must match:   "et la 3eme ?", "première ?", "le 5e ?", "seconde ?"
# Examples that must NOT match: "le troisième jour", "il est premier de la classe"
# "second" is excluded from the ordinal alternation — too ambiguous in English.
# ---------------------------------------------------------------------------

_CONTINUATION_ORDINAL_RE = re.compile(
    r"^\s*"
    r"(?:(?:et\s+)?(?:la|le|mon|ma|the|my)\s*)?"       # optional preamble
    r"(?P<ordinal>"
    r"(?:\d+\s*(?:i?ère|er|i?ème|eme|e|st|nd|rd|th)\b)"  # numeric: 1er, 2e, 3ème
    r"|premier|première"                                 # French 1st
    r"|deuxième|seconde"                                 # French 2nd (NOT "second")
    r"|troisième|quatrième|cinquième"
    r"|sixième|septième|huitième|neuvième|dixième"
    r"|first|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    r")"
    r"\s*[?.!,;]?\s*$",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# Count keyword lists — explicit substrings, auditable, easy to extend
# ---------------------------------------------------------------------------

_COUNT_MESSAGE_SUBSTRINGS: tuple[str, ...] = (
    "combien de message",
    "tu as combien de message",
    "tu as un historique de combien de message",
    "how many message",
    "nombre de message",
    "/ message",
    "messages /",
)

_COUNT_PROMPT_SUBSTRINGS: tuple[str, ...] = (
    "combien de prompt",
    "how many prompt",
    "nombre de prompt",
    "/ prompt",
    "prompts /",
)

# ---------------------------------------------------------------------------
# List keyword list — explicit substrings
# ---------------------------------------------------------------------------

_LIST_PROMPTS_SUBSTRINGS: tuple[str, ...] = (
    "liste tous mes prompts",
    "liste de mes prompts",
    "liste mes prompts",
    "lister mes prompts",
    "tous mes prompts dans l'ordre",
    "tous mes prompts dans l\u2019ordre",
    "mes prompts dans l'ordre",
    "mes prompts dans l\u2019ordre",
    "list all my prompts",
    "list my prompts",
    "peux tu me faire la liste de mes prompts",
    "liste de toutes mes questions",
    "liste de tous mes messages",
    "tous mes messages dans l'ordre",
    "liste de mes questions",
    "liste mes questions",
)


# ---------------------------------------------------------------------------
# Matching — public API
# ---------------------------------------------------------------------------

def match_exact_recall_intent(text: str) -> ExactRecallIntent | None:
    """Return an ExactRecallIntent if *text* is a clearly scoped exact recall request.

    Returns None when the text does not match any defined recall pattern;
    the caller must fall through to normal runtime routing in that case.

    This function is pure (no side effects) and operates only on the current
    user message text — not on conversation history.

    Priority order: LIST > COUNT > NTH (most specific first to avoid false matches).
    """
    lowered = text.lower().strip()

    # 1. LIST check — most specific, checked first
    for sub in _LIST_PROMPTS_SUBSTRINGS:
        if sub in lowered:
            return ExactRecallIntent(kind=ExactRecallKind.LIST_USER_PROMPTS)

    # 2. COUNT checks
    has_count_message = any(sub in lowered for sub in _COUNT_MESSAGE_SUBSTRINGS)
    has_count_prompt = any(sub in lowered for sub in _COUNT_PROMPT_SUBSTRINGS)

    if has_count_message:
        return ExactRecallIntent(kind=ExactRecallKind.COUNT_ALL_MESSAGES)
    if has_count_prompt:
        return ExactRecallIntent(kind=ExactRecallKind.COUNT_USER_PROMPTS)

    # 3. NTH check — strategy A: ordinal word/number + history context word
    nth = _find_ordinal_in_text(lowered)
    if nth is not None and _has_history_context_word(lowered):
        return ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=nth)

    # 3. NTH check — strategy B: continuation shorthand
    #    The whole message IS almost entirely an ordinal (e.g. "et la 3eme ?")
    continuation_match = _CONTINUATION_ORDINAL_RE.match(lowered)
    if continuation_match:
        ordinal_token = (continuation_match.group("ordinal") or "").strip().lower()
        nth_cont = _parse_ordinal_token(ordinal_token)
        if nth_cont is not None and nth_cont > 0:
            return ExactRecallIntent(kind=ExactRecallKind.NTH_USER_PROMPT, nth=nth_cont)

    return None


# ---------------------------------------------------------------------------
# Reply building — pure function
#
# Shared between the runtime shortcut path and tests so both paths cannot
# silently drift in formatting or counting logic.
# ---------------------------------------------------------------------------

def build_exact_recall_reply(
    records: list[ChatMemoryRecord],
    intent: ExactRecallIntent,
) -> str:
    """Build a deterministic plain-text reply for an exact recall intent.

    Source of truth: the persisted ChatMemoryStore records passed in.
    The model is never called on this path.
    """
    user_records = [r for r in records if r.role == "user"]
    total_messages = len(records)
    user_count = len(user_records)
    assistant_count = sum(1 for r in records if r.role == "assistant")
    tool_count = sum(1 for r in records if r.role == "tool")

    if intent.kind is ExactRecallKind.NTH_USER_PROMPT:
        n = intent.nth
        if n < 1 or n > user_count:
            return (
                f"There are only {user_count} user message(s) in this session "
                f"(you requested #{n})."
            )
        record = user_records[n - 1]
        return (
            f"User message #{n}:\n"
            f'"{record.content}"'
        )

    if intent.kind is ExactRecallKind.COUNT_USER_PROMPTS:
        return (
            f"You have sent {user_count} user prompt(s) in this session "
            f"({total_messages} messages total)."
        )

    if intent.kind is ExactRecallKind.COUNT_ALL_MESSAGES:
        return (
            f"Session history — {total_messages} messages total:\n"
            f"  \u2022 User (prompts): {user_count}\n"
            f"  \u2022 Assistant: {assistant_count}\n"
            f"  \u2022 Tool results: {tool_count}"
        )

    if intent.kind is ExactRecallKind.LIST_USER_PROMPTS:
        if not user_records:
            return "No user prompts in this session yet."
        lines = [f"Your {user_count} user prompt(s) in chronological order:"]
        for i, r in enumerate(user_records, start=1):
            lines.append(f"{i}. {r.content}")
        return "\n".join(lines)

    return ""  # unreachable — all enum values handled above


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_ordinal_in_text(lowered: str) -> int | None:
    """Return the first ordinal number found in *lowered* text, or None.

    Checks word ordinals first (sorted longest→shortest to avoid partial
    matches), then numeric suffix ordinals.
    """
    for word, n in sorted(_ORDINAL_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, lowered, re.IGNORECASE | re.UNICODE):
            return n

    match = _NUMERIC_ORDINAL_RE.search(lowered)
    if match:
        return int(match.group(1))

    return None


def _has_history_context_word(lowered: str) -> bool:
    """Return True if *lowered* contains at least one history context word."""
    words = re.findall(r"\b\w+\b", lowered, re.UNICODE)
    return any(w in _HISTORY_CONTEXT_WORDS for w in words)


def _parse_ordinal_token(token: str) -> int | None:
    """Parse a single ordinal token (word or numeric suffix) to an integer."""
    n = _ORDINAL_WORDS.get(token)
    if n is not None:
        return n
    m = re.match(
        r"^(\d+)\s*(?:i?ère|er|i?ème|eme|e|st|nd|rd|th)?$",
        token,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None


__all__ = [
    "ExactRecallIntent",
    "ExactRecallKind",
    "build_exact_recall_reply",
    "match_exact_recall_intent",
]
