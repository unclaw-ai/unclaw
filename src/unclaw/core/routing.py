"""Routing-related runtime helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
import unicodedata

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.session_manager import SessionManager
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.tools.contracts import ToolCall

_CURRENT_REQUEST_ROUTING_NOTE_PREFIX = "Current request routing hint:"
_URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_DIRECTORY_SCOPE_PATTERN = re.compile(
    r"\b(?:directory|folder|dossier|repertoire|repo|repository|project tree|tree)\b"
)
_DIRECTORY_REQUEST_PATTERN = re.compile(
    r"\b(?:list|show|inspect|check|browse|display|see|contents? of|files? in|"
    r"what(?:'s| is) in|liste|montre|affiche|contenu de)\b"
)
_FILE_REQUEST_PATTERN = re.compile(
    r"\b(?:read|open|inspect|summarize|show|display|print|view|cat|lire|ouvre|"
    r"ouvrir|affiche|inspecte|resume|resumer|voir)\b"
)
_FILE_EXTENSION_HINT_PATTERN = re.compile(
    r"\.(?:txt|md|json|csv|py|js|ts|tsx|jsx|yaml|yml|toml|ini|cfg|log|"
    r"html|css|sh|sql)\b"
)
_PATH_HINT_PATTERN = re.compile(
    r"(?:^|[\s`])(?:\.{1,2}/|~/|/)?[a-z0-9_.-]+(?:/[a-z0-9_.-]+)+/?"
)
_TERMINAL_REQUEST_PATTERN = re.compile(
    r"\b(?:run|execute|launch)\b.*\b(?:terminal|shell|command|cmd|script)\b"
)
_TERMINAL_SNIPPET_PATTERN = re.compile(r"`[^`]+`")
_SYSTEM_INFO_REQUEST_PATTERN = re.compile(
    r"\b(?:operating system|os\b|hostname|locale|cpu|ram|memory|machine specs?|"
    r"hardware specs?|system info|local machine|this machine|on this machine|"
    r"local date|local time|local day|local datetime|today'?s date|current "
    r"(?:local )?(?:date|time)|what day is it(?: today)?|what date is it(?: today)?|"
    r"quelle heure locale|quelle date locale|quel jour sommes-nous|quel jour est-on|"
    r"heure locale|date du jour)\b"
)
_IDENTITY_REQUEST_PATTERN = re.compile(
    r"\b(?:who is|who are|who's|qui est|qui sont|c'?est qui|bio de|biographie de|"
    r"biography of|bio of)\b"
)
_WEB_LOOKUP_REQUEST_PATTERN = re.compile(
    r"\b(?:search(?: the)? (?:web|internet)|web search|look up|lookup|"
    r"search online|find online|recherche en ligne|cherche sur (?:le )?"
    r"(?:web|internet)|sur (?:le )?(?:web|internet)|news about|latest news|"
    r"actualites sur)\b"
)
_URL_FETCH_REQUEST_PATTERN = re.compile(
    r"\b(?:fetch|open|read|inspect|summarize|check|visit|load|show|analyze|"
    r"look at|ouvre|lire|inspecte|resume|resumer|visite)\b"
)
_FOLLOW_UP_ENTITYLESS_PATTERN = re.compile(
    r"\b(?:their|them|they|that|this|those|it|him|her|his|its|"
    r"leur|leurs|eux|elles?|ils?|sa|son|ses|bio(?:graphy)?|biographie|"
    r"profile|profil|background|parcours|resume|resumer|shorter|brief|"
    r"court|courte|more about|plus sur|full bio|complete bio)\b"
)
_EXPLICIT_ENTITY_CORRECTION_PATTERN = re.compile(
    r"^\s*(?:non|no)\b|"
    r"\b(?:je parle(?: bien)? de|je veux dire|i mean|i meant|"
    r"i am talking about|i'm talking about|talking about)\b",
    flags=re.IGNORECASE,
)
_DEEP_SEARCH_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    # Bio depth (matched against normalized/accent-stripped text)
    r"bio(?:graphie)? (?:complete|detaillee|approfondie|exhaustive|courte)|"
    r"biographie courte|"
    r"bio courte|"
    r"full bio(?:graphy)?|"
    r"complete bio(?:graphy)?|"
    r"detailed bio(?:graphy)?|"
    r"fiche (?:complete|detaillee)|"
    r"dossier (?:complet|detaille)|"
    r"resume (?:complet|detaille)|"
    r"profil (?:complet|detaille)|"
    # Research depth
    r"recherche (?:complete|approfondie|detaillee|plus complete|plus approfondie)|"
    r"recherche en detail|"
    r"cherche (?:en detail|plus en detail|plus approfondi)|"
    r"deep (?:research|search|dive)|"
    r"in[- ]?depth|"
    r"en profondeur|"
    # "Everything" signals
    r"tout ce que tu sais|"
    r"tout ce qu.il y a|"
    r"everything (?:you know|about)|"
    # File-write with research intent
    r"dans un fichier|"
    r"in a (?:text )?file|"
    r"ecris une bio|"
    r"write.*bio|"
    r"sauvegarde.*bio|"
    r"enregistre.*bio|"
    # Explicit escalation or frustration signals
    r"pourquoi.*fast|"
    r"plus (?:de|en) (?:detail|profondeur)"
    r")\b",
    flags=re.IGNORECASE,
)
_DUO_ENTITY_PATTERN = re.compile(
    r"\b([A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:et|and|&)\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)\b",
)
_MULTI_ENTITY_SEQUENCE_PATTERN = re.compile(
    r"\b(?:puis(?:\s+sur)?|ensuite(?:\s+sur)?|et\s+ensuite|et\s+puis|"
    r"then(?:\s+for)?|and\s+then(?:\s+for)?)\b",
    re.IGNORECASE,
)
_SEARCH_TOOL_NAMES_FOR_DEDUP: frozenset[str] = frozenset({"fast_web_search", "search_web"})


@dataclass(frozen=True, slots=True)
class _EntityAnchor:
    surface: str
    corrected: bool = False
    # True when the entity surface was extracted from the CURRENT user turn.
    # False when it came from a history traversal (follow-up fallback).
    # The entity guard is only applied when from_current_turn is True OR
    # the anchor is an explicit correction — never for stale history fallbacks.
    from_current_turn: bool = False


def _normalize_runtime_routing_text(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD",
        text.replace("’", "'").casefold(),
    )
    without_marks = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(without_marks.split())


def _looks_like_explicit_entity_correction(user_input: str) -> bool:
    return _EXPLICIT_ENTITY_CORRECTION_PATTERN.search(user_input) is not None


def _looks_like_follow_up_entity_request(
    *,
    user_input: str,
    normalized_user_input: str,
) -> bool:
    if _looks_like_identity_request(normalized_user_input):
        return True
    if _FOLLOW_UP_ENTITYLESS_PATTERN.search(normalized_user_input) is not None:
        return True
    return user_input.strip().endswith("?") and len(normalized_user_input.split()) <= 8


def _find_recent_entity_anchor(
    history: Sequence[ChatMessage],
) -> _EntityAnchor | None:
    from unclaw.tools.web_entity_guard import extract_user_entity_surface

    fallback_anchor: _EntityAnchor | None = None
    user_messages_seen = 0
    # When a correction utterance is found but the entity can't be extracted
    # from it (e.g. the sentence is too long: "Non, je parle bien du duo
    # YouTube francais McFly et Carlito"), carry this flag forward so the
    # next entity-bearing message is still treated as the corrected target.
    correction_context_active = False

    for message in reversed(history):
        if message.role is not MessageRole.USER:
            continue
        user_messages_seen += 1
        if user_messages_seen > 8:
            break

        message_is_correction = _looks_like_explicit_entity_correction(message.content)
        if message_is_correction:
            correction_context_active = True

        surface = extract_user_entity_surface(message.content)
        if not surface:
            continue
        normalized_message = _normalize_runtime_routing_text(message.content)
        if (
            _looks_like_follow_up_entity_request(
                user_input=message.content,
                normalized_user_input=normalized_message,
            )
            and _normalize_runtime_routing_text(surface).strip(" .!?")
            == normalized_message.strip(" .!?")
        ):
            continue

        anchor = _EntityAnchor(
            surface=surface,
            # Corrected either when this message is itself a correction, or when
            # a correction utterance appeared more recently in history (and its
            # entity could not be extracted, but it signals the same topic).
            corrected=message_is_correction or correction_context_active,
        )
        if anchor.corrected:
            return anchor
        if fallback_anchor is None:
            fallback_anchor = anchor

    return fallback_anchor


def _resolve_entity_anchor_for_turn(
    *,
    session_manager: SessionManager,
    session_id: str,
    user_input: str,
) -> _EntityAnchor | None:
    from unclaw.tools.web_entity_guard import extract_user_entity_surface

    explicit_surface = extract_user_entity_surface(user_input)
    normalized_user_input = _normalize_runtime_routing_text(user_input)
    normalized_explicit_surface = _normalize_runtime_routing_text(explicit_surface)
    explicit_surface_is_generic_follow_up = (
        explicit_surface
        and _looks_like_follow_up_entity_request(
            user_input=user_input,
            normalized_user_input=normalized_user_input,
        )
        and normalized_explicit_surface.strip(" .!?") == normalized_user_input.strip(" .!?")
    )
    if explicit_surface and not explicit_surface_is_generic_follow_up:
        # Multi-entity sequential requests (e.g. "Michou puis Cyprien puis Squeezie")
        # cannot be reduced to one enforcement surface — disable the guard so each
        # tool call keeps the entity the model naturally picks per sub-request.
        if _looks_like_multi_entity_request(user_input):
            return None
        return _EntityAnchor(
            surface=explicit_surface,
            corrected=_looks_like_explicit_entity_correction(user_input),
            from_current_turn=True,
        )

    if not _looks_like_follow_up_entity_request(
        user_input=user_input,
        normalized_user_input=normalized_user_input,
    ):
        return None

    # Follow-up: walk history to find the anchor entity.  History anchors always
    # have from_current_turn=False so the entity guard stays inactive for them
    # (only the recentering note nudges the model — no hard enforcement).
    return _find_recent_entity_anchor(session_manager.list_messages(session_id))


def _looks_like_direct_url_fetch_request(
    *,
    user_input: str,
    normalized_user_input: str,
) -> bool:
    if _URL_PATTERN.search(user_input) is None:
        return False
    if _URL_FETCH_REQUEST_PATTERN.search(normalized_user_input) is not None:
        return True
    return any(
        token in normalized_user_input
        for token in (" url ", " link ", " page ", " site ")
    )


def _looks_like_explicit_terminal_request(
    *,
    user_input: str,
    normalized_user_input: str,
) -> bool:
    if _TERMINAL_REQUEST_PATTERN.search(normalized_user_input) is not None:
        return True
    if _TERMINAL_SNIPPET_PATTERN.search(user_input) is None:
        return False
    return bool(
        re.search(
            r"\b(?:run|execute|launch|show|what does)\b",
            normalized_user_input,
        )
    )


def _looks_like_system_info_request(normalized_user_input: str) -> bool:
    return _SYSTEM_INFO_REQUEST_PATTERN.search(normalized_user_input) is not None


def _looks_like_identity_request(normalized_user_input: str) -> bool:
    return _IDENTITY_REQUEST_PATTERN.search(normalized_user_input) is not None


def _looks_like_local_directory_request(
    *,
    user_input: str,
    normalized_user_input: str,
) -> bool:
    if _URL_PATTERN.search(user_input) is not None:
        return False
    if _DIRECTORY_REQUEST_PATTERN.search(normalized_user_input) is None:
        return False
    if _DIRECTORY_SCOPE_PATTERN.search(normalized_user_input) is not None:
        return True
    if re.search(r"\b(?:files|contents?)\s+in\b", normalized_user_input) is not None:
        return True
    return (
        _PATH_HINT_PATTERN.search(normalized_user_input) is not None
        and _FILE_EXTENSION_HINT_PATTERN.search(normalized_user_input) is None
    )


def _looks_like_local_file_request(
    *,
    user_input: str,
    normalized_user_input: str,
) -> bool:
    if _URL_PATTERN.search(user_input) is not None:
        return False
    if _FILE_REQUEST_PATTERN.search(normalized_user_input) is None:
        return False
    if _FILE_EXTENSION_HINT_PATTERN.search(normalized_user_input) is not None:
        return True
    if _PATH_HINT_PATTERN.search(normalized_user_input) is None:
        return False
    return bool(re.search(r"\b(?:file|fichier)\b", normalized_user_input))


def _looks_like_explicit_web_lookup_request(normalized_user_input: str) -> bool:
    return _WEB_LOOKUP_REQUEST_PATTERN.search(normalized_user_input) is not None


def _looks_like_multi_entity_request(user_input: str) -> bool:
    """Return True when the request clearly sequences multiple entity lookups.

    Detects patterns like 'Michou, puis Cyprien, puis Squeezie' or
    'search Michou then Cyprien then Squeezie'.  When True, no single entity
    surface should be enforced across all planned tool calls.
    """
    normalized = _normalize_runtime_routing_text(user_input)
    return _MULTI_ENTITY_SEQUENCE_PATTERN.search(normalized) is not None


def _looks_like_deep_search_request(user_input: str) -> bool:
    """Return True when the user explicitly signals they want deep or complete research.

    Normalizes the input before matching so accented characters (e.g.
    'complète' → 'complete') are handled correctly.  Covers bio depth signals,
    research depth signals, file-write-from-research intents, and explicit
    escalation or frustration phrases.
    """
    normalized = _normalize_runtime_routing_text(user_input)
    return _DEEP_SEARCH_REQUEST_PATTERN.search(normalized) is not None


def _looks_like_joint_entity_request(user_input: str) -> bool:
    """Return True when the request targets a paired or duo entity like 'McFly et Carlito'.

    Detects 'X et Y' / 'X and Y' / 'X & Y' where X and Y look like proper
    names (capitalized tokens).  Requires an identity or bio signal so that
    non-person pairs like 'Python et Java' are not mistaken for duo lookups.
    Does NOT fire for sequential multi-entity requests (puis/then/ensuite).
    """
    if _looks_like_multi_entity_request(user_input):
        return False
    if _DUO_ENTITY_PATTERN.search(user_input) is None:
        return False
    normalized = _normalize_runtime_routing_text(user_input)
    return (
        _looks_like_identity_request(normalized)
        or _FOLLOW_UP_ENTITYLESS_PATTERN.search(normalized) is not None
        or _DEEP_SEARCH_REQUEST_PATTERN.search(normalized) is not None
    )


def _deduplicate_search_tool_calls(
    tool_calls: tuple[ToolCall, ...],
) -> tuple[ToolCall, ...]:
    """Drop duplicate search calls with identical (tool, query) in a single batch.

    When entity-routing goes wrong and the same query is emitted for all
    planned searches (e.g. 'Michou' three times instead of Michou/Cyprien/
    Squeezie), executing them all wastes cycles and produces misleading output.
    This guard eliminates the duplicates before execution.  Non-search tool
    calls are always kept.
    """
    seen_signatures: set[tuple[str, str]] = set()
    deduped: list[ToolCall] = []
    for tc in tool_calls:
        if tc.tool_name in _SEARCH_TOOL_NAMES_FOR_DEDUP:
            query = str(tc.arguments.get("query", "")).casefold().strip()
            sig = (tc.tool_name, query)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
        deduped.append(tc)
    return tuple(deduped)


def _build_request_routing_note(
    *,
    user_input: str,
    capability_summary: RuntimeCapabilitySummary,
) -> str | None:
    if capability_summary.model_can_call_tools is not True:
        return None

    normalized_user_input = _normalize_runtime_routing_text(user_input)
    if not normalized_user_input:
        return None

    if (
        capability_summary.url_fetch_available
        and _looks_like_direct_url_fetch_request(
            user_input=user_input,
            normalized_user_input=normalized_user_input,
        )
    ):
        return (
            f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
            "the user gave a specific public URL. Call fetch_url_text first and "
            "answer from the fetched content before guessing or asking for clarification."
        )

    if (
        capability_summary.shell_command_execution_available
        and _looks_like_explicit_terminal_request(
            user_input=user_input,
            normalized_user_input=normalized_user_input,
        )
    ):
        return (
            f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
            "this is an explicit local shell or terminal request. Call "
            "run_terminal_command with the requested command before explaining "
            "what you would do."
        )

    if (
        capability_summary.system_info_available
        and _looks_like_system_info_request(normalized_user_input)
    ):
        return (
            f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
            "this is an obvious local machine or runtime question. Call "
            "system_info now and answer from its output instead of guessing or "
            "asking whether you should check."
        )

    # Deep search signal: skip fast_web_search, route directly to search_web.
    # This check comes before the identity check so that 'fais une bio complète'
    # does not fall into the lighter fast_web_search-first identity path.
    if _looks_like_deep_search_request(user_input) and capability_summary.web_search_available:
        is_duo = _looks_like_joint_entity_request(user_input)
        is_identity = _looks_like_identity_request(normalized_user_input)
        if is_duo:
            entity_hint = (
                "with the exact duo/pair name as the user wrote it (both names as a unit). "
                "Do not split the duo into separate unrelated lookups."
            )
        elif is_identity:
            entity_hint = "with the exact entity wording from the user."
        else:
            entity_hint = "with the user's exact query terms."
        return (
            f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
            "the user explicitly asked for deep, complete, or file-output research. "
            f"Call search_web {entity_hint} "
            "search_web is the deeper grounded path — it fetches and condenses pages "
            "for richer answers. Do not stop at fast_web_search alone."
        )

    if _looks_like_identity_request(normalized_user_input):
        is_duo = _looks_like_joint_entity_request(user_input)
        if capability_summary.fast_web_search_available:
            if is_duo:
                return (
                    f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                    "this is a duo or joint entity biography lookup. Call fast_web_search "
                    "with the duo name as a unit exactly as the user wrote it (both names "
                    "together). Do not separate the duo into unrelated single-entity lookups. "
                    "If the grounding note is thin or the user wants more, call search_web "
                    "next with the same duo name — it is the deeper grounded path."
                )
            return (
                f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                "this is an identity or biography lookup. Call fast_web_search "
                "first with the exact entity wording from the user. If the result "
                "is thin or the user wants a full answer, call search_web next — "
                "it is the deeper grounded web path for complete biographies."
            )
        if capability_summary.web_search_available:
            return (
                f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                "this is an identity or biography lookup. Call search_web first "
                "with the exact entity wording from the user instead of answering "
                "from weak memory."
            )

    if _looks_like_local_directory_request(
        user_input=user_input,
        normalized_user_input=normalized_user_input,
    ):
        if capability_summary.local_directory_listing_available:
            follow_up = (
                " If the listing surfaces a relevant supported text file and the "
                "user needs its contents, read that file next."
                if capability_summary.local_file_read_available
                else ""
            )
            return (
                f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                "this is an explicit local directory inspection request. Call "
                "list_directory on the provided path or scope before asking for "
                f"clarification.{follow_up}"
            )
        if capability_summary.shell_command_execution_available:
            return (
                f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                "this is a local directory inspection request and list_directory "
                "is unavailable. Use run_terminal_command for a bounded local listing "
                "before asking for clarification."
            )

    if _looks_like_local_file_request(
        user_input=user_input,
        normalized_user_input=normalized_user_input,
    ):
        if capability_summary.local_file_read_available:
            follow_up = (
                " If the path turns out to be a directory or you need discovery first, "
                "call list_directory before clarifying."
                if capability_summary.local_directory_listing_available
                else ""
            )
            return (
                f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                "this is an explicit local file inspection request. Call "
                f"read_text_file on the provided path.{follow_up}"
            )
        if capability_summary.local_directory_listing_available:
            return (
                f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
                "this looks like a local file inspection request, but a direct "
                "file read is unavailable. Start with list_directory on the provided "
                "path or scope before asking for clarification."
            )

    if (
        capability_summary.web_search_available
        and _looks_like_explicit_web_lookup_request(normalized_user_input)
    ):
        return (
            f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
            "this explicitly asks for a web lookup or current external information. "
            "Call search_web first and use the user's exact wording in the first lookup."
        )

    return None
