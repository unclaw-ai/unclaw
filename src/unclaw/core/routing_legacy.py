"""Legacy deterministic structural request-routing fallback helpers.

These helpers exist only to support the bounded one-shot legacy retry after
the normal model-first turn stays text-only. Semantic routing is intentionally
disabled here; only explicit structural/technical hints may trigger a retry.
"""

from __future__ import annotations

import re

from unclaw.core.capabilities import RuntimeCapabilitySummary

_CURRENT_REQUEST_ROUTING_NOTE_PREFIX = "Current request routing hint:"
_URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_FILE_EXTENSION_HINT_PATTERN = re.compile(
    r"\.(?:txt|md|json|csv|py|js|ts|tsx|jsx|yaml|yml|toml|ini|cfg|log|"
    r"html|css|sh|sql)\b",
    flags=re.IGNORECASE,
)
_PATH_HINT_PATTERN = re.compile(
    r"(?:^|[\s`])(?:\.{1,2}/|~/|/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+/?"
)
_TERMINAL_SNIPPET_PATTERN = re.compile(r"`[^`]+`")


def _looks_like_direct_url_fetch_request(
    *,
    user_input: str,
) -> bool:
    return _URL_PATTERN.search(user_input) is not None


def _looks_like_explicit_terminal_request(
    *,
    user_input: str,
) -> bool:
    return _TERMINAL_SNIPPET_PATTERN.search(user_input) is not None


def _looks_like_local_directory_request(
    *,
    user_input: str,
) -> bool:
    if _URL_PATTERN.search(user_input) is not None:
        return False
    path_hint = _PATH_HINT_PATTERN.search(user_input)
    if path_hint is None:
        return False
    return _FILE_EXTENSION_HINT_PATTERN.search(path_hint.group(0).strip(" `")) is None


def _looks_like_standalone_file_hint(user_input: str) -> bool:
    stripped = user_input.strip().strip("`'\"")
    if not stripped or any(character.isspace() for character in stripped):
        return False
    if _URL_PATTERN.search(stripped) is not None:
        return False
    if _FILE_EXTENSION_HINT_PATTERN.search(stripped) is None:
        return False
    base, _, _ = stripped.rpartition(".")
    return bool(base) and any(character.isalnum() for character in base)


def _looks_like_local_file_request(
    *,
    user_input: str,
) -> bool:
    if _URL_PATTERN.search(user_input) is not None:
        return False
    path_hint = _PATH_HINT_PATTERN.search(user_input)
    if (
        path_hint is not None
        and _FILE_EXTENSION_HINT_PATTERN.search(path_hint.group(0).strip(" `"))
        is not None
    ):
        return True
    return _looks_like_standalone_file_hint(user_input)


def _build_request_routing_note(
    *,
    user_input: str,
    capability_summary: RuntimeCapabilitySummary,
) -> str | None:
    """Build the bounded legacy routing note used only by the retry path.

    The legacy retry is intentionally limited to obvious operational or
    local-first requests. Semantic web-intent recovery stays model-driven.
    """
    if capability_summary.model_can_call_tools is not True:
        return None

    if not user_input.strip():
        return None

    if (
        capability_summary.url_fetch_available
        and _looks_like_direct_url_fetch_request(
            user_input=user_input,
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
        )
    ):
        return (
            f"{_CURRENT_REQUEST_ROUTING_NOTE_PREFIX} "
            "this is an explicit local shell or terminal request. Call "
            "run_terminal_command with the requested command before explaining "
            "what you would do."
        )

    if _looks_like_local_directory_request(
        user_input=user_input,
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

    return None
