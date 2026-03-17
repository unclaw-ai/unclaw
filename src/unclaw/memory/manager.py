"""Session memory helpers built on top of the session manager."""

from __future__ import annotations

from dataclasses import dataclass

from unclaw.constants import (
    DEFAULT_SESSION_MEMORY_SNIPPET_LIMIT,
    SESSION_MEMORY_SNIPPET_CHARACTER_LIMIT,
)
from unclaw.core.session_manager import SessionManager, SessionManagerError
from unclaw.schemas.chat import ChatMessage, MessageRole
from unclaw.schemas.session import SessionRecord

from unclaw.memory.summarizer import (
    SessionMemoryStats,
    StructuredSessionMemory,
    build_structured_session_memory,
    parse_persisted_session_memory,
    render_session_memory_summary,
    serialize_structured_session_memory,
)


@dataclass(frozen=True, slots=True)
class SessionMemoryState:
    """Compact view of the current session memory state."""

    session_id: str
    title: str
    updated_at: str
    message_count: int
    user_message_count: int
    assistant_message_count: int
    summary: StructuredSessionMemory
    recent_snippets: tuple[str, ...]

    @property
    def summary_text(self) -> str:
        """Rendered session summary preserved for user-facing output."""
        return render_session_memory_summary(self.summary)


@dataclass(slots=True)
class MemoryManager:
    """Manage small session summaries and recent snippets."""

    session_manager: SessionManager
    recent_snippet_limit: int = DEFAULT_SESSION_MEMORY_SNIPPET_LIMIT

    def build_context_note(self, session_id: str | None = None) -> str | None:
        """Build one bounded note that can be injected into model context."""

        state = self.get_session_state(
            session_id,
            recent_limit=0,
        )
        if state.message_count <= 1:
            return None
        if state.summary_text == "No messages yet.":
            return None

        return "\n".join(
            (
                "Persisted session memory (deterministic summary):",
                state.summary_text,
                (
                    "Use this as prior conversation context. Prefer the more "
                    "specific messages below when they overlap, and prefer current "
                    "tool output for fresh external facts."
                ),
            )
        )

    def build_or_refresh_session_summary(self, session_id: str | None = None) -> str:
        """Rebuild the stored summary text for one session."""

        session = self._resolve_session(session_id)
        messages = self.session_manager.list_messages(session.id)
        return self._store_summary(session.id, messages).summary_text

    def get_session_summary(self, session_id: str | None = None) -> str:
        """Return the current stored summary, generating it when missing."""

        session = self._resolve_session(session_id)
        messages = self.session_manager.list_messages(session.id)
        summary = self._load_summary(session.id, messages)
        if summary is not None:
            return summary.summary_text

        return self._store_summary(session.id, messages).summary_text

    def get_session_state(
        self,
        session_id: str | None = None,
        *,
        recent_limit: int | None = None,
    ) -> SessionMemoryState:
        """Return a compact view of summary text, counts, and recent snippets."""

        session = self._resolve_session(session_id)
        messages = self.session_manager.list_messages(session.id)

        summary = self._load_summary(session.id, messages)
        if summary is None:
            summary = self._store_summary(session.id, messages)

        return SessionMemoryState(
            session_id=session.id,
            title=session.title,
            updated_at=session.updated_at,
            message_count=len(messages),
            user_message_count=sum(
                1 for message in messages if message.role == MessageRole.USER
            ),
            assistant_message_count=sum(
                1 for message in messages if message.role == MessageRole.ASSISTANT
            ),
            summary=summary,
            recent_snippets=self.list_recent_snippets(
                session.id,
                limit=(
                    self.recent_snippet_limit
                    if recent_limit is None
                    else recent_limit
                ),
                messages=messages,
            ),
        )

    def list_recent_snippets(
        self,
        session_id: str | None = None,
        *,
        limit: int | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> tuple[str, ...]:
        """Return short snippets from the latest messages in a session."""

        session = self._resolve_session(session_id)
        resolved_limit = self.recent_snippet_limit if limit is None else max(0, limit)
        if resolved_limit == 0:
            return ()

        ordered_messages = messages
        if ordered_messages is None:
            ordered_messages = self.session_manager.list_messages(session.id)

        snippets: list[str] = []
        for message in reversed(ordered_messages):
            snippet = self._format_message_snippet(message)
            if snippet is None:
                continue

            snippets.append(snippet)
            if len(snippets) >= resolved_limit:
                break

        snippets.reverse()
        return tuple(snippets)

    def _resolve_session(self, session_id: str | None) -> SessionRecord:
        if session_id is None:
            return self.session_manager.ensure_current_session()

        session = self.session_manager.load_session(session_id)
        if session is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")
        return session

    def _load_summary(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> StructuredSessionMemory | None:
        raw_summary = self.session_manager.session_repository.get_summary_text(session_id)
        if raw_summary is None:
            return None

        return parse_persisted_session_memory(
            raw_summary,
            fallback_stats=SessionMemoryStats.from_messages(messages),
        )

    def _store_summary(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> StructuredSessionMemory:
        summary = build_structured_session_memory(messages)
        stored_summary = self.session_manager.session_repository.update_summary_text(
            session_id,
            serialize_structured_session_memory(summary),
        )
        if stored_summary is None:
            raise SessionManagerError(f"Session '{session_id}' was not found.")

        parsed_summary = parse_persisted_session_memory(
            stored_summary,
            fallback_stats=summary.stats,
        )
        return parsed_summary or summary

    def _format_message_snippet(self, message: ChatMessage) -> str | None:
        normalized_content = " ".join(message.content.split()).strip()
        if not normalized_content:
            return None

        if len(normalized_content) > SESSION_MEMORY_SNIPPET_CHARACTER_LIMIT:
            normalized_content = (
                f"{normalized_content[:SESSION_MEMORY_SNIPPET_CHARACTER_LIMIT - 3].rstrip(' ,;:.')}..."
            )
        return f"- {message.role.value}: {normalized_content}"
