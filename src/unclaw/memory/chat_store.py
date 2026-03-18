"""Thread-safe append-only JSONL chat-memory store for persistent conversation recall.

One JSONL file per session under base_dir.
Layout: {base_dir}/{session_id}.jsonl
Each line: JSON object with keys "role", "content", "ts".

Thread-safety model:
- Writes: one threading.Lock per session; serialises concurrent appends.
- Reads: stateless file.read_text() — safe to call from any thread without a lock.

This is phase-1 of the 3-memory architecture: per-session persistent chat history,
read-only from the recall tool, decoupled from the thread-bound SQLite connection.
Design requirements: local, deterministic, auditable, thread-safe, no dependencies.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ChatMemoryRecord:
    """One persisted message from the chat memory store."""

    seq: int          # 1-indexed line position in the JSONL file
    role: str         # "user", "assistant", or "tool"
    content: str
    created_at: str   # ISO-8601 UTC timestamp as stored by the runtime


class ChatMemoryStore:
    """Append-only JSONL store for per-session chat-turn history.

    One JSONL file per session: {base_dir}/{session_id}.jsonl
    The directory and files are created lazily on first write.

    Thread-safety:
      - append_message: protected by a per-session Lock; creates the Lock
        on first use under a global Lock so no two threads race on creation.
      - read_messages: opens a fresh file handle every time; holds no lock;
        safe to call from any thread including tool execution worker threads.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._global_lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}

    def _get_session_lock(self, session_id: str) -> threading.Lock:
        with self._global_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    def _session_path(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.jsonl"

    def append_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        created_at: str,
    ) -> None:
        """Append one message entry to the session's JSONL file.

        Creates the base directory and file on first write.
        Protected by a per-session lock for concurrent-writer safety.
        """
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._session_path(session_id)
        entry = json.dumps(
            {"role": role, "content": content, "ts": created_at},
            ensure_ascii=False,
        )
        lock = self._get_session_lock(session_id)
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(entry + "\n")

    def read_messages(self, session_id: str) -> list[ChatMemoryRecord]:
        """Return all persisted messages for a session in chronological order.

        Opens a fresh file handle on every call.
        Holds no lock — safe to call from any thread.
        Returns an empty list when no history file exists yet.
        Malformed or empty JSONL lines are silently skipped.
        """
        path = self._session_path(session_id)
        if not path.exists():
            return []

        raw_text = path.read_text(encoding="utf-8")
        records: list[ChatMemoryRecord] = []
        for seq, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                records.append(
                    ChatMemoryRecord(
                        seq=seq,
                        role=str(data.get("role", "")),
                        content=str(data.get("content", "")),
                        created_at=str(data.get("ts", "")),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return records


__all__ = ["ChatMemoryRecord", "ChatMemoryStore"]
