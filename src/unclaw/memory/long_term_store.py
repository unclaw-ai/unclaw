"""Persistent cross-session user memory store backed by a local SQLite database.

Stored under: data_dir / "memory" / "long_term.db"

Separate from the session database (unclaw.db) and from the per-session JSONL
chat mirrors. This store is the LONG-TERM layer of the three-layer memory
architecture — the model retrieves from it explicitly via tools, it is never
silently injected into context.

Design rules:
- stdlib only (sqlite3). No embeddings, no vector DB, no external service.
- Thread-safe for single-process use: writes are serialised by a global lock;
  reads open a fresh connection each time (no thread-affinity issue).
- Path must be derived from settings.paths.data_dir — never hardcoded.
- Conservative by default: the model stores only when explicitly asked.
- Fully auditable: plain text content in a single SQLite file.

Schema
------
id               TEXT PRIMARY KEY  — UUID
key              TEXT NOT NULL     — short title / memory name
value            TEXT NOT NULL     — full content
category         TEXT DEFAULT ''   — optional category (e.g. "preference")
tags             TEXT DEFAULT ''   — comma-separated tags for search
source_session_id TEXT DEFAULT ''  — session where the memory was created
source_seq       INTEGER DEFAULT 0 — message sequence within that session
confidence       REAL DEFAULT 1.0  — future use; default 1.0 = explicit store
created_at       TEXT NOT NULL     — ISO-8601 UTC
updated_at       TEXT NOT NULL     — ISO-8601 UTC
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_TOKEN_SPLIT_RE = re.compile(r"[\s\W]+")
_MIN_TOKEN_LEN = 3


def _fold_for_search(text: str) -> str:
    """Casefold and strip Unicode combining characters (accent folding).

    Language-agnostic: uses NFD decomposition to detach diacritics, then
    drops all Unicode category Mn (Non-spacing Mark) code points.
    Examples: 'prénom' → 'prenom', 'Ñoño' → 'nono'.
    Purpose: allow normalized token matching across accent variants without
    any language-specific synonym table.
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn").casefold()


def _extract_search_tokens(query: str) -> list[str]:
    """Split a natural-language query into individual search tokens.

    Language-agnostic: splits on whitespace and non-word characters,
    keeps tokens with at least _MIN_TOKEN_LEN characters.
    No stop-word list — length filtering removes most noise (single chars,
    two-letter particles) without any hardcoded language-specific table.
    """
    raw = _TOKEN_SPLIT_RE.split(query.lower())
    return [t for t in raw if len(t) >= _MIN_TOKEN_LEN]


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS long_term_memories (
    id                TEXT PRIMARY KEY,
    key               TEXT NOT NULL,
    value             TEXT NOT NULL,
    category          TEXT DEFAULT '',
    tags              TEXT DEFAULT '',
    source_session_id TEXT DEFAULT '',
    source_seq        INTEGER DEFAULT 0,
    confidence        REAL DEFAULT 1.0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
)
"""
_CREATE_KEY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_key ON long_term_memories(key)"
)
_CREATE_CATEGORY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memories(category)"
)


@dataclass(frozen=True, slots=True)
class LongTermMemoryRecord:
    """One row from the long_term_memories table."""

    id: str
    key: str
    value: str
    category: str
    tags: str
    source_session_id: str
    source_seq: int
    confidence: float
    created_at: str
    updated_at: str


class LongTermStore:
    """Local SQLite store for persistent cross-session user memories.

    Path: data_dir / "memory" / "long_term.db" (caller supplies the Path).

    Thread-safety:
    - Writes: serialised by a single threading.Lock.
    - Reads: open a fresh connection per call; no lock required.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._write_lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store(
        self,
        *,
        key: str,
        value: str,
        category: str = "",
        tags: str = "",
        source_session_id: str = "",
        source_seq: int = 0,
        confidence: float = 1.0,
    ) -> str:
        """Insert a new memory record. Returns the generated UUID id."""
        mem_id = str(uuid4())
        now = _utc_now()
        with self._write_lock:
            conn = self._open()
            try:
                conn.execute(
                    """
                    INSERT INTO long_term_memories
                        (id, key, value, category, tags, source_session_id,
                         source_seq, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mem_id,
                        key,
                        value,
                        category,
                        tags,
                        source_session_id,
                        source_seq,
                        confidence,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return mem_id

    def upsert(
        self,
        *,
        key: str,
        value: str,
        category: str = "",
        tags: str = "",
        source_session_id: str = "",
        source_seq: int = 0,
        confidence: float = 1.0,
    ) -> tuple[str, bool]:
        """Insert or update a memory record matched by (category, key).

        If a record with the same (category, key) already exists, its value,
        tags, updated_at, source_session_id, and source_seq are updated and
        no duplicate is created. If multiple matches exist the most-recently-
        updated one is updated; stale duplicates are left unchanged.

        If no match exists a new record is inserted (same as store()).

        Returns (id, created) where created=True means a new row was inserted,
        False means an existing row was updated.

        Use this method (not store()) for stable facts that may be corrected:
        name, GPU, preferences. This prevents stale duplicates from surviving
        after a user correction.
        """
        with self._write_lock:
            conn = self._open()
            try:
                if category:
                    row = conn.execute(
                        """
                        SELECT id FROM long_term_memories
                        WHERE category = ? AND key = ?
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (category, key),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT id FROM long_term_memories
                        WHERE key = ?
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (key,),
                    ).fetchone()
                now = _utc_now()
                if row:
                    mem_id = str(row["id"])
                    conn.execute(
                        """
                        UPDATE long_term_memories
                        SET value = ?, tags = ?, updated_at = ?,
                            source_session_id = ?, source_seq = ?
                        WHERE id = ?
                        """,
                        (value, tags, now, source_session_id, source_seq, mem_id),
                    )
                    conn.commit()
                    return mem_id, False
                else:
                    mem_id = str(uuid4())
                    conn.execute(
                        """
                        INSERT INTO long_term_memories
                            (id, key, value, category, tags, source_session_id,
                             source_seq, confidence, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            mem_id,
                            key,
                            value,
                            category,
                            tags,
                            source_session_id,
                            source_seq,
                            confidence,
                            now,
                            now,
                        ),
                    )
                    conn.commit()
                    return mem_id, True
            finally:
                conn.close()

    def forget(self, mem_id: str) -> bool:
        """Delete a memory by id. Returns True if deleted, False if not found."""
        with self._write_lock:
            conn = self._open()
            try:
                cursor = conn.execute(
                    "DELETE FROM long_term_memories WHERE id = ?",
                    (mem_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Read operations (lock-free — fresh connection per call)
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        query: str,
        category: str = "",
        limit: int = 20,
    ) -> list[LongTermMemoryRecord]:
        """Return memories matching query across key, value, tags, and category.

        Four-pass strategy (each pass only runs if under the limit):
        1. Full-phrase LIKE match on key, value, tags, and category fields.
        2. Per-token LIKE match using individual words extracted from the query
           (words >= 3 chars, language-agnostic, no stop-word list).
        3. Normalized (accent-folded, casefolded) token match in Python.
           Handles accent variants: stored 'prénom' found by query 'prenom'.
        4. Category fallback: if category is specified and results are still
           empty after passes 1-3, return the most-recent records in that
           category. Ensures any identity/hardware/etc. category always
           returns *something* when the user's query word doesn't lexically
           match the stored key/value/tags.
        Results across passes are de-duplicated, newest-first.
        Optionally restrict to an exact category value.
        """
        tokens = _extract_search_tokens(query)
        conn = self._open()
        try:
            seen_ids: set[str] = set()
            results: list[LongTermMemoryRecord] = []

            # Pass 1: full-phrase match (key, value, tags, category).
            for row in self._query_rows(conn, f"%{query}%", category, limit):
                record = _row_to_record(row)
                seen_ids.add(record.id)
                results.append(record)

            # Pass 2: per-token fallback when tokens differ from full phrase
            # or when the first pass left room under the limit.
            if tokens and len(results) < limit:
                for token in tokens:
                    if len(results) >= limit:
                        break
                    remaining = limit - len(results)
                    for row in self._query_rows(conn, f"%{token}%", category, remaining):
                        record = _row_to_record(row)
                        if record.id not in seen_ids:
                            seen_ids.add(record.id)
                            results.append(record)

            # Pass 3: normalized (accent-folded) Python-level token match.
            # Handles cases like stored tag 'prénom' matched by query 'prenom'.
            if len(results) < limit:
                norm_tokens = [
                    t
                    for t in _TOKEN_SPLIT_RE.split(_fold_for_search(query))
                    if len(t) >= _MIN_TOKEN_LEN
                ]
                if norm_tokens:
                    candidates = self._load_candidates(conn, category, limit * 5)
                    for row in candidates:
                        if len(results) >= limit:
                            break
                        record = _row_to_record(row)
                        if record.id in seen_ids:
                            continue
                        norm_text = _fold_for_search(
                            f"{record.key} {record.value} {record.tags} {record.category}"
                        )
                        if any(nt in norm_text for nt in norm_tokens):
                            seen_ids.add(record.id)
                            results.append(record)

            # Pass 4: category fallback — when a category is specified but the
            # query returned nothing at all after passes 1-3, return the most-
            # recent records in that category. This is the safety net for
            # complete lexical mismatches (e.g. French 'prénom' against
            # English-keyed identity records with no accent tag).
            if category and not results:
                for row in self._query_rows(conn, "%", category, limit):
                    record = _row_to_record(row)
                    if record.id not in seen_ids:
                        seen_ids.add(record.id)
                        results.append(record)
                        if len(results) >= limit:
                            break

        finally:
            conn.close()
        return results[:limit]

    def _query_rows(
        self,
        conn: sqlite3.Connection,
        pattern: str,
        category: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Execute a LIKE search across key, value, tags, and category fields."""
        if category:
            return conn.execute(
                """
                SELECT id, key, value, category, tags, source_session_id,
                       source_seq, confidence, created_at, updated_at
                FROM long_term_memories
                WHERE (key LIKE ? OR value LIKE ? OR tags LIKE ? OR category LIKE ?)
                  AND category = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, pattern, category, limit),
            ).fetchall()
        return conn.execute(
            """
            SELECT id, key, value, category, tags, source_session_id,
                   source_seq, confidence, created_at, updated_at
            FROM long_term_memories
            WHERE key LIKE ? OR value LIKE ? OR tags LIKE ? OR category LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, limit),
        ).fetchall()

    def _load_candidates(
        self,
        conn: sqlite3.Connection,
        category: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Load up to limit recent records for Python-level matching (Pass 3).

        Scoped to category when given; otherwise loads globally.
        Used only for normalized (accent-folded) search — limit * 5 cap in caller
        keeps this bounded for a personal memory store.
        """
        if category:
            return conn.execute(
                """
                SELECT id, key, value, category, tags, source_session_id,
                       source_seq, confidence, created_at, updated_at
                FROM long_term_memories
                WHERE category = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (category, limit),
            ).fetchall()
        return conn.execute(
            """
            SELECT id, key, value, category, tags, source_session_id,
                   source_seq, confidence, created_at, updated_at
            FROM long_term_memories
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def list_all(
        self,
        *,
        category: str = "",
        limit: int = 50,
    ) -> list[LongTermMemoryRecord]:
        """Return all memory records, optionally filtered by category.

        Results ordered newest-first.
        """
        conn = self._open()
        try:
            if category:
                rows = conn.execute(
                    """
                    SELECT id, key, value, category, tags, source_session_id,
                           source_seq, confidence, created_at, updated_at
                    FROM long_term_memories
                    WHERE category = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, key, value, category, tags, source_session_id,
                           source_seq, confidence, created_at, updated_at
                    FROM long_term_memories
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        finally:
            conn.close()
        return [_row_to_record(row) for row in rows]

    def get_by_id(self, mem_id: str) -> LongTermMemoryRecord | None:
        """Retrieve one memory by its UUID id, or None."""
        conn = self._open()
        try:
            row = conn.execute(
                """
                SELECT id, key, value, category, tags, source_session_id,
                       source_seq, confidence, created_at, updated_at
                FROM long_term_memories
                WHERE id = ?
                """,
                (mem_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _row_to_record(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        try:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_KEY_INDEX_SQL)
            conn.execute(_CREATE_CATEGORY_INDEX_SQL)
            conn.commit()
        finally:
            conn.close()

    def to_json(self, record: LongTermMemoryRecord) -> str:
        """Serialise a record to a compact JSON string (for tool output)."""
        return json.dumps(
            {
                "id": record.id,
                "key": record.key,
                "value": record.value,
                "category": record.category,
                "tags": record.tags,
                "source_session_id": record.source_session_id,
                "confidence": record.confidence,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            },
            ensure_ascii=False,
        )


def _row_to_record(row: sqlite3.Row) -> LongTermMemoryRecord:
    return LongTermMemoryRecord(
        id=str(row["id"]),
        key=str(row["key"]),
        value=str(row["value"]),
        category=str(row["category"] or ""),
        tags=str(row["tags"] or ""),
        source_session_id=str(row["source_session_id"] or ""),
        source_seq=int(row["source_seq"] or 0),
        confidence=float(row["confidence"] or 1.0),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


__all__ = [
    "LongTermMemoryRecord",
    "LongTermStore",
    "_extract_search_tokens",
    "_fold_for_search",
]
