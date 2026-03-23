"""File-backed research workspace for the search_web pipeline.

Persists Layer A-B-C research artifacts under:
  <base_dir>/<YYYYMMDD_HHMMSS>_<query_slug>/
    meta.json              — query, budget summary, source count
    source_<N>_raw.txt     — cleaned source text (bounded by budget)
    source_<N>_note.txt    — per-source condensed note
    merged_note.txt        — final merged research note

All writes are best-effort: any IO failure is caught silently so the
tool never breaks because of persistence errors.

This module is intentionally dumb — it stores, it never reasons.
Intelligence lives in the model-driven pipeline layers.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unclaw.tools.web_research import ResearchWorkspace

# Maximum characters kept for the query slug portion of a workspace ID.
_MAX_SLUG_CHARS = 40

# Default cap on the number of search workspace directories under base_dir.
_MAX_WORKSPACE_ENTRIES = 100


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert a query string into a safe, human-readable filesystem slug."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w]+", "_", ascii_text.lower()).strip("_")
    return slug[:_MAX_SLUG_CHARS] if slug else "search"


def _make_workspace_id(query: str) -> str:
    """Generate a timestamped, human-readable workspace identifier."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slugify(query)
    return f"{timestamp}_{slug}"


# ---------------------------------------------------------------------------
# Public reference type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchWorkspaceRef:
    """Opaque reference to a persisted search workspace on disk.

    Included in the search_web payload so callers (follow-up queries,
    document writers, deep-search) can locate and reuse the artifacts.
    """

    workspace_id: str
    workspace_dir: str  # Absolute path as a string for easy serialisation


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


def create_workspace_dir(base_dir: Path, query: str) -> tuple[Path, str]:
    """Create a new timestamped workspace directory under *base_dir*.

    Returns (workspace_path, workspace_id).
    Raises OSError if the directory cannot be created.
    """
    workspace_id = _make_workspace_id(query)
    workspace_path = base_dir / workspace_id
    workspace_path.mkdir(parents=True, exist_ok=True)
    return workspace_path, workspace_id


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def persist_research_workspace(
    *,
    workspace_dir: Path,
    workspace: ResearchWorkspace,
    query: str,
    executed_queries: tuple[str, ...] = (),
) -> None:
    """Write all research pipeline artifacts to *workspace_dir*.

    Each file is written independently so a single IO error does not
    prevent the remaining files from being persisted.

    Files written:
    - meta.json                   — query, budget, source count, mode
    - source_<N>_raw.txt          — cleaned source text (one per source)
    - source_<N>_note.txt         — per-source condensed note
    - merged_note.txt             — final merged research note
    """
    budget = workspace.budget
    model_driven = any(
        note.model_generated for note in workspace.source_notes
    ) or (
        workspace.merged_note is not None
        and workspace.merged_note.model_generated
    )

    # --- meta.json ---
    try:
        meta: dict = {
            "query": query,
            "source_count": workspace.source_count,
            "model_driven": model_driven,
            "executed_staged_queries": list(executed_queries),
            "budget": {
                "max_sources": budget.max_sources,
                "max_source_chars": budget.max_source_chars,
                "max_source_note_chars": budget.max_source_note_chars,
                "max_merged_note_chars": budget.max_merged_note_chars,
            },
        }
        (workspace_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    # --- per-source raw text ---
    for index, artifact in enumerate(workspace.source_artifacts, start=1):
        try:
            raw_content = (
                f"URL: {artifact.url}\n"
                f"Title: {artifact.title}\n"
                f"Truncated: {artifact.truncated}\n"
                f"\n"
                f"{artifact.cleaned_text}"
            )
            (workspace_dir / f"source_{index:02d}_raw.txt").write_text(
                raw_content,
                encoding="utf-8",
            )
        except OSError:
            pass

    # --- per-source condensed notes ---
    for index, note in enumerate(workspace.source_notes, start=1):
        try:
            note_content = (
                f"URL: {note.url}\n"
                f"Title: {note.title}\n"
                f"Model-generated: {note.model_generated}\n"
                f"\n"
                f"{note.condensed_text}"
            )
            (workspace_dir / f"source_{index:02d}_note.txt").write_text(
                note_content,
                encoding="utf-8",
            )
        except OSError:
            pass

    # --- merged note ---
    if workspace.merged_note is not None:
        try:
            merged_content = (
                f"Query: {query}\n"
                f"Source count: {workspace.merged_note.source_count}\n"
                f"Model-generated: {workspace.merged_note.model_generated}\n"
                f"\n"
                f"{workspace.merged_note.text}"
            )
            (workspace_dir / "merged_note.txt").write_text(
                merged_content,
                encoding="utf-8",
            )
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Lifecycle management
# ---------------------------------------------------------------------------


def prune_old_workspaces(
    base_dir: Path,
    *,
    max_entries: int = _MAX_WORKSPACE_ENTRIES,
) -> int:
    """Remove oldest workspace directories when the count exceeds *max_entries*.

    Directories are sorted by modification time; the oldest are removed first.
    Returns the number of directories removed (0 if none or base_dir missing).
    """
    if not base_dir.exists():
        return 0

    try:
        entries = [p for p in base_dir.iterdir() if p.is_dir()]
    except OSError:
        return 0

    if len(entries) <= max_entries:
        return 0

    # Oldest-first by mtime so we keep the most recent entries.
    try:
        entries.sort(key=lambda p: p.stat().st_mtime)
    except OSError:
        return 0

    to_remove = entries[: len(entries) - max_entries]
    removed = 0
    for entry in to_remove:
        try:
            shutil.rmtree(entry)
            removed += 1
        except OSError:
            pass

    return removed


__all__ = [
    "SearchWorkspaceRef",
    "create_workspace_dir",
    "persist_research_workspace",
    "prune_old_workspaces",
]
