"""Skill-owned read-only local text search tool."""

from __future__ import annotations

import os
from pathlib import Path

from skills.local_text_search.contracts import (
    LocalTextMatchPayload,
    LocalTextSearchPayload,
)
from unclaw.tools.contracts import (
    ToolArgumentSpec,
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry

# Output bounds
_DEFAULT_MAX_RESULTS = 20
_MIN_MAX_RESULTS = 1
_MAX_MAX_RESULTS = 100
_DEFAULT_CONTEXT_CHARS = 200
_MIN_CONTEXT_CHARS = 20
_MAX_CONTEXT_CHARS = 500

# File size cap — skip files larger than this to stay bounded
_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB

# How many bytes to probe for null-byte binary detection
_BINARY_PROBE_BYTES = 8192

# Default file extensions treated as searchable text/code/config
_DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".env",
        ".rst",
        ".html",
        ".js",
        ".ts",
        ".css",
        ".sh",
        ".bash",
        ".zsh",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".tf",
        ".xml",
        ".csv",
    }
)

# Directory names to skip unconditionally during recursive walk
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        ".env",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
    }
)

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

SEARCH_LOCAL_TEXT_DEFINITION = ToolDefinition(
    name="search_local_text",
    description=(
        "Search text files recursively in a local directory for a query string. "
        "Returns bounded matches with file path, line number, and surrounding snippet."
    ),
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={
        "query": ToolArgumentSpec(
            description="Non-empty search string to find in local files.",
        ),
        "root": ToolArgumentSpec(
            description=(
                "Directory to search in. Defaults to '.' (current working directory)."
            ),
        ),
        "extensions": ToolArgumentSpec(
            description=(
                "Optional list of file extensions to include, e.g. [\".py\", \".md\"]. "
                "Defaults to a standard set of text/code/config extensions."
            ),
            value_type="array",
        ),
        "max_results": ToolArgumentSpec(
            description=(
                f"Maximum number of matches to return. "
                f"Clamped to {_MIN_MAX_RESULTS}–{_MAX_MAX_RESULTS}. "
                f"Defaults to {_DEFAULT_MAX_RESULTS}."
            ),
            value_type="integer",
        ),
        "context_chars": ToolArgumentSpec(
            description=(
                f"Characters of context to include around each match in the snippet. "
                f"Clamped to {_MIN_CONTEXT_CHARS}–{_MAX_CONTEXT_CHARS}. "
                f"Defaults to {_DEFAULT_CONTEXT_CHARS}."
            ),
            value_type="integer",
        ),
    },
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_root(raw: object) -> Path:
    """Resolve and validate a root path argument. Returns an absolute Path."""
    p = Path(str(raw) if raw else ".").resolve()
    if not p.exists():
        raise ValueError(f"Search root does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Search root is not a directory: {p}")
    return p


def _is_binary(file_path: Path) -> bool:
    """Return True if the file appears to be binary (contains a null byte in the first chunk)."""
    try:
        with open(file_path, "rb") as fh:
            chunk = fh.read(_BINARY_PROBE_BYTES)
        return b"\x00" in chunk
    except OSError:
        return True


def _normalize_extensions(raw: object) -> frozenset[str] | None:
    """Normalize the extensions argument into a frozenset. Returns None to use defaults."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    result: list[str] = []
    for item in raw:
        s = str(item).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        result.append(s)
    return frozenset(result) if result else None


def _build_snippet(line: str, query: str, context_chars: int) -> str:
    """Return a compact snippet centered around the first occurrence of query in line."""
    line = line.rstrip()
    idx = line.lower().find(query.lower())
    if idx < 0:
        return line[: context_chars * 2]
    half = context_chars // 2
    start = max(0, idx - half)
    end = min(len(line), idx + len(query) + half)
    prefix = "\u2026" if start > 0 else ""
    suffix = "\u2026" if end < len(line) else ""
    return prefix + line[start:end] + suffix


# ---------------------------------------------------------------------------
# Tool: search_local_text
# ---------------------------------------------------------------------------


def search_local_text(call: ToolCall) -> ToolResult:
    """Search local text files for a query string."""
    raw_query = call.arguments.get("query", "")
    raw_root = call.arguments.get("root", ".")
    raw_extensions = call.arguments.get("extensions")
    raw_max_results = call.arguments.get("max_results", _DEFAULT_MAX_RESULTS)
    raw_context_chars = call.arguments.get("context_chars", _DEFAULT_CONTEXT_CHARS)

    query = str(raw_query).strip() if raw_query else ""
    if not query:
        return ToolResult.failure(
            tool_name=call.tool_name,
            error="query must be a non-empty string.",
        )

    try:
        root = _resolve_root(raw_root)
    except ValueError as exc:
        return ToolResult.failure(tool_name=call.tool_name, error=str(exc))

    try:
        max_results = int(raw_max_results)
    except (TypeError, ValueError):
        max_results = _DEFAULT_MAX_RESULTS
    max_results = max(_MIN_MAX_RESULTS, min(_MAX_MAX_RESULTS, max_results))

    try:
        context_chars = int(raw_context_chars)
    except (TypeError, ValueError):
        context_chars = _DEFAULT_CONTEXT_CHARS
    context_chars = max(_MIN_CONTEXT_CHARS, min(_MAX_CONTEXT_CHARS, context_chars))

    user_extensions = _normalize_extensions(raw_extensions)
    active_extensions: frozenset[str] = (
        user_extensions if user_extensions is not None else _DEFAULT_EXTENSIONS
    )

    matches: list[LocalTextMatchPayload] = []
    total_found = 0
    query_lower = query.lower()

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune unwanted directories in-place for efficiency
        dirnames[:] = sorted(
            d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
        )
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() not in active_extensions:
                continue
            try:
                file_size = file_path.stat().st_size
            except OSError:
                continue
            if file_size > _MAX_FILE_SIZE_BYTES:
                continue
            if _is_binary(file_path):
                continue
            try:
                with open(file_path, encoding="utf-8", errors="replace") as fh:
                    for line_number, line in enumerate(fh, start=1):
                        if query_lower in line.lower():
                            total_found += 1
                            if len(matches) < max_results:
                                snippet = _build_snippet(line, query, context_chars)
                                matches.append(
                                    {
                                        "file_path": str(file_path),
                                        "line_number": line_number,
                                        "line_text": line.rstrip(),
                                        "snippet": snippet,
                                    }
                                )
            except OSError:
                continue

    truncated = total_found > max_results
    payload: LocalTextSearchPayload = {
        "query": query,
        "root": str(root),
        "extensions_filter": sorted(user_extensions) if user_extensions is not None else None,
        "max_results": max_results,
        "total_matches_found": total_found,
        "truncated": truncated,
        "matches": matches,
    }

    if not matches:
        output = f"No matches found for '{query}' in {root}."
    else:
        header = (
            f"Found {total_found} match(es) for '{query}' in {root} "
            f"(showing {len(matches)}"
            + (" — truncated" if truncated else "")
            + "):"
        )
        output_lines = [header]
        for m in matches:
            output_lines.append(f"  {m['file_path']}:{m['line_number']}: {m['snippet']}")
        output = "\n".join(output_lines)

    return ToolResult.ok(
        tool_name=call.tool_name,
        output_text=output,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Registration hook
# ---------------------------------------------------------------------------


def register_skill_tools(registry: ToolRegistry) -> None:
    """Register local text search tools."""
    registry.register(SEARCH_LOCAL_TEXT_DEFINITION, search_local_text)
