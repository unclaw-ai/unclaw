"""Skill-owned read-only git repository inspection tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from skills.git_repo.contracts import (
    GitChangedFilePayload,
    GitCommitPayload,
    GitDiffSummaryPayload,
    GitFileStatusPayload,
    GitRecentCommitsPayload,
    GitStatusPayload,
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
_MAX_STATUS_FILES_PER_BUCKET = 50
_MIN_COMMIT_LIMIT = 1
_MAX_COMMIT_LIMIT = 50
_DEFAULT_COMMIT_LIMIT = 10
_MAX_DIFF_FILES = 100

# Only allow git ref characters that are safe to pass as subprocess arguments.
# This covers commit hashes, branch names, tags, and common relative refs.
_SAFE_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._/^~@{}\-:]+$")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

GIT_STATUS_DEFINITION = ToolDefinition(
    name="git_status",
    description=(
        "Return a compact structured summary of the given local git repository: "
        "current branch, ahead/behind upstream (if available), staged/unstaged/untracked "
        "file counts, and a bounded file list."
    ),
    permission_level=ToolPermissionLevel.LOCAL_EXECUTE,
    arguments={
        "repo_path": ToolArgumentSpec(
            description=(
                "Path to the local git repository root. "
                "Defaults to '.' (current working directory)."
            ),
        ),
    },
)

GIT_RECENT_COMMITS_DEFINITION = ToolDefinition(
    name="git_recent_commits",
    description=(
        "Return the most recent commits from the given local git repository, "
        "including full hash, short hash, author, date, and subject line."
    ),
    permission_level=ToolPermissionLevel.LOCAL_EXECUTE,
    arguments={
        "repo_path": ToolArgumentSpec(
            description="Path to the local git repository root. Defaults to '.'.",
        ),
        "limit": ToolArgumentSpec(
            description=(
                f"Maximum number of commits to return. "
                f"Clamped to {_MIN_COMMIT_LIMIT}–{_MAX_COMMIT_LIMIT}. "
                f"Defaults to {_DEFAULT_COMMIT_LIMIT}."
            ),
            value_type="integer",
        ),
    },
)

GIT_DIFF_SUMMARY_DEFINITION = ToolDefinition(
    name="git_diff_summary",
    description=(
        "Return a bounded structured diff summary for the given local git repository: "
        "list of changed files with insertions/deletions, and a compact stat summary."
    ),
    permission_level=ToolPermissionLevel.LOCAL_EXECUTE,
    arguments={
        "repo_path": ToolArgumentSpec(
            description="Path to the local git repository root. Defaults to '.'.",
        ),
        "target": ToolArgumentSpec(
            description=(
                "Git ref or commit to diff against. Defaults to 'HEAD'. "
                "Examples: 'HEAD~3', 'main', a full or short commit hash."
            ),
        ),
        "pathspec": ToolArgumentSpec(
            description=(
                "Optional path or glob to restrict the diff (e.g. 'src/', '*.py'). "
                "Omit to diff all files."
            ),
        ),
    },
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_repo_path(raw: object) -> Path:
    """Resolve and validate a repo_path argument. Returns an absolute Path."""
    p = Path(str(raw) if raw else ".").resolve()
    if not p.exists():
        raise ValueError(f"Repository path does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Repository path is not a directory: {p}")
    return p


def _assert_git_repo(repo_path: Path) -> None:
    """Raise ValueError if repo_path is not inside a git repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Not a git repository: {repo_path}")


def _run_git(repo_path: Path, *args: str) -> tuple[int, str, str]:
    """Run a git subcommand. Returns (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def _validate_git_ref(ref: str) -> str:
    """Validate and return a git ref. Raises ValueError for unsafe values."""
    ref = ref.strip()
    if not ref:
        raise ValueError("Git ref must not be empty.")
    if not _SAFE_GIT_REF_RE.match(ref):
        raise ValueError(
            f"Unsafe git ref: {ref!r}. "
            "Only alphanumeric characters, '.', '-', '_', '/', '^', '~', '@', '{{', '}}', "
            "and ':' are allowed."
        )
    return ref


# ---------------------------------------------------------------------------
# Tool: git_status
# ---------------------------------------------------------------------------


def git_status(call: ToolCall) -> ToolResult:
    """Return a compact structured status summary for a local git repository."""
    raw_path = call.arguments.get("repo_path", ".")
    try:
        repo_path = _resolve_repo_path(raw_path)
        _assert_git_repo(repo_path)
    except ValueError as exc:
        return ToolResult.failure(tool_name=call.tool_name, error=str(exc))

    # Current branch
    _rc, branch_out, _ = _run_git(repo_path, "branch", "--show-current")
    branch: str | None = branch_out.strip() or None

    # Ahead/behind upstream (may not be available for all branches)
    ahead: int | None = None
    behind: int | None = None
    _rc2, ab_out, _ = _run_git(
        repo_path, "rev-list", "--left-right", "--count", "HEAD...@{u}"
    )
    if _rc2 == 0:
        parts = ab_out.strip().split()
        if len(parts) == 2:
            try:
                ahead = int(parts[0])
                behind = int(parts[1])
            except ValueError:
                pass

    # Porcelain status (machine-readable)
    _rc3, status_out, _ = _run_git(repo_path, "status", "--porcelain=v1")
    staged: list[GitFileStatusPayload] = []
    unstaged: list[GitFileStatusPayload] = []
    untracked: list[GitFileStatusPayload] = []
    for line in status_out.splitlines():
        if len(line) < 3:
            continue
        x, y, path = line[0], line[1], line[3:]
        if x == "?" and y == "?":
            untracked.append({"path": path, "status": "??"})
        else:
            if x not in (" ", "?"):
                staged.append({"path": path, "status": x})
            if y not in (" ", "?"):
                unstaged.append({"path": path, "status": y})

    truncated = (
        len(staged) > _MAX_STATUS_FILES_PER_BUCKET
        or len(unstaged) > _MAX_STATUS_FILES_PER_BUCKET
        or len(untracked) > _MAX_STATUS_FILES_PER_BUCKET
    )
    payload: GitStatusPayload = {
        "repo_path": str(repo_path),
        "branch": branch,
        "ahead": ahead,
        "behind": behind,
        "staged_count": len(staged),
        "unstaged_count": len(unstaged),
        "untracked_count": len(untracked),
        "staged_files": staged[:_MAX_STATUS_FILES_PER_BUCKET],
        "unstaged_files": unstaged[:_MAX_STATUS_FILES_PER_BUCKET],
        "untracked_files": untracked[:_MAX_STATUS_FILES_PER_BUCKET],
        "truncated": truncated,
    }

    lines: list[str] = [
        f"Repository: {repo_path}",
        f"Branch: {branch or '(detached HEAD)'}",
    ]
    if ahead is not None and behind is not None:
        lines.append(f"Upstream: {ahead} ahead, {behind} behind")
    lines.append(
        f"Changes: {len(staged)} staged, {len(unstaged)} unstaged, "
        f"{len(untracked)} untracked"
    )
    if staged:
        lines.append("Staged files:")
        for f in staged[:_MAX_STATUS_FILES_PER_BUCKET]:
            lines.append(f"  [{f['status']}] {f['path']}")
    if unstaged:
        lines.append("Unstaged files:")
        for f in unstaged[:_MAX_STATUS_FILES_PER_BUCKET]:
            lines.append(f"  [{f['status']}] {f['path']}")
    if untracked:
        lines.append("Untracked files:")
        for f in untracked[:_MAX_STATUS_FILES_PER_BUCKET]:
            lines.append(f"  {f['path']}")
    if truncated:
        lines.append("(Output truncated — too many files to display fully)")

    return ToolResult.ok(
        tool_name=call.tool_name,
        output_text="\n".join(lines),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Tool: git_recent_commits
# ---------------------------------------------------------------------------


def git_recent_commits(call: ToolCall) -> ToolResult:
    """Return the most recent commits from a local git repository."""
    raw_path = call.arguments.get("repo_path", ".")
    raw_limit = call.arguments.get("limit", _DEFAULT_COMMIT_LIMIT)
    try:
        repo_path = _resolve_repo_path(raw_path)
        _assert_git_repo(repo_path)
    except ValueError as exc:
        return ToolResult.failure(tool_name=call.tool_name, error=str(exc))

    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        limit = _DEFAULT_COMMIT_LIMIT
    limit = max(_MIN_COMMIT_LIMIT, min(_MAX_COMMIT_LIMIT, limit))

    # Use a non-printable unit-separator (0x1F) as the field delimiter so
    # commit subjects with pipes, tabs, or commas are preserved cleanly.
    rc, log_out, stderr = _run_git(
        repo_path,
        "log",
        f"-n{limit}",
        "--format=%H\x1f%h\x1f%an\x1f%ad\x1f%s",
        "--date=short",
    )
    if rc != 0:
        return ToolResult.failure(
            tool_name=call.tool_name,
            error=f"git log failed: {stderr.strip()}",
        )

    commits: list[GitCommitPayload] = []
    for raw_line in log_out.splitlines():
        parts = raw_line.split("\x1f", 4)
        if len(parts) == 5:
            commits.append(
                {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                    "subject": parts[4],
                }
            )

    payload: GitRecentCommitsPayload = {
        "repo_path": str(repo_path),
        "limit": limit,
        "commits": commits,
    }
    lines = [
        f"Repository: {repo_path}",
        f"Showing {len(commits)} most recent commit(s):",
    ]
    for c in commits:
        lines.append(f"  {c['short_hash']}  {c['date']}  {c['author']}: {c['subject']}")

    return ToolResult.ok(
        tool_name=call.tool_name,
        output_text="\n".join(lines),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Tool: git_diff_summary
# ---------------------------------------------------------------------------


def git_diff_summary(call: ToolCall) -> ToolResult:
    """Return a bounded structured diff summary for a local git repository."""
    raw_path = call.arguments.get("repo_path", ".")
    raw_target = call.arguments.get("target", "HEAD")
    raw_pathspec = call.arguments.get("pathspec")
    try:
        repo_path = _resolve_repo_path(raw_path)
        _assert_git_repo(repo_path)
        target = _validate_git_ref(str(raw_target) if raw_target else "HEAD")
    except ValueError as exc:
        return ToolResult.failure(tool_name=call.tool_name, error=str(exc))

    pathspec: str | None = None
    if raw_pathspec and str(raw_pathspec).strip():
        pathspec = str(raw_pathspec).strip()

    # --stat for the human-readable summary line
    stat_args = ["diff", "--stat", target]
    if pathspec:
        stat_args += ["--", pathspec]
    rc_stat, stat_out, stat_err = _run_git(repo_path, *stat_args)
    if rc_stat != 0:
        return ToolResult.failure(
            tool_name=call.tool_name,
            error=f"git diff failed: {stat_err.strip()}",
        )

    # --numstat for structured per-file data
    numstat_args = ["diff", "--numstat", target]
    if pathspec:
        numstat_args += ["--", pathspec]
    _rc_ns, numstat_out, _ = _run_git(repo_path, *numstat_args)

    changed_files: list[GitChangedFilePayload] = []
    total_insertions = 0
    total_deletions = 0
    all_numeric = True
    numstat_lines = numstat_out.splitlines()

    for raw_line in numstat_lines[:_MAX_DIFF_FILES]:
        parts = raw_line.split("\t", 2)
        if len(parts) == 3:
            ins_raw, del_raw, path = parts
            try:
                ins = int(ins_raw)
                del_ = int(del_raw)
                total_insertions += ins
                total_deletions += del_
                changed_files.append({"path": path, "insertions": ins, "deletions": del_})
            except ValueError:
                # Binary files show "-" instead of counts
                all_numeric = False
                changed_files.append({"path": path, "insertions": None, "deletions": None})

    truncated = len(numstat_lines) > _MAX_DIFF_FILES

    # The last non-empty line from --stat contains the aggregate totals
    stat_lines = [ln for ln in stat_out.splitlines() if ln.strip()]
    stat_summary = stat_lines[-1].strip() if stat_lines else ""

    payload: GitDiffSummaryPayload = {
        "repo_path": str(repo_path),
        "target": target,
        "pathspec": pathspec,
        "changed_file_count": len(changed_files),
        "total_insertions": total_insertions if all_numeric else None,
        "total_deletions": total_deletions if all_numeric else None,
        "changed_files": changed_files,
        "truncated": truncated,
        "stat_summary": stat_summary,
    }

    lines: list[str] = [
        f"Repository: {repo_path}",
        f"Diff target: {target}",
    ]
    if pathspec:
        lines.append(f"Pathspec: {pathspec}")
    lines.append(f"Changed files: {len(changed_files)}")
    if all_numeric and changed_files:
        lines.append(f"Total: +{total_insertions} insertions, -{total_deletions} deletions")
    for f in changed_files:
        if f["insertions"] is not None:
            lines.append(f"  {f['path']}  (+{f['insertions']}/-{f['deletions']})")
        else:
            lines.append(f"  {f['path']}  (binary)")
    if truncated:
        lines.append("(Output truncated — too many changed files)")
    if stat_summary:
        lines.append(f"Summary: {stat_summary}")

    return ToolResult.ok(
        tool_name=call.tool_name,
        output_text="\n".join(lines),
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Registration hook
# ---------------------------------------------------------------------------


def register_skill_tools(registry: ToolRegistry) -> None:
    """Register git repository inspection tools."""
    registry.register(GIT_STATUS_DEFINITION, git_status)
    registry.register(GIT_RECENT_COMMITS_DEFINITION, git_recent_commits)
    registry.register(GIT_DIFF_SUMMARY_DEFINITION, git_diff_summary)
