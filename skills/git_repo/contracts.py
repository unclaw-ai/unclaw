"""Typed payload contracts for the git_repo skill."""

from __future__ import annotations

from typing import TypedDict


class GitFileStatusPayload(TypedDict):
    """One file entry in a git status result."""

    path: str
    status: str  # single-character code, e.g. "M", "A", "D", or "??"


class GitStatusPayload(TypedDict):
    """Structured result from git_status."""

    repo_path: str
    branch: str | None
    ahead: int | None
    behind: int | None
    staged_count: int
    unstaged_count: int
    untracked_count: int
    staged_files: list[GitFileStatusPayload]
    unstaged_files: list[GitFileStatusPayload]
    untracked_files: list[GitFileStatusPayload]
    truncated: bool


class GitCommitPayload(TypedDict):
    """One commit entry in a git log result."""

    hash: str
    short_hash: str
    author: str
    date: str
    subject: str


class GitRecentCommitsPayload(TypedDict):
    """Structured result from git_recent_commits."""

    repo_path: str
    limit: int
    commits: list[GitCommitPayload]


class GitChangedFilePayload(TypedDict):
    """One changed file entry in a git diff summary."""

    path: str
    insertions: int | None
    deletions: int | None


class GitDiffSummaryPayload(TypedDict):
    """Structured result from git_diff_summary."""

    repo_path: str
    target: str
    pathspec: str | None
    changed_file_count: int
    total_insertions: int | None
    total_deletions: int | None
    changed_files: list[GitChangedFilePayload]
    truncated: bool
    stat_summary: str


__all__ = [
    "GitChangedFilePayload",
    "GitCommitPayload",
    "GitDiffSummaryPayload",
    "GitFileStatusPayload",
    "GitRecentCommitsPayload",
    "GitStatusPayload",
]
