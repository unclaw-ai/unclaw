"""Small shared timing helpers for core runtime paths."""

from __future__ import annotations

from time import perf_counter


def elapsed_ms(started_at: float) -> int:
    """Return a non-negative elapsed duration in milliseconds."""
    return max(0, round((perf_counter() - started_at) * 1000))
