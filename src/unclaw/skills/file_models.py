"""File-first optional skill bundle metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Process-lifetime cache for SKILL.md raw content.  Keys are resolved absolute
# paths (set by SkillBundle.__post_init__).  Skills are configured at startup
# and their files do not change at runtime, so a plain dict is safe and avoids
# the per-call disk read on every turn a skill is selected.
_raw_content_cache: dict[Path, str] = {}


def _require_non_empty(value: str, *, field_name: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized_value


def _normalize_string_tuple(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized_values: list[str] = []

    for raw_value in values:
        normalized_value = _require_non_empty(raw_value, field_name=field_name)
        if normalized_value not in normalized_values:
            normalized_values.append(normalized_value)

    return tuple(normalized_values)


@dataclass(frozen=True, slots=True)
class SkillBundle:
    """Compact metadata for one file-first optional skill bundle."""

    skill_id: str
    bundle_dir: Path
    skill_md_path: Path
    display_name: str
    summary: str
    tool_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        resolved_bundle_dir = self.bundle_dir.expanduser().resolve()
        resolved_skill_md_path = self.skill_md_path.expanduser().resolve()

        if resolved_skill_md_path.name != "SKILL.md":
            raise ValueError("skill_md_path must point to SKILL.md.")
        if resolved_skill_md_path.parent != resolved_bundle_dir:
            raise ValueError("skill_md_path must live inside bundle_dir.")

        object.__setattr__(
            self,
            "skill_id",
            _require_non_empty(self.skill_id, field_name="skill_id"),
        )
        object.__setattr__(self, "bundle_dir", resolved_bundle_dir)
        object.__setattr__(self, "skill_md_path", resolved_skill_md_path)
        object.__setattr__(
            self,
            "display_name",
            _require_non_empty(self.display_name, field_name="display_name"),
        )
        object.__setattr__(
            self,
            "summary",
            _require_non_empty(self.summary, field_name="summary"),
        )
        object.__setattr__(
            self,
            "tool_hints",
            _normalize_string_tuple(self.tool_hints, field_name="tool_hints"),
        )

    def load_raw_content(self) -> str:
        """Return the raw text of SKILL.md, reading from disk only on first call.

        The result is cached in ``_raw_content_cache`` keyed by the resolved
        absolute path.  Subsequent calls for the same path are free (no I/O).
        Each test that writes to a unique ``tmp_path`` gets its own cache entry,
        so there are no cross-test collisions.
        """
        cached = _raw_content_cache.get(self.skill_md_path)
        if cached is not None:
            return cached
        content = self.skill_md_path.read_text(encoding="utf-8")
        _raw_content_cache[self.skill_md_path] = content
        return content


class UnknownSkillIdError(ValueError):
    """Raised when config or activation references an unknown file-first skill."""

    def __init__(
        self,
        *,
        unknown_skill_ids: tuple[str, ...],
        known_skill_ids: tuple[str, ...],
    ) -> None:
        self.unknown_skill_ids = unknown_skill_ids
        self.known_skill_ids = known_skill_ids
        unknown_labels = ", ".join(unknown_skill_ids)
        known_labels = ", ".join(known_skill_ids) if known_skill_ids else "[none]"
        super().__init__(
            f"Unknown enabled skill id(s): {unknown_labels}. "
            f"Known skill ids: {known_labels}."
        )


def clear_skill_bundle_cache(skill_md_path: Path) -> None:
    """Drop one cached SKILL.md entry if it exists."""
    _raw_content_cache.pop(skill_md_path.expanduser().resolve(), None)


__all__ = ["SkillBundle", "UnknownSkillIdError", "clear_skill_bundle_cache"]
