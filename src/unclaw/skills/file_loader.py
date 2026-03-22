"""Filesystem discovery and activation for file-first optional skill bundles."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from functools import lru_cache
from pathlib import Path
import re
from types import MappingProxyType

from unclaw.skills.file_models import SkillBundle, UnknownSkillIdError

_SKILL_FILE_NAME = "SKILL.md"
_LEGACY_TO_FILE_FIRST_SKILL_IDS = MappingProxyType(
    {
        "information.weather": "weather",
    }
)
_FILE_FIRST_TO_LEGACY_SKILL_IDS = MappingProxyType(
    {
        file_first_skill_id: legacy_skill_id
        for legacy_skill_id, file_first_skill_id in _LEGACY_TO_FILE_FIRST_SKILL_IDS.items()
    }
)
_CODE_SPAN_PATTERN = re.compile(r"`([^`]+)`")
_INLINE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_ASTERISK_EMPHASIS_PATTERN = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
_LIST_PREFIX_PATTERN = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)")


def shipped_skill_bundle_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return project_root.expanduser().resolve() / "skills"

    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if _looks_like_repo_root(candidate):
            return (candidate / "skills").resolve()

    return (module_path.parents[3] / "skills").resolve()


def get_skill_bundle_roots(project_root: Path | None = None) -> tuple[Path, ...]:
    return (shipped_skill_bundle_root(project_root),)


@lru_cache(maxsize=1)
def discover_internal_skill_bundles() -> tuple[SkillBundle, ...]:
    return discover_skill_bundles(skills_root=shipped_skill_bundle_root())


def discover_skill_bundles(*, skills_root: Path | None = None) -> tuple[SkillBundle, ...]:
    resolved_skills_root = _resolve_skills_root(skills_root)
    if not resolved_skills_root.is_dir():
        return ()

    discovered_bundles = [
        load_skill_bundle(bundle_dir)
        for bundle_dir in sorted(resolved_skills_root.iterdir(), key=lambda path: path.name)
        if bundle_dir.is_dir()
        and bundle_dir.name != "__pycache__"
        and not bundle_dir.name.startswith(".")
        and (bundle_dir / _SKILL_FILE_NAME).is_file()
    ]
    return tuple(discovered_bundles)


def load_skill_bundle(bundle_dir: Path) -> SkillBundle:
    resolved_bundle_dir = bundle_dir.expanduser().resolve()
    skill_md_path = resolved_bundle_dir / _SKILL_FILE_NAME

    if not resolved_bundle_dir.is_dir():
        raise FileNotFoundError(f"Skill bundle directory does not exist: {resolved_bundle_dir}")
    if not skill_md_path.is_file():
        raise FileNotFoundError(f"Skill bundle is missing SKILL.md: {resolved_bundle_dir}")

    raw_content = skill_md_path.read_text(encoding="utf-8")
    display_name = _extract_display_name(raw_content, bundle_name=resolved_bundle_dir.name)

    return SkillBundle(
        skill_id=resolved_bundle_dir.name,
        bundle_dir=resolved_bundle_dir,
        skill_md_path=skill_md_path,
        display_name=display_name,
        summary=_extract_summary(raw_content, fallback_display_name=display_name),
        tool_hints=_extract_tool_hints(raw_content),
    )


def list_known_skill_ids(
    *,
    skills_root: Path | None = None,
    discovered_skill_bundles: Sequence[SkillBundle] | None = None,
) -> tuple[str, ...]:
    bundles = _resolve_discovered_bundles(
        skills_root=skills_root,
        discovered_skill_bundles=discovered_skill_bundles,
    )
    known_skill_ids: list[str] = []
    discovered_skill_ids = frozenset(bundle.skill_id for bundle in bundles)

    for bundle in bundles:
        _append_unique(known_skill_ids, bundle.skill_id)

    for legacy_skill_id, file_first_skill_id in sorted(
        _LEGACY_TO_FILE_FIRST_SKILL_IDS.items()
    ):
        if file_first_skill_id in discovered_skill_ids:
            _append_unique(known_skill_ids, legacy_skill_id)

    return tuple(known_skill_ids)


def load_active_skill_bundles(
    *,
    enabled_skill_ids: Collection[str],
    skills_root: Path | None = None,
    discovered_skill_bundles: Sequence[SkillBundle] | None = None,
) -> tuple[SkillBundle, ...]:
    bundles = _resolve_discovered_bundles(
        skills_root=skills_root,
        discovered_skill_bundles=discovered_skill_bundles,
    )
    bundles_by_id = {bundle.skill_id: bundle for bundle in bundles}
    known_skill_ids = list_known_skill_ids(
        skills_root=skills_root,
        discovered_skill_bundles=bundles,
    )

    active_bundles: list[SkillBundle] = []
    seen_bundle_ids: set[str] = set()
    unknown_skill_ids: list[str] = []

    for raw_skill_id in enabled_skill_ids:
        normalized_skill_id = _normalize_skill_id(raw_skill_id)
        file_first_skill_id = resolve_file_first_skill_id(normalized_skill_id)
        bundle = bundles_by_id.get(file_first_skill_id)

        if bundle is None:
            unknown_skill_ids.append(normalized_skill_id)
            continue
        if file_first_skill_id in seen_bundle_ids:
            continue

        active_bundles.append(bundle)
        seen_bundle_ids.add(file_first_skill_id)

    if unknown_skill_ids:
        raise UnknownSkillIdError(
            unknown_skill_ids=tuple(unknown_skill_ids),
            known_skill_ids=known_skill_ids,
        )

    return tuple(active_bundles)


def resolve_file_first_skill_id(skill_id: str) -> str:
    normalized_skill_id = _normalize_skill_id(skill_id)
    return _LEGACY_TO_FILE_FIRST_SKILL_IDS.get(normalized_skill_id, normalized_skill_id)


def resolve_legacy_manifest_skill_id(skill_id: str) -> str:
    normalized_skill_id = _normalize_skill_id(skill_id)
    return _FILE_FIRST_TO_LEGACY_SKILL_IDS.get(normalized_skill_id, normalized_skill_id)


def _resolve_discovered_bundles(
    *,
    skills_root: Path | None,
    discovered_skill_bundles: Sequence[SkillBundle] | None,
) -> tuple[SkillBundle, ...]:
    if discovered_skill_bundles is not None:
        return tuple(discovered_skill_bundles)
    if skills_root is None:
        return discover_internal_skill_bundles()
    return discover_skill_bundles(skills_root=skills_root)


def _resolve_skills_root(skills_root: Path | None) -> Path:
    return (skills_root or shipped_skill_bundle_root()).expanduser().resolve()


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "unclaw").is_dir()


def _extract_display_name(raw_content: str, *, bundle_name: str) -> str:
    for line in _iter_markdown_lines(raw_content):
        if line.startswith("#"):
            title = _normalize_markdown_text(line.lstrip("#").strip())
            if title:
                return title
    return _derive_display_name(bundle_name)


def _extract_summary(raw_content: str, *, fallback_display_name: str) -> str:
    for line in _iter_markdown_lines(raw_content):
        if line.startswith("#"):
            continue
        lowered_line = line.lower()
        if lowered_line.startswith("tool hint:") or lowered_line.startswith(
            "tool hints:"
        ):
            continue
        if lowered_line.startswith("tools:"):
            continue

        summary = _normalize_markdown_text(_strip_list_prefix(line))
        if summary:
            return _ensure_terminal_punctuation(summary)

    return f"{fallback_display_name} skill."


def _extract_tool_hints(raw_content: str) -> tuple[str, ...]:
    tool_hints: list[str] = []

    for line in _iter_markdown_lines(raw_content):
        lowered_line = line.lower()
        if lowered_line.startswith("tool hint:") or lowered_line.startswith(
            "tool hints:"
        ):
            raw_hint = line.split(":", 1)[1]
            normalized_hint = _normalize_markdown_text(raw_hint)
            if normalized_hint:
                tool_hints.append(_ensure_terminal_punctuation(normalized_hint))
        elif lowered_line.startswith("tools:"):
            raw_tools = line.split(":", 1)[1]
            normalized_tools = _normalize_markdown_text(raw_tools)
            if normalized_tools:
                tool_hints.append(
                    _ensure_terminal_punctuation(f"Tools: {normalized_tools}")
                )

    return tuple(tool_hints)


def _iter_markdown_lines(raw_content: str) -> tuple[str, ...]:
    lines: list[str] = []
    inside_code_block = False

    for raw_line in raw_content.splitlines():
        stripped_line = raw_line.strip()
        if stripped_line.startswith("```"):
            inside_code_block = not inside_code_block
            continue
        if inside_code_block or not stripped_line:
            continue
        lines.append(stripped_line)

    return tuple(lines)


def _strip_list_prefix(line: str) -> str:
    return _LIST_PREFIX_PATTERN.sub("", line, count=1).strip()


def _normalize_markdown_text(text: str) -> str:
    normalized_text = text.strip()
    normalized_text = _INLINE_LINK_PATTERN.sub(r"\1", normalized_text)
    normalized_text = _CODE_SPAN_PATTERN.sub(r"\1", normalized_text)
    normalized_text = _ASTERISK_EMPHASIS_PATTERN.sub(r"\1", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text)
    return normalized_text.strip()


def _derive_display_name(bundle_name: str) -> str:
    return bundle_name.replace("_", " ").replace("-", " ").strip().title()


def _normalize_skill_id(raw_skill_id: str) -> str:
    if not isinstance(raw_skill_id, str):
        raise ValueError("enabled skill ids must be strings.")
    normalized_skill_id = raw_skill_id.strip()
    if not normalized_skill_id:
        raise ValueError("enabled skill ids must be non-empty strings.")
    return normalized_skill_id


def _ensure_terminal_punctuation(text: str) -> str:
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


__all__ = [
    "discover_internal_skill_bundles",
    "discover_skill_bundles",
    "get_skill_bundle_roots",
    "list_known_skill_ids",
    "load_active_skill_bundles",
    "load_skill_bundle",
    "resolve_file_first_skill_id",
    "resolve_legacy_manifest_skill_id",
    "shipped_skill_bundle_root",
]
