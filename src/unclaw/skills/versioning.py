"""Lightweight, tolerant version comparison for the skills hub."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class VersionComparison(StrEnum):
    """Relationship between one local version and one catalog version."""

    CATALOG_NEWER = "catalog_newer"
    EQUAL = "equal"
    LOCAL_NEWER = "local_newer"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ParsedVersion:
    """Small semver-ish representation used for internal comparisons."""

    release: tuple[int, ...]
    prerelease: tuple[int | str, ...]


_VERSION_PREFIX_RE = re.compile(r"^[vV]?(?P<release>\d+(?:\.\d+)*)(?P<suffix>.*)$")
_PRERELEASE_SPLIT_RE = re.compile(r"[.\-_]+")


def compare_versions(
    local_version: str | None,
    catalog_version: str | None,
) -> VersionComparison:
    """Compare one installed version to one catalog version.

    The parser intentionally accepts a broader range than strict semver:

    - ``1``, ``1.2``, ``1.2.3``
    - optional ``v`` prefix
    - prerelease-ish suffixes such as ``-beta.1`` or ``rc1``

    When either value is missing or cannot be compared safely, ``UNKNOWN`` is
    returned instead of raising.
    """

    normalized_local = _normalize_version_text(local_version)
    normalized_catalog = _normalize_version_text(catalog_version)
    if normalized_local is None or normalized_catalog is None:
        return VersionComparison.UNKNOWN
    if normalized_local == normalized_catalog:
        return VersionComparison.EQUAL

    parsed_local = _parse_version(normalized_local)
    parsed_catalog = _parse_version(normalized_catalog)
    if parsed_local is None or parsed_catalog is None:
        return VersionComparison.UNKNOWN

    ordering = _compare_parsed_versions(parsed_local, parsed_catalog)
    if ordering < 0:
        return VersionComparison.CATALOG_NEWER
    if ordering > 0:
        return VersionComparison.LOCAL_NEWER
    return VersionComparison.EQUAL


def _normalize_version_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized_value = value.strip()
    if not normalized_value:
        return None
    return normalized_value


def _parse_version(value: str) -> ParsedVersion | None:
    match = _VERSION_PREFIX_RE.fullmatch(value)
    if match is None:
        return None

    release = tuple(int(part) for part in match.group("release").split("."))
    suffix = match.group("suffix").strip()

    build_index = suffix.find("+")
    if build_index >= 0:
        suffix = suffix[:build_index]

    prerelease_text = suffix.lstrip(".-_")
    prerelease = tuple(_coerce_prerelease_identifier(part) for part in _split_suffix(prerelease_text))
    return ParsedVersion(release=release, prerelease=prerelease)


def _split_suffix(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(
        part
        for part in _PRERELEASE_SPLIT_RE.split(value)
        if part
    )


def _coerce_prerelease_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value.casefold()


def _compare_parsed_versions(local: ParsedVersion, catalog: ParsedVersion) -> int:
    release_ordering = _compare_release_segments(local.release, catalog.release)
    if release_ordering != 0:
        return release_ordering

    if not local.prerelease and not catalog.prerelease:
        return 0
    if not local.prerelease:
        return 1
    if not catalog.prerelease:
        return -1

    for local_identifier, catalog_identifier in zip(local.prerelease, catalog.prerelease):
        if local_identifier == catalog_identifier:
            continue
        if isinstance(local_identifier, int) and isinstance(catalog_identifier, int):
            return -1 if local_identifier < catalog_identifier else 1
        if isinstance(local_identifier, int):
            return -1
        if isinstance(catalog_identifier, int):
            return 1
        return -1 if local_identifier < catalog_identifier else 1

    if len(local.prerelease) == len(catalog.prerelease):
        return 0
    return -1 if len(local.prerelease) < len(catalog.prerelease) else 1


def _compare_release_segments(
    local_segments: tuple[int, ...],
    catalog_segments: tuple[int, ...],
) -> int:
    max_length = max(len(local_segments), len(catalog_segments))
    padded_local = (*local_segments, *(0 for _ in range(max_length - len(local_segments))))
    padded_catalog = (
        *catalog_segments,
        *(0 for _ in range(max_length - len(catalog_segments))),
    )
    for local_part, catalog_part in zip(padded_local, padded_catalog):
        if local_part == catalog_part:
            continue
        return -1 if local_part < catalog_part else 1
    return 0


__all__ = ["ParsedVersion", "VersionComparison", "compare_versions"]
