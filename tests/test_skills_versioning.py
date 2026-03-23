from __future__ import annotations

import pytest

from unclaw.skills.versioning import VersionComparison, compare_versions

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("local_version", "catalog_version", "expected"),
    [
        ("0.1.0", "0.1.1", VersionComparison.CATALOG_NEWER),
        ("0.1.2", "0.1.1", VersionComparison.LOCAL_NEWER),
        ("0.1.0", "0.1.0", VersionComparison.EQUAL),
        ("1.0", "1.0.0", VersionComparison.EQUAL),
        ("v1.2.0", "1.2.1", VersionComparison.CATALOG_NEWER),
        ("1.0.0-rc.1", "1.0.0", VersionComparison.CATALOG_NEWER),
    ],
)
def test_compare_versions_semver_rules(
    local_version: str,
    catalog_version: str,
    expected: VersionComparison,
) -> None:
    assert compare_versions(local_version, catalog_version) is expected


@pytest.mark.parametrize(
    ("local_version", "catalog_version"),
    [
        ("dev-build", "1.0.0"),
        ("1.0.0", "release-latest"),
        (None, "1.0.0"),
        ("1.0.0", None),
    ],
)
def test_compare_versions_returns_unknown_for_unparseable_values(
    local_version: str | None,
    catalog_version: str | None,
) -> None:
    assert compare_versions(local_version, catalog_version) is VersionComparison.UNKNOWN
