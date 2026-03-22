from __future__ import annotations

from pathlib import Path

import pytest

from unclaw.core.capability_fragments import (
    CapabilityPromptSourceKind,
    load_builtin_capability_fragment_registry,
)
from unclaw.settings import load_settings

pytestmark = pytest.mark.unit


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_shipped_prompt_directory_contains_only_the_stable_system_prompt() -> None:
    repo_root = _repo_root()
    settings = load_settings(project_root=repo_root)
    prompts_dir = repo_root / "config" / "prompts"
    prompt_files = tuple(
        path.relative_to(prompts_dir).as_posix()
        for path in sorted(prompts_dir.iterdir())
        if path.is_file()
    )

    assert prompt_files == ("system.txt",)
    assert settings.paths.system_prompt_path == prompts_dir / "system.txt"


def test_builtin_capability_prompt_sources_stay_inside_capability_fragments_module() -> None:
    registry = load_builtin_capability_fragment_registry()

    for fragment in registry.list_fragments():
        assert fragment.prompt_source.kind in {
            CapabilityPromptSourceKind.INLINE,
            CapabilityPromptSourceKind.FUNCTION,
        }
        assert fragment.prompt_source.reference.startswith(
            "unclaw.core.capability_fragments"
        )


