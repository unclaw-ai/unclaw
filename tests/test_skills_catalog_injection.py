"""Unit tests for compact active-skill catalog injection into context messages.

Phase 2 success criteria verified here:
- active skills appear in compact catalog form in context messages
- inactive skills (no enabled ids) produce no catalog injection
- full SKILL.md content is NOT injected
- built-in capability context and optional-skill catalog are separate messages
- legacy skill notes still work alongside the new catalog
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.core.capabilities import RuntimeCapabilitySummary
from unclaw.core.context_builder import build_context_messages
from unclaw.llm.base import LLMRole

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


def _make_session_manager(
    enabled_skill_ids: tuple[str, ...] = (),
    system_prompt: str = "Base system prompt.",
) -> SimpleNamespace:
    """Minimal stub that satisfies context_builder's attribute accesses."""
    skills = SimpleNamespace(enabled_skill_ids=enabled_skill_ids)
    settings = SimpleNamespace(
        system_prompt=system_prompt,
        model_pack="dev",
        skills=skills,
    )
    return SimpleNamespace(
        settings=settings,
        list_messages=lambda session_id: [],
    )


def _minimal_capability_summary() -> RuntimeCapabilitySummary:
    return RuntimeCapabilitySummary(
        available_builtin_tool_names=("get_weather",),
        local_file_read_available=True,
        local_directory_listing_available=False,
        url_fetch_available=False,
        web_search_available=False,
        system_info_available=False,
        memory_summary_available=False,
        model_can_call_tools=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compact_catalog_appears_as_separate_system_message(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Active skills produce a compact catalog as a distinct SYSTEM message."""
    catalog_text = "Active optional skills:\n- weather: Live weather and short forecasts. Prefer get_weather."
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_catalog",
        lambda **kwargs: catalog_text,
    )
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_context_notes",
        lambda **kwargs: (),
    )

    session_manager = _make_session_manager(enabled_skill_ids=("weather",))
    messages = build_context_messages(
        session_manager=session_manager,
        session_id="test-session",
        user_message="What is the weather in Paris?",
        capability_summary=_minimal_capability_summary(),
        model_profile_name="native",
    )

    system_messages = [m for m in messages if m.role is LLMRole.SYSTEM]
    catalog_msgs = [m for m in system_messages if "Active optional skills" in m.content]

    assert len(catalog_msgs) == 1
    assert "- weather:" in catalog_msgs[0].content
    assert "Prefer get_weather" in catalog_msgs[0].content


def test_no_catalog_message_when_no_skills_enabled(monkeypatch) -> None:
    """Empty enabled_skill_ids produces no catalog message."""
    called = []
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_catalog",
        lambda **kwargs: called.append(kwargs) or "",
    )
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_context_notes",
        lambda **kwargs: (),
    )

    session_manager = _make_session_manager(enabled_skill_ids=())
    messages = build_context_messages(
        session_manager=session_manager,
        session_id="test-session",
        user_message="Hello.",
        capability_summary=_minimal_capability_summary(),
        model_profile_name="native",
    )

    catalog_msgs = [m for m in messages if "Active optional skills" in m.content]
    assert catalog_msgs == []
    # build_active_skill_catalog should not be called when there are no enabled skills
    assert called == []


def test_catalog_is_separate_from_capability_context(monkeypatch) -> None:
    """Built-in tool context and optional-skill catalog are in separate SYSTEM messages."""
    catalog_text = "Active optional skills:\n- weather: Live weather."
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_catalog",
        lambda **kwargs: catalog_text,
    )
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_context_notes",
        lambda **kwargs: (),
    )

    session_manager = _make_session_manager(enabled_skill_ids=("weather",))
    messages = build_context_messages(
        session_manager=session_manager,
        session_id="test-session",
        user_message="What is the weather?",
        capability_summary=_minimal_capability_summary(),
        model_profile_name="native",
    )

    system_messages = [m for m in messages if m.role is LLMRole.SYSTEM]
    # message 0: base system prompt
    # message 1: capability context (built-in tools)
    # message 2: compact skill catalog
    assert len(system_messages) >= 3
    capability_msg = system_messages[1]
    catalog_msg = system_messages[2]

    # built-in tools are in the capability message, not mixed into the catalog
    assert "Available built-in tools" in capability_msg.content or "built-in" in capability_msg.content.lower()
    assert "Active optional skills" in catalog_msg.content
    assert "Active optional skills" not in capability_msg.content


def test_full_skill_md_content_is_not_injected(monkeypatch) -> None:
    """The catalog is compact — no raw SKILL.md body should appear."""
    full_skill_md_body = (
        "# Weather\n\nLive weather and short forecasts.\n\n"
        "Tool hints: Prefer `get_weather`; use `search_web` only as fallback.\n\n"
        "This bundle is present for Skills discovery and compact-catalog loading.\n"
        "Live runtime prompt ownership stays in the legacy manifest path for now.\n"
    )
    # Simulate catalog returning only the compact form, not the full body
    compact_catalog = "Active optional skills:\n- weather: Live weather and short forecasts."
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_catalog",
        lambda **kwargs: compact_catalog,
    )
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_context_notes",
        lambda **kwargs: (),
    )

    session_manager = _make_session_manager(enabled_skill_ids=("weather",))
    messages = build_context_messages(
        session_manager=session_manager,
        session_id="test-session",
        user_message="Weather in London?",
        capability_summary=_minimal_capability_summary(),
        model_profile_name="native",
    )

    all_content = "\n".join(m.content for m in messages)
    # The raw multi-line bundle description should not appear verbatim
    assert "This bundle is present for Skills discovery" not in all_content
    assert "Live runtime prompt ownership stays in the legacy manifest" not in all_content
    # The compact catalog IS there
    assert "Active optional skills:" in all_content


def test_unknown_skill_id_silently_produces_no_catalog(monkeypatch) -> None:
    """An UnknownSkillIdError from the catalog path does not crash the turn."""
    from unclaw.skills.file_models import UnknownSkillIdError

    def _raise(**kwargs):
        raise UnknownSkillIdError(
            unknown_skill_ids=("ghost.skill",),
            known_skill_ids=("weather",),
        )

    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_catalog",
        _raise,
    )
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_context_notes",
        lambda **kwargs: (),
    )

    session_manager = _make_session_manager(enabled_skill_ids=("ghost.skill",))
    # Should not raise
    messages = build_context_messages(
        session_manager=session_manager,
        session_id="test-session",
        user_message="Hello.",
        capability_summary=_minimal_capability_summary(),
        model_profile_name="native",
    )

    catalog_msgs = [m for m in messages if "Active optional skills" in m.content]
    assert catalog_msgs == []


def test_catalog_coexists_with_legacy_skill_notes(monkeypatch) -> None:
    """Compact catalog and legacy manifest notes both appear — weather preserved."""
    catalog_text = "Active optional skills:\n- weather: Live weather."
    legacy_note = "Active optional skill: Weather\n- Prefer get_weather for conditions."
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_catalog",
        lambda **kwargs: catalog_text,
    )
    monkeypatch.setattr(
        "unclaw.core.context_builder.build_active_skill_context_notes",
        lambda **kwargs: (legacy_note,),
    )

    session_manager = _make_session_manager(enabled_skill_ids=("weather",))
    messages = build_context_messages(
        session_manager=session_manager,
        session_id="test-session",
        user_message="What is the weather?",
        capability_summary=_minimal_capability_summary(),
        model_profile_name="native",
    )

    system_messages = [m for m in messages if m.role is LLMRole.SYSTEM]
    catalog_msgs = [m for m in system_messages if "Active optional skills:" in m.content]
    legacy_msgs = [m for m in system_messages if "Active optional skill: Weather" in m.content]

    assert len(catalog_msgs) == 1, "compact catalog must appear"
    assert len(legacy_msgs) == 1, "legacy weather notes must still appear"
    # They must be distinct messages
    assert catalog_msgs[0] is not legacy_msgs[0]


def test_catalog_uses_real_shipped_weather_bundle() -> None:
    """Integration smoke: the real skills/weather/SKILL.md produces a valid compact entry."""
    from unclaw.skills.catalog import build_active_skill_catalog

    catalog = build_active_skill_catalog(enabled_skill_ids=("weather",))

    assert catalog.startswith("Active optional skills:")
    assert "- weather:" in catalog
    assert "get_weather" in catalog
    # Must be compact — much shorter than a typical SKILL.md
    assert len(catalog) < 300
