from __future__ import annotations

import pytest

from unclaw.core.capabilities import (
    build_runtime_capability_context,
    build_runtime_capability_summary,
)
from unclaw.core.capability_budget import (
    CapabilityBudgetTier,
    resolve_capability_budget_policy,
)
from unclaw.core.context_builder import build_context_messages
from unclaw.core.executor import create_default_tool_registry
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMRole
from unclaw.settings import load_settings
from unclaw.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("model_pack", "profile_name", "expected_tier"),
    [
        ("lite", "fast", CapabilityBudgetTier.MINIMAL),
        ("lite", "main", CapabilityBudgetTier.COMPACT),
        ("lite", "deep", CapabilityBudgetTier.COMPACT),
        ("lite", "codex", CapabilityBudgetTier.COMPACT),
        ("sweet", "main", CapabilityBudgetTier.STANDARD),
        ("power", "deep", CapabilityBudgetTier.STANDARD),
        ("dev", "fast", CapabilityBudgetTier.MINIMAL),
        ("dev", "codex", CapabilityBudgetTier.STANDARD),
        ("dev", "analysis", CapabilityBudgetTier.STANDARD),
    ],
)
def test_capability_budget_policy_resolves_deterministically_by_pack_and_profile(
    model_pack: str,
    profile_name: str,
    expected_tier: CapabilityBudgetTier,
) -> None:
    policy = resolve_capability_budget_policy(
        model_pack=model_pack,
        model_profile_name=profile_name,
    )

    assert policy.tier is expected_tier


def test_fast_budget_keeps_capability_context_minimal() -> None:
    summary = build_runtime_capability_summary(
        tool_registry=ToolRegistry(),
        memory_summary_available=False,
        model_can_call_tools=False,
    )
    policy = resolve_capability_budget_policy(
        model_pack="lite",
        model_profile_name="fast",
    )

    context = build_runtime_capability_context(summary, budget_policy=policy)

    assert "Enabled built-in tools: 0" in context
    assert "Available built-in tools:" not in context
    assert "Available built-in tools (compact):" not in context
    assert "Unavailable capabilities:" not in context
    assert "Unavailable capabilities (compact):" not in context
    assert "user-initiated slash commands only" in context
    assert "Do not claim you already searched" in context


def test_lite_compact_budget_renders_smaller_context_than_power_for_same_surface(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        registry = create_default_tool_registry(
            settings,
            session_manager=session_manager,
        )
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=True,
        )
    finally:
        session_manager.close()

    lite_context = build_runtime_capability_context(
        summary,
        budget_policy=resolve_capability_budget_policy(
            model_pack="lite",
            model_profile_name="main",
        ),
    )
    power_context = build_runtime_capability_context(
        summary,
        budget_policy=resolve_capability_budget_policy(
            model_pack="power",
            model_profile_name="main",
        ),
    )

    assert len(lite_context) < len(power_context)
    assert "Available built-in tools (compact):" in lite_context
    assert "Available built-in tools:" in power_context
    assert "/fetch <url>: fetch one public URL and extract text." not in lite_context
    assert "/fetch <url>: fetch one public URL and extract text." in power_context
    assert "Unavailable capabilities:" not in lite_context
    assert "Unavailable capabilities:" in power_context
    assert "Use system_info for current local machine facts and runtime facts" in lite_context
    assert "Use search_long_term_memory for targeted recall of a stored fact." in lite_context


def test_build_context_messages_resolves_live_pack_profile_budget(
    make_temp_project,
    write_models_config,
) -> None:
    project_root = make_temp_project()
    write_models_config(project_root, active_pack="lite")

    lite_context = _build_context_messages_capability_note(
        project_root=project_root,
        profile_name="main",
    )
    write_models_config(project_root, active_pack="power")
    power_context = _build_context_messages_capability_note(
        project_root=project_root,
        profile_name="main",
    )

    assert len(lite_context) < len(power_context)
    assert "Available built-in tools (compact):" in lite_context
    assert "Unavailable capabilities:" not in lite_context
    assert "Available built-in tools:" in power_context
    assert "Unavailable capabilities:" in power_context


def test_build_context_messages_resolves_minimal_fast_budget(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.create_session(make_current=False)
        summary = build_runtime_capability_summary(
            tool_registry=ToolRegistry(),
            memory_summary_available=False,
            model_can_call_tools=False,
        )
        context_messages = build_context_messages(
            session_manager=session_manager,
            session_id=session.id,
            user_message="Keep this minimal.",
            capability_summary=summary,
            model_profile_name="fast",
        )
    finally:
        session_manager.close()

    capability_note = _extract_capability_note(context_messages)
    assert "Enabled built-in tools: 0" in capability_note
    assert "Available built-in tools:" not in capability_note
    assert "Unavailable capabilities:" not in capability_note
    assert "user-initiated slash commands only" in capability_note


def _build_context_messages_capability_note(
    *,
    project_root,
    profile_name: str,
) -> str:
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)

    try:
        session = session_manager.create_session(make_current=False)
        registry = create_default_tool_registry(
            settings,
            session_manager=session_manager,
        )
        summary = build_runtime_capability_summary(
            tool_registry=registry,
            memory_summary_available=False,
            model_can_call_tools=settings.models[profile_name].tool_mode == "native",
        )
        context_messages = build_context_messages(
            session_manager=session_manager,
            session_id=session.id,
            user_message="Describe the runtime contract.",
            capability_summary=summary,
            model_profile_name=profile_name,
        )
    finally:
        session_manager.close()

    return _extract_capability_note(context_messages)


def _extract_capability_note(context_messages) -> str:  # type: ignore[no-untyped-def]
    return next(
        message.content
        for message in context_messages
        if message.role is LLMRole.SYSTEM
        and "Runtime capability status:" in message.content
    )
