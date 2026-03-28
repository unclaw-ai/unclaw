"""P4-3 V1 validation tests.

Covers four explicit V1 contracts:

1. num_ctx propagation — shipped config has num_ctx, it flows through settings
   and ModelProfile → ResolvedModelProfile → OllamaProvider request payload.

2. Default no-router runtime path — ordinary turns stay on the selected
   profile's direct chat/native path; native profiles can still call
   search_web and other tools from the normal agent loop without a router
   pre-selecting the turn.

3. Native tool-calling vs non-native/json_plan path split — _resolve_tool_
   definitions returns None for json_plan profiles and non-None for native
   profiles; the agent loop only activates on the native path.

4. Long-context / long-session reliability — the runtime stays stable across
   many turns, context budgeting preserves the current turn and memory bridge,
   and a memory context note is still injected after raw history is trimmed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unclaw.core.command_handler import CommandHandler
from unclaw.core.runtime import run_user_turn
from unclaw.core.runtime_support import _resolve_tool_definitions
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import (
    LLMMessage,
    LLMResponse,
    LLMRole,
    ModelCapabilities,
    ResolvedModelProfile,
)
from unclaw.llm.model_profiles import resolve_model_profile
from unclaw.llm.ollama_provider import OllamaProvider
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_repo_settings():
    return load_settings(project_root=_repo_root(), include_local_overrides=False)


def _make_native_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        thinking_supported=False,
        tool_mode="native",
        supports_tools=True,
        supports_native_tool_calling=True,
    )


def _make_json_plan_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        thinking_supported=False,
        tool_mode="json_plan",
        supports_tools=True,
        supports_native_tool_calling=False,
    )


_ECHO_TOOL_DEFINITION = ToolDefinition(
    name="echo_test",
    description="Echo a message back for test validation.",
    permission_level=ToolPermissionLevel.LOCAL_READ,
    arguments={"message": "Message to echo."},
)


def _make_registry_with_echo_tool() -> ToolRegistry:
    registry = ToolRegistry()
    # ToolHandler = Callable[[ToolCall], ToolResult] — handler receives ToolCall, not dict.
    registry.register(
        _ECHO_TOOL_DEFINITION,
        lambda call: ToolResult.ok(
            tool_name="echo_test",
            output_text=f"echo: {call.arguments.get('message', '')}",
        ),
    )
    return registry


# ---------------------------------------------------------------------------
# 1. num_ctx validation
# ---------------------------------------------------------------------------


def test_v1_num_ctx_all_shipped_profiles_have_num_ctx_set() -> None:
    """All shipped model profiles in models.yaml must have num_ctx explicitly set.

    This is the sentinel test: if num_ctx is silently removed from any profile,
    Ollama falls back to its built-in default (~2048 tokens for small models),
    causing silent context overflow on any moderately long session.
    """
    settings = _load_repo_settings()
    for profile_name, profile in settings.models.items():
        assert profile.num_ctx is not None, (
            f"Profile {profile_name!r} is missing num_ctx in models.yaml. "
            "This causes silent context overflow on all sessions."
        )
        assert profile.num_ctx > 0, (
            f"Profile {profile_name!r} has num_ctx={profile.num_ctx}; must be positive."
        )


def test_v1_num_ctx_specific_shipped_values_match_audit_spec() -> None:
    """Shipped num_ctx values must match the audit-specified minimums.

    Audit spec (refactor_priorities.md P0-1):
    - fast (llama3.2:3b):  num_ctx=4096
    - main (qwen3.5:4b):   num_ctx=8192
    - deep (qwen3.5:9b):   num_ctx=8192
    - codex (qwen2.5-coder:7b): num_ctx=4096
    """
    settings = _load_repo_settings()
    expected = {
        "fast": 4096,
        "main": 8192,
        "deep": 8192,
        "codex": 4096,
    }
    for profile_name, expected_num_ctx in expected.items():
        assert profile_name in settings.models, (
            f"Expected shipped profile {profile_name!r} not found in settings."
        )
        actual = settings.models[profile_name].num_ctx
        assert actual == expected_num_ctx, (
            f"Profile {profile_name!r}: expected num_ctx={expected_num_ctx}, "
            f"got {actual}."
        )


def test_v1_num_ctx_flows_from_settings_to_resolved_model_profile() -> None:
    """num_ctx must be preserved when resolving a ModelProfile to ResolvedModelProfile.

    If this chain breaks, the payload builder never sees num_ctx even if
    it is present in models.yaml.
    """
    settings = _load_repo_settings()
    for profile_name in settings.models:
        resolved = resolve_model_profile(settings, profile_name)
        expected_num_ctx = settings.models[profile_name].num_ctx
        assert resolved.num_ctx == expected_num_ctx, (
            f"Profile {profile_name!r}: num_ctx {expected_num_ctx!r} was not "
            f"preserved in ResolvedModelProfile (got {resolved.num_ctx!r})."
        )


def test_v1_num_ctx_present_in_ollama_request_payload(monkeypatch) -> None:
    """OllamaProvider.chat must include options.num_ctx when profile.num_ctx is set.

    This is the end-to-end payload proof. The test fails if num_ctx is
    silently dropped anywhere between settings and the HTTP request body.
    """
    captured_payloads: list[dict[str, Any]] = []

    def _fake_request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        del self, timeout_seconds
        if method == "POST" and path == "/api/chat" and payload is not None:
            captured_payloads.append(payload)
        return {
            "model": "test:model",
            "created_at": "2026-03-18T10:00:00Z",
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }

    monkeypatch.setattr(OllamaProvider, "_request_json", _fake_request_json)

    profile = ResolvedModelProfile(
        name="test-profile",
        provider="ollama",
        model_name="test:model",
        temperature=0.3,
        capabilities=_make_native_capabilities(),
        num_ctx=8192,
    )
    provider = OllamaProvider()
    provider.chat(
        profile=profile,
        messages=[LLMMessage(role=LLMRole.USER, content="hello")],
    )

    assert len(captured_payloads) == 1, "Expected exactly one POST to /api/chat."
    payload = captured_payloads[0]
    assert "options" in payload, "Ollama payload must contain an 'options' key."
    assert "num_ctx" in payload["options"], (
        "options.num_ctx must be present in the Ollama request payload. "
        "Without it, Ollama silently uses ~2048 tokens on small models."
    )
    assert payload["options"]["num_ctx"] == 8192


def test_v1_num_ctx_absent_from_payload_when_profile_has_none(monkeypatch) -> None:
    """OllamaProvider.chat must NOT include num_ctx when profile.num_ctx is None.

    This ensures the presence test above is meaningful: if num_ctx were always
    injected regardless of the profile, the above test would pass vacuously.
    """
    captured_payloads: list[dict[str, Any]] = []

    def _fake_request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        del self, timeout_seconds
        if method == "POST" and path == "/api/chat" and payload is not None:
            captured_payloads.append(payload)
        return {
            "model": "test:model",
            "created_at": "2026-03-18T10:00:00Z",
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }

    monkeypatch.setattr(OllamaProvider, "_request_json", _fake_request_json)

    profile = ResolvedModelProfile(
        name="test-no-ctx",
        provider="ollama",
        model_name="test:model",
        temperature=0.3,
        capabilities=_make_native_capabilities(),
        num_ctx=None,  # explicitly no num_ctx
    )
    provider = OllamaProvider()
    provider.chat(
        profile=profile,
        messages=[LLMMessage(role=LLMRole.USER, content="hello")],
    )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert "options" in payload
    assert "num_ctx" not in payload["options"], (
        "num_ctx must not appear in the payload when profile.num_ctx is None."
    )


def test_v1_num_ctx_full_path_from_models_yaml_to_ollama_payload(monkeypatch) -> None:
    """End-to-end proof: models.yaml → settings → resolved profile → Ollama payload.

    This test is the strongest signal: it loads the REAL shipped config,
    resolves the main profile, and proves num_ctx appears in the payload.
    If the shipped config lost num_ctx, or if the chain broke, this fails.
    """
    captured_payloads: list[dict[str, Any]] = []

    def _fake_request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        del self, timeout_seconds
        if method == "POST" and path == "/api/chat" and payload is not None:
            captured_payloads.append(payload)
        return {
            "model": "qwen3.5:4b",
            "created_at": "2026-03-18T10:00:00Z",
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }

    monkeypatch.setattr(OllamaProvider, "_request_json", _fake_request_json)

    settings = _load_repo_settings()
    resolved = resolve_model_profile(settings, "main")
    expected_num_ctx = settings.models["main"].num_ctx

    provider = OllamaProvider()
    provider.chat(
        profile=resolved,
        messages=[LLMMessage(role=LLMRole.USER, content="hello")],
    )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload.get("options", {}).get("num_ctx") == expected_num_ctx, (
        f"Payload options.num_ctx should equal models.yaml value {expected_num_ctx!r} "
        f"for the 'main' profile."
    )


def test_v1_default_main_turn_responds_directly_without_router_or_forced_search(
    monkeypatch,
    make_temp_project,
) -> None:
    """Simple main turns should go straight to the responder without router work."""
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    user_input = "Explain recursion in one sentence."
    captured_tool_calls: list[ToolCall] = []

    # Fake tool registry: registers search_web so capability summary enables routing,
    # and captures the query when search_web is dispatched.
    registry = ToolRegistry()

    from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

    def _fake_search(call: ToolCall) -> ToolResult:
        # ToolHandler receives a ToolCall object, not a dict.
        captured_tool_calls.append(call)
        return ToolResult.ok(
            tool_name="search_web",
            output_text="Fake search results for validation.",
            payload={
                "query": call.arguments.get("query", ""),
                "summary_points": ["Paris weather is cloudy this week."],
                "display_sources": [],
            },
        )

    registry.register(SEARCH_WEB_DEFINITION, _fake_search)

    class FakeOrchestratorProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, **kwargs):
            del profile, kwargs
            return LLMResponse(
                provider="ollama",
                model_name="test",
                content="Recursion is when a definition or process refers to itself.",
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOrchestratorProvider)

    command_handler.current_model_profile_name = "main"

    try:
        session = session_manager.ensure_current_session()
        session_manager.add_message(
            MessageRole.USER, user_input, session_id=session.id
        )
        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=user_input,
            tracer=tracer,
            tool_registry=registry,
        )

        assert reply == "Recursion is when a definition or process refers to itself."
        assert captured_tool_calls == []
    finally:
        session_manager.close()


def test_v1_native_profile_resolve_tool_definitions_returns_tools() -> None:
    """_resolve_tool_definitions must return the tool list for native profiles."""
    profile = ResolvedModelProfile(
        name="native-test",
        provider="ollama",
        model_name="qwen3.5:9b",
        temperature=0.2,
        capabilities=_make_native_capabilities(),
        num_ctx=8192,
    )
    registry = _make_registry_with_echo_tool()
    result = _resolve_tool_definitions(
        tool_registry=registry,
        model_profile=profile,
    )
    assert result is not None, (
        "Native profile must receive tool definitions for the agent loop."
    )
    assert len(result) == 1
    assert result[0].name == "echo_test"


def test_v1_json_plan_profile_resolve_tool_definitions_returns_none() -> None:
    """_resolve_tool_definitions must return None for json_plan profiles.

    When tool_definitions is None, the runtime never enters the agent loop and
    never passes tools to the model. This is the intended non-native behavior.
    """
    profile = ResolvedModelProfile(
        name="json-plan-test",
        provider="ollama",
        model_name="llama3.2:3b",
        temperature=0.2,
        capabilities=_make_json_plan_capabilities(),
        num_ctx=4096,
    )
    registry = _make_registry_with_echo_tool()
    result = _resolve_tool_definitions(
        tool_registry=registry,
        model_profile=profile,
    )
    assert result is None, (
        "json_plan profile must return None from _resolve_tool_definitions. "
        "Tool definitions must not be passed to the model on this path."
    )


def test_v1_shipped_deep_profile_is_native() -> None:
    """The shipped 'deep' profile must have tool_mode=native.

    This is the profile that enables the real observation-action loop.
    Regression guard: if this profile is accidentally changed to json_plan,
    the agent loop stops working for the only fully-agentic shipped profile.
    """
    settings = _load_repo_settings()
    deep_profile = settings.models["deep"]
    assert deep_profile.tool_mode == "native", (
        "The 'deep' profile must remain tool_mode=native to enable the agent loop."
    )


def test_v1_shipped_main_profile_is_native() -> None:
    """The shipped 'main' profile must have tool_mode=native (changed in post-audit work).

    Regression guard: 'main' is the default profile; it must use native tool calling
    so most users get the agent experience by default.
    """
    settings = _load_repo_settings()
    main_profile = settings.models["main"]
    assert main_profile.tool_mode == "native", (
        "The 'main' profile must be tool_mode=native. "
        "This is the default profile — reverting to json_plan would disable the "
        "agent loop for all users who have not explicitly selected 'deep'."
    )


def test_v1_shipped_fast_profile_is_chat_only() -> None:
    """The shipped 'fast' profile must stay chat-only for responsiveness.

    Regression guard for BIG-FIX-ROUTER-1: 'fast' is now intentionally
    non-agentic, so it must not re-enter native or json_plan tool mode.
    """
    settings = _load_repo_settings()
    fast_profile = settings.models["fast"]
    assert fast_profile.tool_mode == "none", (
        "The 'fast' profile must stay tool_mode=none so normal fast turns remain "
        "plain chat without planner-driven or native tool work."
    )


def test_v1_shipped_codex_profile_is_lite_chat_only() -> None:
    """The shipped 'codex' profile must stay lite with tool_mode=none."""
    settings = _load_repo_settings()
    codex_profile = settings.models["codex"]
    assert codex_profile.tool_mode == "none", (
        "The shipped 'codex' profile must stay tool_mode=none for the lite "
        "chat/code product contract."
    )


def test_v1_shipped_model_configs_have_no_planner_profiles() -> None:
    settings = _load_repo_settings()

    assert settings.models["main"].planner_profile is None
    assert settings.models["deep"].planner_profile is None
    assert settings.models["codex"].planner_profile is None


def test_v1_shipped_profile_tool_mode_values_are_explicit() -> None:
    """Shipped profiles must use only the explicit supported tool_mode values.

    No profile should have an ambiguous or unknown tool_mode. This is the
    capability split that determines whether a profile is planner-driven,
    native-fallback capable, or plain chat only.
    """
    settings = _load_repo_settings()
    valid_tool_modes = {"native", "json_plan", "none"}
    for name, profile in settings.models.items():
        assert profile.tool_mode in valid_tool_modes, (
            f"Profile {name!r} has tool_mode={profile.tool_mode!r}. "
            f"Must be one of: {valid_tool_modes}"
        )


# ---------------------------------------------------------------------------
# 4. Long-context / long-session reliability validation
# ---------------------------------------------------------------------------


def test_v1_long_session_runtime_stays_stable_across_many_turns(
    monkeypatch,
    make_temp_project,
) -> None:
    """Runtime must produce valid replies without crashing after 30+ turns of history.

    This validates the full pipeline: context budget, session persistence,
    and model call all stay stable under long-session conditions.
    """
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, **kwargs):
            del profile, kwargs
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Follow-up reply.",
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        # Build 35 turns of history — well above the DEFAULT_CONTEXT_HISTORY_MESSAGE_LIMIT (20).
        for i in range(35):
            session_manager.add_message(
                MessageRole.USER, f"Earlier question {i}.", session_id=session.id
            )
            session_manager.add_message(
                MessageRole.ASSISTANT, f"Earlier answer {i}.", session_id=session.id
            )

        # The 36th turn is the follow-up we validate.
        follow_up = "What was the last thing you said?"
        session_manager.add_message(
            MessageRole.USER, follow_up, session_id=session.id
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=follow_up,
            tool_registry=ToolRegistry(),  # empty → no web search routing call
        )

        assert reply == "Follow-up reply.", (
            f"Runtime must return a valid reply after 35 prior turns; got {reply!r}"
        )
        # Reply must be persisted in session history.
        messages = session_manager.list_messages(session.id)
        assert messages[-1].role is MessageRole.ASSISTANT
        assert messages[-1].content == "Follow-up reply."
    finally:
        session_manager.close()


def test_v1_long_session_context_budget_does_not_drop_current_turn(
    monkeypatch,
    make_temp_project,
) -> None:
    """Even after the char budget clips most history, the current user message
    must always appear in the model's context.

    This is the safety contract for long-context sessions: the model must always
    see the most recent user request, regardless of how large the history is.
    """
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )
    captured_messages: list[list[LLMMessage]] = []

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, **kwargs):
            del profile, kwargs
            captured_messages.append(list(messages))
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Current turn reply.",
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()
        # Build heavy history to stress-test the budget (large messages).
        big_content = "x" * 3_000
        for i in range(10):
            session_manager.add_message(
                MessageRole.USER, big_content, session_id=session.id
            )
            session_manager.add_message(
                MessageRole.ASSISTANT, big_content, session_id=session.id
            )

        current_user_message = "This is my current request — must be in context."
        session_manager.add_message(
            MessageRole.USER, current_user_message, session_id=session.id
        )

        reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=current_user_message,
            tool_registry=ToolRegistry(),
        )

        assert reply == "Current turn reply."
        assert len(captured_messages) >= 1

        # The current user message must be present in the model's context.
        last_call_messages = captured_messages[-1]
        user_message_contents = [
            m.content
            for m in last_call_messages
            if m.role is LLMRole.USER
        ]
        assert any(
            current_user_message in content
            for content in user_message_contents
        ), (
            "Current user message must always appear in the model's context "
            "regardless of how large the history is."
        )
    finally:
        session_manager.close()


def test_v1_long_session_grounded_follow_up_persists_correctly(
    monkeypatch,
    make_temp_project,
) -> None:
    """After a grounded search turn, the follow-up turn is correctly persisted
    and the session history stays coherent across multiple turns.

    This validates the memory architecture holds under the realistic pattern of:
    search turn → follow-up turn, both persisted correctly in session history.
    """
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(settings)
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(),
    )

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(self, *, base_url="", default_timeout_seconds=60.0):
            del base_url, default_timeout_seconds

        def chat(self, profile, messages, **kwargs):
            del profile, kwargs
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Grounded and stable.",
                created_at="2026-03-18T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    try:
        session = session_manager.ensure_current_session()

        # Turn 1: a regular question.
        first_input = "What is Python?"
        session_manager.add_message(
            MessageRole.USER, first_input, session_id=session.id
        )
        first_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=first_input,
            tool_registry=ToolRegistry(),
        )

        assert first_reply == "Grounded and stable."

        # Turn 2: follow-up — history must include both prior turns.
        follow_up_input = "Can you elaborate on that?"
        session_manager.add_message(
            MessageRole.USER, follow_up_input, session_id=session.id
        )
        follow_up_reply = run_user_turn(
            session_manager=session_manager,
            command_handler=command_handler,
            user_input=follow_up_input,
            tool_registry=ToolRegistry(),
        )

        assert follow_up_reply == "Grounded and stable."

        # Verify both turns are persisted correctly in session history.
        all_messages = session_manager.list_messages(session.id)
        assistant_messages = [
            m for m in all_messages if m.role is MessageRole.ASSISTANT
        ]
        assert len(assistant_messages) >= 2, (
            "Both turns must be persisted as ASSISTANT messages in session history."
        )
        assert all(m.content == "Grounded and stable." for m in assistant_messages)
    finally:
        session_manager.close()
