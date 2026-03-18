from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unclaw.channels import cli as cli_channel
from unclaw.channels.cli import _TerminalAssistantStream
from unclaw.channels.cli import run_cli
from unclaw.core.command_handler import CommandHandler
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.settings import load_settings
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.integration


def test_terminal_assistant_stream_finishes_cleanly_when_stream_matches(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Hel")
    stream.write("lo")
    stream.finish("Hello")

    assert capsys.readouterr().out == "Unclaw> Hello\n"


def test_terminal_assistant_stream_appends_only_missing_suffix(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Saved")
    stream.finish("Saved reply")

    assert capsys.readouterr().out == "Unclaw> Saved reply\n"


def test_terminal_assistant_stream_renders_final_reply_when_nothing_streamed(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.finish("Saved reply")

    assert capsys.readouterr().out == "Unclaw> Saved reply\n"


def test_terminal_assistant_stream_shows_refinement_signal_when_grounding_rewrites_answer(
    capsys,
) -> None:
    """Divergence case: streamed draft differs from grounded final answer.

    This is the P4-4 path: the grounding transform replaced the streamed text
    with a different final answer. The signal must appear and the final answer
    must be printed. Plain chat never reaches this branch because streamed text
    equals the final text when no grounding transform is active.
    """
    stream = _TerminalAssistantStream()

    stream.write("The answer is probably 42.")
    stream.finish("The answer is 37, confirmed by three sources.")

    out = capsys.readouterr().out
    assert "[answer refined]" in out
    assert "The answer is 37, confirmed by three sources." in out
    # "Unclaw>" appears only once — from the streaming prefix, not from the refined answer.
    assert out.count("Unclaw>") == 1


def test_terminal_assistant_stream_no_refinement_signal_for_plain_chat(capsys) -> None:
    """Plain local chat turn: streamed text matches final — no refinement signal."""
    stream = _TerminalAssistantStream()

    stream.write("Hello, how can I help?")
    stream.finish("Hello, how can I help?")

    out = capsys.readouterr().out
    assert "[answer refined]" not in out
    assert "Unclaw> Hello, how can I help?" in out


def test_terminal_assistant_stream_suppressed_shows_only_final_answer_not_draft(
    capsys,
) -> None:
    """P5-3B: suppressed-stream path shows only the final answer, never the draft.

    When suppress_live_output() is called before any writes, chunks are buffered
    silently. finish() renders the final answer once with the refinement signal.
    The draft text must never appear in the output.
    """
    stream = _TerminalAssistantStream()
    stream.suppress_live_output()

    stream.write("Draft: the answer is probably 42.")
    stream.finish("The answer is 37, confirmed by sources.")

    out = capsys.readouterr().out
    assert "Draft" not in out
    assert "probably 42" not in out
    assert "Unclaw> The answer is 37, confirmed by sources." in out
    assert "[answer refined]" in out
    assert out.count("Unclaw>") == 1


def test_terminal_assistant_stream_suppressed_no_signal_when_text_unchanged(
    capsys,
) -> None:
    """P5-3B: suppressed-stream path — no refinement signal when grounding did not change the text."""
    stream = _TerminalAssistantStream()
    stream.suppress_live_output()

    stream.write("Same answer.")
    stream.finish("Same answer.")

    out = capsys.readouterr().out
    assert "[answer refined]" not in out
    assert "Unclaw> Same answer." in out
    assert out.count("Unclaw>") == 1


def test_terminal_main_requests_default_model_warm_load(
    monkeypatch,
    make_temp_project,
    capsys,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_channel, "bootstrap", lambda project_root=None: settings)

    def fake_build_startup_report(settings_arg, **kwargs):  # type: ignore[no-untyped-def]
        assert settings_arg is settings
        captured["warm_default_model"] = kwargs["warm_default_model"]
        return SimpleNamespace(has_errors=True)

    monkeypatch.setattr(
        cli_channel,
        "build_startup_report",
        fake_build_startup_report,
    )
    monkeypatch.setattr(cli_channel, "build_banner", lambda **kwargs: "banner")
    monkeypatch.setattr(cli_channel, "format_startup_report", lambda report: "report")

    assert cli_channel.main(project_root=project_root) == 1
    assert captured["warm_default_model"] is True
    assert capsys.readouterr().out == "banner\nreport\n"


def test_cli_search_returns_a_natural_reply_with_compact_sources(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    capsys,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    set_profile_tool_mode(settings, "main", tool_mode="native")
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(
            build_or_refresh_session_summary=lambda _session_id: None
        ),
    )

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text=(
            "Search query: latest news about Ollama\n"
            "Sources fetched: 2 of 2 attempted\n"
            "Evidence kept: 4\n"
        ),
        payload={
            "query": "latest news about Ollama",
            "summary_points": [
                "Ollama shipped a new update with improved search grounding."
            ],
            "display_sources": [
                {
                    "title": "Ollama Blog",
                    "url": "https://ollama.com/blog/search-update",
                },
                {
                    "title": "Release Notes",
                    "url": "https://example.com/releases/ollama-search",
                },
            ],
        },
    )
    search_registry = ToolRegistry()
    search_registry.register(SEARCH_WEB_DEFINITION, lambda _call: search_tool_result)

    call_count = 0

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            nonlocal call_count
            call_count += 1
            del profile, messages, timeout_seconds, thinking_enabled, tools
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name="qwen3.5:4b",
                    content="",
                    created_at="2026-03-13T12:00:00Z",
                    finish_reason="stop",
                    tool_calls=(ToolCall(
                        tool_name="search_web",
                        arguments={"query": "latest news about Ollama"},
                    ),),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "search_web", "arguments": {"query": "latest news about Ollama"}}}
                            ],
                        }
                    },
                )
            if content_callback is not None:
                content_callback("Ollama shipped a new update with improved search grounding.")
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content="Ollama shipped a new update with improved search grounding.",
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    scripted_inputs = iter(["/search latest news about Ollama"])

    def fake_input(_prompt: str) -> str:
        try:
            return next(scripted_inputs)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", fake_input)

    try:
        exit_code = run_cli(
            session_manager=session_manager,
            command_handler=command_handler,
            memory_manager=SimpleNamespace(
                build_or_refresh_session_summary=lambda _session_id: None
            ),
            tracer=tracer,
            tool_executor=SimpleNamespace(
                list_tools=lambda: [],
                execute=lambda _tool_call: search_tool_result,
                registry=search_registry,
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Unclaw> Ollama shipped a new update with improved search grounding."
        in output
    )
    assert (
        output.count(
            "Unclaw> Ollama shipped a new update with improved search grounding."
        )
        == 1
    )
    assert "[stream interrupted; showing the saved final reply]" not in output
    assert "Search query:" not in output
    assert "Sources fetched:" not in output
    assert "Evidence kept:" not in output
    assert "Sources:\n- Ollama Blog: https://ollama.com/blog/search-update" in output
    assert "- Release Notes: https://example.com/releases/ollama-search" in output


def test_cli_search_non_native_uses_runtime_tool_path_without_channel_preexecution(
    monkeypatch,
    make_temp_project,
    set_profile_tool_mode,
    capsys,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    # Explicitly pin to non-native (json_plan) — this test covers the non-native
    # runtime tool path. main is now native by default (P2-5 shipped), so we
    # must be explicit about which profile this test exercises.
    set_profile_tool_mode(settings, "main", tool_mode="json_plan")
    session_manager = SessionManager.from_settings(settings)
    tracer = Tracer(
        event_bus=EventBus(),
        event_repository=session_manager.event_repository,
    )
    command_handler = CommandHandler(
        settings=settings,
        session_manager=session_manager,
        memory_manager=SimpleNamespace(
            build_or_refresh_session_summary=lambda _session_id: None
        ),
    )

    search_tool_result = ToolResult.ok(
        tool_name="search_web",
        output_text=(
            "Search query: latest news about Ollama\n"
            "Sources fetched: 2 of 2 attempted\n"
            "Evidence kept: 4\n"
        ),
        payload={
            "query": "latest news about Ollama",
            "summary_points": [
                "Ollama shipped a new update with improved search grounding."
            ],
            "display_sources": [
                {
                    "title": "Ollama Blog",
                    "url": "https://ollama.com/blog/search-update",
                },
                {
                    "title": "Release Notes",
                    "url": "https://example.com/releases/ollama-search",
                },
            ],
        },
    )
    search_registry = ToolRegistry()
    search_registry.register(SEARCH_WEB_DEFINITION, lambda _call: search_tool_result)
    captured_messages: list[list[object]] = []

    class FakeOllamaProvider:
        provider_name = "ollama"

        def __init__(
            self,
            *,
            base_url: str = "http://127.0.0.1:11434",
            default_timeout_seconds: float = 60.0,
        ) -> None:
            del base_url, default_timeout_seconds

        def chat(  # type: ignore[no-untyped-def]
            self,
            profile,
            messages,
            *,
            timeout_seconds=None,
            thinking_enabled=False,
            content_callback=None,
            tools=None,
        ):
            del profile, timeout_seconds, thinking_enabled
            assert tools is None
            captured_messages.append(list(messages))
            assert any(
                message.role is LLMRole.TOOL
                and "Ollama Blog: https://ollama.com/blog/search-update" in message.content
                for message in messages
            )
            reply = "Ollama shipped a new update with improved search grounding."
            if content_callback is not None:
                content_callback(reply)
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=reply,
                created_at="2026-03-13T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    scripted_inputs = iter(["/search latest news about Ollama"])

    def fake_input(_prompt: str) -> str:
        try:
            return next(scripted_inputs)
        except StopIteration as exc:
            raise EOFError from exc

    def unexpected_execute(_tool_call):
        raise AssertionError(
            "CLI channel should not pre-execute /search via ToolExecutor.execute()."
        )

    monkeypatch.setattr("builtins.input", fake_input)

    try:
        exit_code = run_cli(
            session_manager=session_manager,
            command_handler=command_handler,
            memory_manager=SimpleNamespace(
                build_or_refresh_session_summary=lambda _session_id: None
            ),
            tracer=tracer,
            tool_executor=SimpleNamespace(
                list_tools=lambda: [],
                execute=unexpected_execute,
                registry=search_registry,
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "Unclaw> Ollama shipped a new update with improved search grounding."
        in output
    )
    assert (
        output.count(
            "Unclaw> Ollama shipped a new update with improved search grounding."
        )
        == 1
    )
    assert "[stream interrupted; showing the saved final reply]" not in output
    assert "Search query:" not in output
    assert "Sources fetched:" not in output
    assert "Evidence kept:" not in output
    assert "Sources:\n- Ollama Blog: https://ollama.com/blog/search-update" in output
    assert "- Release Notes: https://example.com/releases/ollama-search" in output
    assert captured_messages
