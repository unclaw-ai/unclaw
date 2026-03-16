from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_terminal_assistant_stream_explains_interrupted_streams(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Partial")
    stream.finish("Saved reply")

    assert capsys.readouterr().out == (
        "Unclaw> Partial\n"
        "[stream interrupted; showing the saved final reply]\n"
        "Unclaw> Saved reply\n"
    )


def test_cli_search_returns_a_natural_reply_with_compact_sources(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
    _set_profile_tool_mode(settings, "main", tool_mode="native")
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
    assert "Search query:" not in output
    assert "Sources fetched:" not in output
    assert "Evidence kept:" not in output
    assert "Sources:\n- Ollama Blog: https://ollama.com/blog/search-update" in output
    assert "- Release Notes: https://example.com/releases/ollama-search" in output


def test_cli_search_non_native_uses_runtime_tool_path_without_channel_preexecution(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    project_root = _create_temp_project(tmp_path)
    settings = load_settings(project_root=project_root)
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
    assert "Search query:" not in output
    assert "Sources fetched:" not in output
    assert "Evidence kept:" not in output
    assert "Sources:\n- Ollama Blog: https://ollama.com/blog/search-update" in output
    assert "- Release Notes: https://example.com/releases/ollama-search" in output
    assert captured_messages


def _create_temp_project(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    project_root = tmp_path / "project"
    shutil.copytree(source_root / "config", project_root / "config")
    return project_root


def _set_profile_tool_mode(settings, profile_name: str, *, tool_mode: str) -> None:
    profile = settings.models[profile_name]
    settings.models[profile_name] = profile.__class__(
        name=profile.name,
        provider=profile.provider,
        model_name=profile.model_name,
        temperature=profile.temperature,
        thinking_supported=profile.thinking_supported,
        tool_mode=tool_mode,
    )
