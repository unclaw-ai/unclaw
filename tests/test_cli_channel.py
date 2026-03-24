from __future__ import annotations

import json
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
from unclaw.tools.file_tools import WRITE_TEXT_FILE_DEFINITION
from unclaw.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolPermissionLevel,
    ToolResult,
)
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = pytest.mark.integration


class _CliFakeNonStreamResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> _CliFakeNonStreamResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def read(self) -> bytes:
        return self._body.encode("utf-8")


class _CliFakeStreamResponse:
    def __init__(self, payloads: tuple[dict[str, object], ...]) -> None:
        self._lines = [
            json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
            for payload in payloads
        ]

    def __enter__(self) -> _CliFakeStreamResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def __iter__(self):
        return iter(self._lines)


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


def test_preflight_banner_shows_active_model_pack(make_temp_project) -> None:
    project_root = make_temp_project()
    models_config_path = project_root / "config" / "models.yaml"
    models_config_path.write_text(
        "active_pack: sweet\ndev_profiles: {}\n",
        encoding="utf-8",
    )
    settings = load_settings(project_root=project_root)

    banner = cli_channel._build_preflight_banner(settings)

    assert "PACK" in banner
    assert "sweet" in banner
    assert "CONTROL" in banner
    assert settings.app.security.tools.files.control_preset in banner


def test_cli_shows_model_requested_tool_call_before_final_reply(
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
    call_count = 0
    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolDefinition(
            name="fetch_url_text",
            description="Fetch URL text.",
            permission_level=ToolPermissionLevel.NETWORK,
            arguments={"url": "The URL to fetch"},
        ),
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Example Domain content",
        ),
    )

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
            del messages, timeout_seconds, thinking_enabled, tools
            if call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name=profile.model_name,
                    content="",
                    created_at="2026-03-18T12:00:00Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="fetch_url_text",
                            arguments={"url": "https://example.com"},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fetch_url_text",
                                        "arguments": {"url": "https://example.com"},
                                    }
                                }
                            ],
                        }
                    },
                )
            reply = "The page contains example content."
            if content_callback is not None:
                content_callback(reply)
            return LLMResponse(
                provider="ollama",
                model_name=profile.model_name,
                content=reply,
                created_at="2026-03-18T12:00:01Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    scripted_inputs = iter(["Fetch example.com"])

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
                execute=lambda _tool_call: None,
                registry=tool_registry,
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    tool_line = '[tool] fetch_url_text {"url": "https://example.com"}'
    reply_line = "Unclaw> The page contains example content."
    assert tool_line in output
    assert reply_line in output
    assert output.index(tool_line) < output.index(reply_line)


def test_cli_streamed_native_tool_call_shows_tool_before_final_reply(
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

    tool_registry = ToolRegistry()
    tool_registry.register(
        ToolDefinition(
            name="fetch_url_text",
            description="Fetch URL text.",
            permission_level=ToolPermissionLevel.NETWORK,
            arguments={"url": "The URL to fetch"},
        ),
        lambda call: ToolResult.ok(
            tool_name=call.tool_name,
            output_text="Example Domain content",
        ),
    )

    call_count = 0

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        nonlocal call_count
        del timeout
        payload = json.loads(request.data.decode("utf-8"))

        if not payload.get("stream", False):
            return _CliFakeNonStreamResponse(
                json.dumps(
                    {
                        "model": "qwen3.5:4b",
                        "created_at": "2026-03-20T10:00:00Z",
                        "done_reason": "stop",
                        "message": {"content": '{"route":"chat","search_query":""}'},
                    }
                )
            )

        call_count += 1
        if call_count == 1:
            return _CliFakeStreamResponse(
                (
                    {
                        "model": "qwen3.5:4b",
                        "created_at": "2026-03-20T10:00:01Z",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "fetch_url_text",
                                        "arguments": {"url": "https://example.com"},
                                    }
                                }
                            ],
                        },
                    },
                    {
                        "model": "qwen3.5:4b",
                        "created_at": "2026-03-20T10:00:02Z",
                        "done_reason": "stop",
                    },
                )
            )

        return _CliFakeStreamResponse(
            (
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-20T10:00:03Z",
                    "message": {"content": "The page contains example content."},
                },
                {
                    "model": "qwen3.5:4b",
                    "created_at": "2026-03-20T10:00:04Z",
                    "done_reason": "stop",
                },
            )
        )

    monkeypatch.setattr("unclaw.llm.ollama_provider.urlopen", fake_urlopen)

    scripted_inputs = iter(["Fetch example.com"])

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
                execute=lambda _tool_call: None,
                registry=tool_registry,
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert call_count == 2
    tool_line = '[tool] fetch_url_text {"url": "https://example.com"}'
    reply_line = "Unclaw> The page contains example content."
    assert tool_line in output
    assert reply_line in output
    assert output.index(tool_line) < output.index(reply_line)


def test_cli_plain_chat_does_not_print_tool_visibility_without_tool_calls(
    monkeypatch,
    make_temp_project,
    capsys,
) -> None:
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
        memory_manager=SimpleNamespace(
            build_or_refresh_session_summary=lambda _session_id: None
        ),
    )

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
            del profile, messages, timeout_seconds, thinking_enabled, tools
            reply = "Plain answer without tools."
            if content_callback is not None:
                content_callback(reply)
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=reply,
                created_at="2026-03-18T12:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)

    scripted_inputs = iter(["Just answer directly."])

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
                execute=lambda _tool_call: None,
                registry=ToolRegistry(),
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "[tool]" not in output
    assert "Unclaw> Plain answer without tools." in output


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


def test_cli_native_search_can_continue_into_native_write_tool_loop(
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

    reformulated_query = "Marine Leleu recent profile"
    output_path = project_root / "marine-leleu.txt"
    file_contents = "Marine Leleu is a French endurance athlete and content creator."
    search_calls: list[ToolCall] = []
    write_calls: list[ToolCall] = []
    responder_call_count = 0

    tool_registry = ToolRegistry()

    def _search_tool(call: ToolCall) -> ToolResult:
        search_calls.append(call)
        return ToolResult.ok(
            tool_name="search_web",
            output_text=(
                f"Search query: {call.arguments['query']}\n"
                "Sources fetched: 2 of 2 attempted\n"
                "Evidence kept: 4\n"
            ),
            payload={
                "query": call.arguments["query"],
                "summary_points": [
                    "Marine Leleu is a French endurance athlete and content creator."
                ],
                "display_sources": [
                    {
                        "title": "Marine Leleu",
                        "url": "https://example.com/marine-leleu",
                    },
                    {
                        "title": "Athlete Profile",
                        "url": "https://example.com/athletes/marine-leleu",
                    },
                ],
            },
        )

    def _write_tool(call: ToolCall) -> ToolResult:
        write_calls.append(call)
        output_path.write_text(str(call.arguments["content"]), encoding="utf-8")
        return ToolResult.ok(
            tool_name="write_text_file",
            output_text=f"Wrote text file: {output_path}",
            payload={"path": str(output_path)},
        )

    tool_registry.register(SEARCH_WEB_DEFINITION, _search_tool)
    tool_registry.register(WRITE_TEXT_FILE_DEFINITION, _write_tool)

    class FakeOrchestratorProvider:
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
            del profile, messages, timeout_seconds, thinking_enabled, tools
            nonlocal responder_call_count
            responder_call_count += 1
            if responder_call_count == 1:
                return LLMResponse(
                    provider="ollama",
                    model_name="qwen3.5:4b",
                    content="",
                    created_at="2026-03-20T10:00:01Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="search_web",
                            arguments={"query": reformulated_query},
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "search_web",
                                        "arguments": {
                                            "query": reformulated_query,
                                        },
                                    }
                                }
                            ],
                        }
                    },
                )

            if responder_call_count == 2:
                return LLMResponse(
                    provider="ollama",
                    model_name="qwen3.5:4b",
                    content="",
                    created_at="2026-03-20T10:00:02Z",
                    finish_reason="stop",
                    tool_calls=(
                        ToolCall(
                            tool_name="write_text_file",
                            arguments={
                                "path": str(output_path),
                                "content": file_contents,
                            },
                        ),
                    ),
                    raw_payload={
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "write_text_file",
                                        "arguments": {
                                            "path": str(output_path),
                                            "content": file_contents,
                                        },
                                    }
                                }
                            ],
                        }
                    },
                )

            reply = "I saved a short briefing to marine-leleu.txt."
            if content_callback is not None:
                content_callback(reply)
            return LLMResponse(
                provider="ollama",
                model_name="qwen3.5:4b",
                content=reply,
                created_at="2026-03-20T10:00:03Z",
                finish_reason="stop",
            )

    monkeypatch.setattr(
        "unclaw.core.orchestrator.OllamaProvider",
        FakeOrchestratorProvider,
    )
    monkeypatch.setattr(
        "unclaw.core.runtime.create_default_tool_registry",
        lambda settings_arg, session_manager=None: tool_registry,
    )

    scripted_inputs = iter(
        [
            "Search the web for Marine Leleu, save a short summary to marine-leleu.txt, then tell me what you saved."
        ]
    )

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
                execute=lambda _tool_call: None,
                registry=tool_registry,
            ),
        )
    finally:
        session_manager.close()

    output = capsys.readouterr().out
    assert exit_code == 0
    assert responder_call_count == 3
    assert search_calls == [
        ToolCall(tool_name="search_web", arguments={"query": reformulated_query})
    ]
    assert write_calls == [
        ToolCall(
            tool_name="write_text_file",
            arguments={"path": str(output_path), "content": file_contents},
        )
    ]
    assert output_path.read_text(encoding="utf-8") == file_contents
    tool_line = (
        f'[tool] write_text_file {{"content": "{file_contents}", '
        f'"path": "{output_path}"}}'
    )
    reply_line = "Unclaw> I saved a short briefing to marine-leleu.txt."
    assert tool_line in output
    assert output.index(tool_line) < output.index(reply_line)
    assert "Unclaw> I saved a short briefing to marine-leleu.txt." in output
    assert "Sources:\n- Marine Leleu: https://example.com/marine-leleu" in output
    assert "- Athlete Profile: https://example.com/athletes/marine-leleu" in output
    assert "[answer refined]" not in output
