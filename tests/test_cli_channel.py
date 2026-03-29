from __future__ import annotations

from types import SimpleNamespace

import pytest
import builtins

from unclaw.channels import cli as cli_channel
from unclaw.channels.cli import (
    _TerminalAssistantStream,
    _should_render_cli_status_line,
    run_cli,
)

pytestmark = pytest.mark.integration


def test_terminal_assistant_stream_finishes_cleanly_when_stream_matches(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.write("Hel")
    stream.write("lo")
    stream.finish("Hello")

    assert capsys.readouterr().out == "Unclaw> Hello\n"


def test_terminal_assistant_stream_suppressed_shows_only_final_answer(capsys) -> None:
    stream = _TerminalAssistantStream()
    stream.suppress_live_output()

    stream.write("draft")
    stream.finish("final answer")

    assert capsys.readouterr().out == "Unclaw> final answer\n"


def test_terminal_assistant_stream_hides_generic_mission_scaffolding(capsys) -> None:
    stream = _TerminalAssistantStream()

    stream.render_status("[mission] single-agent loop active")
    stream.render_status("[mission] active task: reply")
    stream.render_status("[tool] search_web {\"query\": \"Inoxtag\"}")

    assert capsys.readouterr().out == "[tool] search_web {\"query\": \"Inoxtag\"}\n"


def test_cli_status_filter_keeps_compact_useful_lines() -> None:
    assert _should_render_cli_status_line("[mission] single-agent loop active") is False
    assert _should_render_cli_status_line("[mission] mission completed") is False
    assert _should_render_cli_status_line("[tool] search_web {}") is True


def test_cli_status_filter_hides_internal_json_blob_lines() -> None:
    assert _should_render_cli_status_line(
        '[mission] {"mission_action":"continue_existing","task_board":[]}'
    ) is False


def test_run_cli_prints_the_runtime_reply_and_exits_cleanly(
    monkeypatch,
    capsys,
) -> None:
    fake_session_manager = SimpleNamespace(
        ensure_current_session=lambda: SimpleNamespace(id="session-1"),
        add_message=lambda *args, **kwargs: None,
    )
    fake_command_handler = SimpleNamespace(
        handle=lambda _: SimpleNamespace(
            refresh_tool_executor=False,
            list_tools=False,
            tool_call=None,
            should_exit=False,
        )
    )
    fake_memory_manager = SimpleNamespace(refresh_for_session=lambda session_id: None)
    fake_tracer = SimpleNamespace()
    fake_tool_executor = SimpleNamespace(
        registry=SimpleNamespace(get_owner_skill_id=lambda tool_name: None)
    )
    captured_kwargs: dict[str, object] = {}

    inputs = iter(["où en est cette mission ?", KeyboardInterrupt])

    def _fake_input(_prompt: str) -> str:
        value = next(inputs)
        if value is KeyboardInterrupt:
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr(cli_channel, "_build_prompt", lambda command_handler: "> ")
    monkeypatch.setattr(builtins, "input", _fake_input)
    monkeypatch.setattr(
        cli_channel,
        "run_user_turn",
        lambda **kwargs: (
            captured_kwargs.update(kwargs)
            or "Mission in progress: demo. Current task: write the file."
        ),
    )

    exit_code = run_cli(
        session_manager=fake_session_manager,
        command_handler=fake_command_handler,
        memory_manager=fake_memory_manager,
        tracer=fake_tracer,
        tool_executor=fake_tool_executor,
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert captured_kwargs.get("mission_event_callback") is None
    assert "Mission in progress: demo." in out
    assert "[mission]" not in out
    assert "Exiting Unclaw." in out
