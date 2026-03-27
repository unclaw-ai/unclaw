from __future__ import annotations

from types import SimpleNamespace

import pytest
import builtins

from unclaw.channels import cli as cli_channel
from unclaw.channels.cli import _TerminalAssistantStream, run_cli

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
        lambda **kwargs: "Mission goal: demo\nCurrent active task: none\nCompleted tasks: none\nBlocked task: none\nNext expected action or evidence: none",
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
    assert "Mission goal: demo" in out
    assert "Exiting Unclaw." in out
