from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unclaw import main as unclaw_main
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMResponse, LLMRole
from unclaw.schemas.chat import MessageRole
from unclaw.settings import load_settings
from unclaw.startup import OllamaStatus
from unclaw.tools.contracts import ToolCall, ToolResult
from unclaw.tools.registry import ToolRegistry
from unclaw.tools.web_tools import SEARCH_WEB_DEFINITION

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def _is_grounded_finalizer_call(messages: list[object]) -> bool:
    if not messages:
        return False
    first_content = getattr(messages[0], "content", "")
    return isinstance(first_content, str) and first_content.startswith(
        "Grounded reply finalizer for one runtime turn."
    )


def test_start_entrypoint_bootstraps_runtime_and_leaves_a_usable_session(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = _patch_ready_ollama(monkeypatch, project_root)

    monkeypatch.setattr("builtins.input", _raise_eof)

    exit_code = unclaw_main.main(["--project-root", str(project_root), "start"])

    assert exit_code == 0
    assert all(path.exists() for path in settings.paths.runtime_directories())
    assert settings.paths.database_path.exists()

    session_manager = SessionManager.from_settings(settings)
    try:
        session = session_manager.ensure_current_session()
        assert session.id
        assert session_manager.list_sessions(limit=1)
        assert session_manager.list_messages(session.id) == []
    finally:
        session_manager.close()


def test_cli_plain_chat_turn_runs_through_start_path_and_persists_reply(
    monkeypatch,
    make_temp_project,
    capsys,
) -> None:
    project_root = make_temp_project()
    settings = _patch_ready_ollama(monkeypatch, project_root)
    scripted_inputs = iter(["Hello from the terminal."])

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
            del profile, timeout_seconds, thinking_enabled, tools
            reply = "Hello from the local model."
            if content_callback is not None:
                content_callback(reply)
            return LLMResponse(
                provider="ollama",
                model_name=settings.default_model.model_name,
                content=reply,
                created_at="2026-03-16T10:00:00Z",
                finish_reason="stop",
            )

    monkeypatch.setattr("unclaw.core.orchestrator.OllamaProvider", FakeOllamaProvider)
    monkeypatch.setattr("builtins.input", lambda _prompt: _next_input(scripted_inputs))

    exit_code = unclaw_main.main(["--project-root", str(project_root), "start"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Unclaw> Hello from the local model." in output

    session_manager = SessionManager.from_settings(settings)
    try:
        session = session_manager.ensure_current_session()
        messages = session_manager.list_messages(session.id)
        assert [message.role for message in messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert messages[0].content == "Hello from the terminal."
        assert messages[1].content == "Hello from the local model."
    finally:
        session_manager.close()


class _FakeExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> list[object]:
        return []

    def execute(self, _tool_call):  # type: ignore[no-untyped-def]
        raise AssertionError("CLI should not pre-execute /search via ToolExecutor.")


def _patch_ready_ollama(monkeypatch, project_root: Path):
    settings = load_settings(project_root=project_root)
    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )
    return settings


def _build_search_registry(
    *,
    query: str,
    summary_point: str,
    sources: tuple[tuple[str, str], ...],
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        SEARCH_WEB_DEFINITION,
        lambda _call: ToolResult.ok(
            tool_name="search_web",
            output_text=(
                f"Search query: {query}\n"
                f"Sources fetched: {len(sources)} of {len(sources)} attempted\n"
                "Evidence kept: 4\n"
            ),
            payload={
                "query": query,
                "summary_points": [summary_point],
                "display_sources": [
                    {"title": title, "url": url} for title, url in sources
                ],
            },
        ),
    )
    return registry

def _force_main_json_plan(project_root: Path) -> None:
    """Override models.yaml in the temp project to use json_plan for main.

    Used by tests that exercise the non-native (pre-executed) tool path.
    main is native by default after P2-5 shipped; tests that need non-native
    behaviour must call this before bootstrapping the runtime.
    """
    models_yaml = project_root / "config" / "models.yaml"
    config = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    profile_key = "dev_profiles" if "dev_profiles" in config else "profiles"
    config[profile_key]["main"]["tool_mode"] = "json_plan"
    models_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _next_input(scripted_inputs) -> str:  # type: ignore[no-untyped-def]
    try:
        return next(scripted_inputs)
    except StopIteration as exc:
        raise EOFError from exc


def _raise_eof(_prompt: str) -> str:
    raise EOFError
