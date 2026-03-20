from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from unclaw.bootstrap import bootstrap
from unclaw.core.session_manager import SessionManager
from unclaw.llm.base import LLMProviderError
from unclaw.schemas.events import EventLevel
from unclaw.settings import load_settings
from unclaw.startup import CheckStatus, OllamaStatus, build_banner, build_startup_report


def test_startup_report_warns_for_missing_optional_models(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(
                settings.default_model.model_name,
                settings.router.model_name,
            ),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
        optional_profile_names=tuple(
            profile_name
            for profile_name in settings.models
            if profile_name != settings.app.default_model_profile
        ),
    )

    assert report.has_errors is False
    assert any(
        check.status is CheckStatus.WARN and check.label == "Extra models"
        for check in report.checks
    )


def test_startup_report_errors_when_required_model_is_missing(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
    )

    assert report.has_errors is True
    assert any(
        check.status is CheckStatus.ERROR and check.label == "Models"
        for check in report.checks
    )


def test_startup_report_ignores_missing_router_model_in_default_preflight(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
    )

    assert report.has_errors is False
    assert all(check.label != "Router" for check in report.checks)


def test_startup_report_warm_loads_default_model_when_requested(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(
                settings.default_model.model_name,
                settings.router.model_name,
            ),
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
            del base_url
            captured["default_timeout_seconds"] = default_timeout_seconds

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
            del content_callback
            captured["profile_name"] = profile.name
            captured["model_name"] = profile.model_name
            captured["messages"] = [
                (str(message.role), message.content) for message in messages
            ]
            captured["timeout_seconds"] = timeout_seconds
            captured["thinking_enabled"] = thinking_enabled
            captured["tools"] = tools
            return object()

    monkeypatch.setattr("unclaw.startup.OllamaProvider", FakeOllamaProvider)

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
        warm_default_model=True,
    )

    warm_check = next(check for check in report.checks if check.label == "Warm-load")

    assert warm_check.status is CheckStatus.OK
    assert (
        captured["default_timeout_seconds"]
        == settings.app.providers.ollama.timeout_seconds
    )
    assert captured["profile_name"] == settings.app.default_model_profile
    assert captured["model_name"] == settings.default_model.model_name
    assert captured["messages"] == [("user", " ")]
    assert (
        captured["timeout_seconds"] == settings.app.providers.ollama.timeout_seconds
    )
    assert captured["thinking_enabled"] is False
    assert captured["tools"] is None


def test_startup_report_warm_load_failure_is_non_fatal(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=True,
            model_names=(settings.default_model.model_name,),
        ),
    )

    class FailingOllamaProvider:
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
            del (
                profile,
                messages,
                timeout_seconds,
                thinking_enabled,
                content_callback,
                tools,
            )
            raise LLMProviderError("warm-load timeout")

    monkeypatch.setattr("unclaw.startup.OllamaProvider", FailingOllamaProvider)

    report = build_startup_report(
        settings,
        channel_name="terminal",
        channel_enabled=True,
        required_profile_names=(settings.app.default_model_profile,),
        warm_default_model=True,
    )

    warm_check = next(check for check in report.checks if check.label == "Warm-load")

    assert report.has_errors is False
    assert report.summary_status is CheckStatus.WARN
    assert warm_check.status is CheckStatus.WARN
    assert "warm-load timeout" in warm_check.detail
    assert "Startup will continue without preloading." in warm_check.detail


def test_startup_report_accepts_local_telegram_token(
    monkeypatch,
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    (project_root / "config" / "secrets.yaml").write_text(
        yaml.safe_dump(
            {
                "telegram": {
                    "bot_token": "123456789:AAExampleTelegramBotTokenValue",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=False,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="telegram",
        channel_enabled=True,
        required_profile_names=(),
        telegram_token_env_var="TELEGRAM_BOT_TOKEN",
    )

    token_check = next(check for check in report.checks if check.label == "Telegram token")
    assert token_check.status is CheckStatus.OK
    assert "local file" in token_check.detail


def test_startup_report_suggests_local_telegram_allow_commands(monkeypatch) -> None:
    settings = load_settings(project_root=_repo_root())

    monkeypatch.setattr(
        "unclaw.startup.inspect_ollama",
        lambda timeout_seconds=1.5: OllamaStatus(
            cli_path="/usr/bin/ollama",
            is_installed=True,
            is_running=False,
            model_names=(),
        ),
    )

    report = build_startup_report(
        settings,
        channel_name="telegram",
        channel_enabled=True,
        required_profile_names=(),
        telegram_allowed_chat_ids=frozenset(),
    )

    access_check = next(check for check in report.checks if check.label == "Telegram access")
    assert access_check.status is CheckStatus.WARN
    assert access_check.guidance is not None
    assert "allow-latest" in access_check.guidance


def test_build_banner_centers_brand_tagline() -> None:
    banner = build_banner(
        title="Onboarding",
        subtitle="Guided local setup.",
        rows=(("mode", "setup"),),
        use_color=False,
    )

    tagline_line = next(
        line for line in banner.splitlines() if "Local-first AI, no cloud claws" in line
    )
    content = tagline_line[2:-2]
    left_padding = len(content) - len(content.lstrip(" "))
    right_padding = len(content) - len(content.rstrip(" "))

    assert abs(left_padding - right_padding) <= 1


def test_bootstrap_prunes_runtime_traces_older_than_retention_window(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)
    session_id: str
    old_created_at = _utc_iso_days_ago(45)
    recent_created_at = _utc_iso_days_ago(2)

    session_manager = SessionManager.from_settings(settings)
    try:
        session = session_manager.create_session(make_current=False)
        session_id = session.id
        session_manager.event_repository.add_event(
            session_id=session.id,
            event_type="trace.old",
            level=EventLevel.INFO,
            message="Old trace",
            created_at=old_created_at,
        )
        session_manager.event_repository.add_event(
            session_id=session.id,
            event_type="trace.recent",
            level=EventLevel.INFO,
            message="Recent trace",
            created_at=recent_created_at,
        )
    finally:
        session_manager.close()

    log_path = settings.paths.log_file_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        _runtime_log_line(
            event_type="trace.old",
            created_at=old_created_at,
            message="Old trace",
        )
        + "\n"
        + _runtime_log_line(
            event_type="trace.recent",
            created_at=recent_created_at,
            message="Recent trace",
        )
        + "\n",
        encoding="utf-8",
    )

    bootstrap(project_root=project_root)

    reloaded_settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(reloaded_settings)
    try:
        persisted_events = session_manager.event_repository.list_recent_events(
            session_id,
            limit=10,
        )
    finally:
        session_manager.close()

    assert [event.event_type for event in persisted_events] == ["trace.recent"]
    assert persisted_events[0].created_at == recent_created_at
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        _runtime_log_line(
            event_type="trace.recent",
            created_at=recent_created_at,
            message="Recent trace",
        )
    ]


def test_bootstrap_keeps_old_runtime_traces_when_retention_is_disabled(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    _set_logging_value(project_root, "retention_days", 0)
    settings = load_settings(project_root=project_root)
    old_created_at = _utc_iso_days_ago(45)

    session_manager = SessionManager.from_settings(settings)
    try:
        session = session_manager.create_session(make_current=False)
        session_manager.event_repository.add_event(
            session_id=session.id,
            event_type="trace.old",
            level=EventLevel.INFO,
            message="Old trace",
            created_at=old_created_at,
        )
        session_id = session.id
    finally:
        session_manager.close()

    log_path = settings.paths.log_file_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_log_line = _runtime_log_line(
        event_type="trace.old",
        created_at=old_created_at,
        message="Old trace",
    )
    log_path.write_text(old_log_line + "\n", encoding="utf-8")

    bootstrap(project_root=project_root)

    reloaded_settings = load_settings(project_root=project_root)
    session_manager = SessionManager.from_settings(reloaded_settings)
    try:
        persisted_events = session_manager.event_repository.list_recent_events(
            session_id,
            limit=10,
        )
    finally:
        session_manager.close()

    assert [event.event_type for event in persisted_events] == ["trace.old"]
    assert log_path.read_text(encoding="utf-8").splitlines() == [old_log_line]


def test_session_manager_initialization_creates_database_with_secure_permissions_on_posix(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    session_manager = SessionManager.from_settings(settings)
    try:
        assert settings.paths.database_path.exists()
    finally:
        session_manager.close()

    if os.name == "posix":
        assert stat.S_IMODE(settings.paths.database_path.stat().st_mode) == 0o600


def test_bootstrap_rehardens_existing_database_permissions_on_posix(
    make_temp_project,
) -> None:
    project_root = make_temp_project()
    settings = load_settings(project_root=project_root)

    session_manager = SessionManager.from_settings(settings)
    session_manager.close()

    if os.name == "posix":
        settings.paths.database_path.chmod(0o644)
        assert stat.S_IMODE(settings.paths.database_path.stat().st_mode) == 0o644

    bootstrap(project_root=project_root)

    assert settings.paths.database_path.exists()
    if os.name == "posix":
        assert stat.S_IMODE(settings.paths.database_path.stat().st_mode) == 0o600


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _set_logging_value(project_root: Path, key: str, value: object) -> None:
    app_config_path = project_root / "config" / "app.yaml"
    payload = yaml.safe_load(app_config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    logging_payload = payload.get("logging")
    assert isinstance(logging_payload, dict)
    logging_payload[key] = value
    app_config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _utc_iso_days_ago(days: int) -> str:
    return (
        datetime.now(tz=UTC) - timedelta(days=days)
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _runtime_log_line(*, event_type: str, created_at: str, message: str) -> str:
    return json.dumps(
        {
            "created_at": created_at,
            "event_type": event_type,
            "level": "info",
            "message": message,
            "payload": {},
            "session_id": None,
        },
        sort_keys=True,
    )
