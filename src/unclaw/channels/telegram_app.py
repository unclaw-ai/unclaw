"""Startup wiring for the Telegram channel."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from unclaw.errors import ConfigurationError, UnclawError


@dataclass(frozen=True, slots=True)
class TelegramAppDependencies:
    """Factories and callables needed to boot the Telegram channel."""

    bootstrap: Callable[..., Any]
    load_telegram_config: Callable[..., Any]
    build_startup_report: Callable[..., Any]
    build_banner: Callable[..., str]
    format_startup_report: Callable[..., str]
    format_telegram_access_mode: Callable[..., str]
    resolve_telegram_bot_token: Callable[..., Any]
    session_manager_factory: Any
    memory_manager_factory: Callable[..., Any]
    event_bus_factory: Callable[[], Any]
    tracer_factory: Callable[..., Any]
    tool_executor_factory: Any
    api_client_factory: Callable[..., Any]
    session_store_factory: Callable[..., Any]
    channel_factory: Callable[..., Any]


def start_telegram_app(
    *,
    project_root: Path | None,
    dependencies: TelegramAppDependencies,
) -> int:
    """Bootstrap and run the Telegram polling channel."""

    session_manager: Any = None
    bot: Any = None
    try:
        settings = dependencies.bootstrap(project_root=project_root)
        telegram_config = dependencies.load_telegram_config(settings)
        startup_report = dependencies.build_startup_report(
            settings,
            channel_name="telegram",
            channel_enabled=settings.app.channels.telegram_enabled,
            required_profile_names=(settings.app.default_model_profile,),
            optional_profile_names=_build_optional_profile_names(settings),
            telegram_token_env_var=telegram_config.bot_token_env_var,
            telegram_allowed_chat_ids=telegram_config.allowed_chat_ids,
        )
        print(
            dependencies.build_banner(
                title="Unclaw Telegram",
                subtitle="Local-first bot channel backed by your local model runtime.",
                rows=(
                    ("mode", "telegram"),
                    (
                        "default",
                        (
                            f"{settings.app.default_model_profile} -> "
                            f"{settings.default_model.model_name}"
                        ),
                    ),
                    ("logging", settings.app.logging.mode),
                    ("polling", f"{telegram_config.polling_timeout_seconds}s"),
                    (
                        "access",
                        dependencies.format_telegram_access_mode(telegram_config),
                    ),
                ),
            )
        )
        print(dependencies.format_startup_report(startup_report))
        if not telegram_config.allowed_chat_ids:
            print(
                "Tip: when your own Telegram chat is rejected, run "
                "`unclaw telegram allow-latest` on this machine."
            )
        if startup_report.has_errors:
            return 1

        resolved_bot_token = dependencies.resolve_telegram_bot_token(
            settings,
            bot_token_env_var=telegram_config.bot_token_env_var,
        )
        if resolved_bot_token is None:
            raise ConfigurationError(
                "Telegram bot token is missing. "
                "Run `unclaw onboard` and paste it into the local project secrets "
                "file, or use the advanced fallback "
                f"{telegram_config.bot_token_env_var} environment variable."
            )

        session_manager = dependencies.session_manager_factory.from_settings(settings)
        memory_manager = dependencies.memory_manager_factory(
            session_manager=session_manager
        )
        event_bus = dependencies.event_bus_factory()
        tracer = dependencies.tracer_factory(
            event_bus=event_bus,
            event_repository=session_manager.event_repository,
            include_reasoning_text=settings.app.logging.include_reasoning_text,
        )
        tracer.runtime_log_path = (
            settings.paths.log_file_path if settings.app.logging.file_enabled else None
        )
        tracer.trace_model_profile_selected(
            session_id=None,
            model_profile_name=settings.app.default_model_profile,
            provider=settings.default_model.provider,
            model_name=settings.default_model.model_name,
            reason="startup",
        )
        tool_executor = dependencies.tool_executor_factory.with_default_tools(settings)
        api_client = dependencies.api_client_factory(bot_token=resolved_bot_token.value)
        session_store = dependencies.session_store_factory(session_manager.connection)
        session_store.initialize()

        bot = dependencies.channel_factory(
            settings=settings,
            config=telegram_config,
            session_manager=session_manager,
            memory_manager=memory_manager,
            tracer=tracer,
            tool_executor=tool_executor,
            api_client=api_client,
            session_store=session_store,
            event_bus=event_bus,
            session_manager_factory=dependencies.session_manager_factory,
            memory_manager_factory=dependencies.memory_manager_factory,
            tracer_factory=dependencies.tracer_factory,
            tool_executor_factory=dependencies.tool_executor_factory,
            session_store_factory=dependencies.session_store_factory,
        )
        bot.run()
        return 0
    except KeyboardInterrupt:
        print("\nStopping Unclaw Telegram bot.")
        return 0
    except UnclawError as exc:
        print(f"Failed to start Unclaw Telegram bot: {exc}", file=sys.stderr)
        return 1
    finally:
        if bot is not None and hasattr(bot, "close"):
            bot.close()
        if session_manager is not None:
            session_manager.close()


def _build_optional_profile_names(settings: Any) -> tuple[str, ...]:
    return tuple(
        profile_name
        for profile_name in settings.models
        if profile_name != settings.app.default_model_profile
    )
