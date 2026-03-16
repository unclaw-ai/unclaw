"""Local Telegram allowlist management commands."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from unclaw.errors import ConfigurationError, UnclawError


@dataclass(frozen=True, slots=True)
class TelegramManagementDependencies:
    """Callables needed for the Telegram management subcommands."""

    bootstrap: Callable[..., Any]
    allow_telegram_chat: Callable[..., Any]
    revoke_telegram_chat: Callable[..., Any]
    find_latest_rejected_chat_id: Callable[..., int | None]
    load_telegram_config: Callable[..., Any]
    print_authorized_chat_list: Callable[..., None]
    format_authorized_chat_count: Callable[[tuple[int, ...]], str]


def run_management_command(
    *,
    project_root: Path | None,
    command: str,
    chat_id: int | None,
    dependencies: TelegramManagementDependencies,
) -> int:
    """Execute a Telegram allowlist management command."""

    try:
        settings = dependencies.bootstrap(project_root=project_root)
        match command:
            case "allow":
                if chat_id is None:
                    raise ConfigurationError(
                        "Telegram allow requires one numeric chat id."
                    )
                update = dependencies.allow_telegram_chat(settings, chat_id)
                if update.was_authorized and not update.file_changed:
                    print(f"Telegram chat {chat_id} is already authorized.")
                elif update.was_authorized:
                    print(
                        "Telegram access list was normalized without adding a new "
                        f"chat. Chat {chat_id} remains authorized."
                    )
                else:
                    print(
                        f"Authorized Telegram chat {chat_id}. "
                        f"{dependencies.format_authorized_chat_count(update.allowed_chat_ids)}"
                    )
                return 0
            case "allow-latest":
                latest_chat_id = dependencies.find_latest_rejected_chat_id(settings)
                if latest_chat_id is None:
                    print(
                        "No rejected Telegram chat is stored locally yet. "
                        "Send one message to the bot first, then rerun "
                        "`unclaw telegram allow-latest`."
                    )
                    return 1
                update = dependencies.allow_telegram_chat(settings, latest_chat_id)
                if update.was_authorized and not update.file_changed:
                    print(
                        f"The latest rejected Telegram chat ({latest_chat_id}) is "
                        "already authorized."
                    )
                else:
                    print(
                        f"Authorized the latest rejected Telegram chat: {latest_chat_id}. "
                        f"{dependencies.format_authorized_chat_count(update.allowed_chat_ids)}"
                    )
                return 0
            case "revoke":
                if chat_id is None:
                    raise ConfigurationError(
                        "Telegram revoke requires one numeric chat id."
                    )
                update = dependencies.revoke_telegram_chat(settings, chat_id)
                if update.was_authorized:
                    print(
                        f"Revoked Telegram chat {chat_id}. "
                        f"{dependencies.format_authorized_chat_count(update.allowed_chat_ids)}"
                    )
                else:
                    print(f"Telegram chat {chat_id} was not authorized.")
                return 0
            case "list":
                config = dependencies.load_telegram_config(settings)
                dependencies.print_authorized_chat_list(
                    config.allowed_chat_ids,
                    latest_rejected_chat_id=dependencies.find_latest_rejected_chat_id(
                        settings
                    ),
                )
                return 0
            case _:
                raise ConfigurationError(
                    f"Unsupported Telegram command '{command}'."
                )
    except UnclawError as exc:
        print(f"Failed to manage Unclaw Telegram access: {exc}", file=sys.stderr)
        return 1
