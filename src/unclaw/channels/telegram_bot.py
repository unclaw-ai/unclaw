"""Polling-based Telegram channel for the Unclaw runtime."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unclaw.bootstrap import bootstrap
from unclaw.channels.telegram_api import TelegramApiClient, TelegramApiError
from unclaw.channels.telegram_app import TelegramAppDependencies, start_telegram_app
from unclaw.channels.telegram_config import (
    TelegramChatSessionStore,
    TelegramConfig,
    allow_telegram_chat,
    build_unauthorized_chat_message,
    find_latest_rejected_chat_id,
    format_authorized_chat_count,
    format_telegram_access_mode,
    load_telegram_config,
    print_authorized_chat_list,
    revoke_telegram_chat,
)
from unclaw.channels.telegram_formatting import (
    MESSAGE_LIMIT,
    NON_TEXT_MESSAGE_REPLY,
    RATE_LIMITED_CHAT_MESSAGE,
    format_command_result,
    format_tool_list,
    format_tool_result,
    normalize_telegram_command,
    read_message_timestamp,
    split_message_chunks,
)
from unclaw.channels.telegram_management import (
    TelegramManagementDependencies,
    run_management_command,
)
from unclaw.core.command_handler import CommandHandler, CommandResult
from unclaw.core.executor import ToolExecutor
from unclaw.core.research_flow import (
    is_search_tool_call,
    persist_tool_result,
    run_search_command,
)
from unclaw.core.runtime import run_user_turn
from unclaw.core.session_manager import SessionManager
from unclaw.core.timing import elapsed_ms
from unclaw.errors import UnclawError
from unclaw.local_secrets import resolve_telegram_bot_token
from unclaw.logs.event_bus import EventBus
from unclaw.logs.tracer import Tracer
from unclaw.memory import MemoryManager
from unclaw.memory.protocols import (
    SessionMemoryChannelInterface,
    SessionMemorySummaryRefresher,
)
from unclaw.schemas.chat import MessageRole
from unclaw.schemas.session import SessionRecord
from unclaw.settings import Settings
from unclaw.startup import build_banner, build_startup_report, format_startup_report

LOGGER = logging.getLogger(__name__)
_POLL_RETRY_DELAY_SECONDS = 3.0
_MAX_PENDING_MESSAGES_PER_CHAT = 2

# Backward-compatible aliases for names that moved to focused Telegram modules.
# These are referenced by tests and other modules via telegram_bot.<name>.
_MESSAGE_LIMIT = MESSAGE_LIMIT
_RATE_LIMITED_CHAT_MESSAGE = RATE_LIMITED_CHAT_MESSAGE
_format_telegram_access_mode = format_telegram_access_mode
_print_authorized_chat_list = print_authorized_chat_list
_format_authorized_chat_count = format_authorized_chat_count
_build_unauthorized_chat_message = build_unauthorized_chat_message
_read_message_timestamp = read_message_timestamp
_normalize_telegram_command = normalize_telegram_command
_format_command_result = format_command_result
_format_tool_list = format_tool_list
_format_tool_result = format_tool_result
_split_message_chunks = split_message_chunks
_run_management_command = run_management_command
_start_telegram_app = start_telegram_app


@dataclass(slots=True)
class TelegramChatRateLimitState:
    """Track queued messages for one chat across polling batches."""

    last_reply_sent_at: int | None = None
    pending_messages_since_reply: int = 0


@dataclass(slots=True)
class TelegramBotChannel:
    """Bridge Telegram messages into the shared local runtime."""

    settings: Settings
    config: TelegramConfig
    session_manager: SessionManager
    memory_manager: SessionMemoryChannelInterface
    tracer: Tracer
    tool_executor: ToolExecutor
    api_client: TelegramApiClient
    session_store: TelegramChatSessionStore
    command_handlers: dict[int, CommandHandler] = field(default_factory=dict)
    rate_limit_states: dict[int, TelegramChatRateLimitState] = field(
        default_factory=dict
    )
    max_pending_messages_per_chat: int = _MAX_PENDING_MESSAGES_PER_CHAT
    clock: Callable[[], float] = time.time

    def run(self) -> None:
        bot_profile = self.api_client.get_me()
        username = bot_profile.get("username", "<unknown>")
        self.tracer.trace_channel_started(
            channel_name="telegram",
            model_profile_name=self.settings.app.default_model_profile,
            extra_payload={
                "polling_timeout_seconds": self.config.polling_timeout_seconds,
                "username": username,
            },
        )
        LOGGER.info(
            "Telegram polling started for @%s with timeout=%ss access=%s",
            username,
            self.config.polling_timeout_seconds,
            format_telegram_access_mode(self.config),
        )

        next_update_offset: int | None = None
        while True:
            try:
                updates = self.api_client.get_updates(
                    offset=next_update_offset,
                    timeout_seconds=self.config.polling_timeout_seconds,
                )
            except TelegramApiError as exc:
                LOGGER.error("Telegram polling failed: %s", exc)
                time.sleep(_POLL_RETRY_DELAY_SECONDS)
                continue

            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    next_update_offset = update_id + 1

                try:
                    self._handle_update(update)
                except TelegramApiError as exc:
                    LOGGER.error("Telegram transport failed for one update: %s", exc)
                except Exception:
                    LOGGER.exception("Unexpected Telegram update failure.")

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return

        chat = message.get("chat")
        if not isinstance(chat, dict):
            return

        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            return

        if not self.config.is_chat_allowed(chat_id):
            self.tracer.trace_telegram_chat_rejected(
                chat_id=chat_id,
                reason="unauthorized",
            )
            LOGGER.warning("Rejected Telegram message from unauthorized chat=%s", chat_id)
            self._send_reply(
                chat_id=chat_id,
                text=build_unauthorized_chat_message(chat_id),
            )
            return

        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send_reply(
                chat_id=chat_id,
                text=NON_TEXT_MESSAGE_REPLY,
            )
            return

        message_timestamp = _read_message_timestamp(message)
        if self._is_rate_limited(
            chat_id=chat_id,
            message_timestamp=message_timestamp,
        ):
            return

        active_session = self._activate_chat_session(chat_id)
        command_handler = self._get_command_handler(chat_id)
        LOGGER.info(
            "Handling Telegram message for chat=%s session=%s",
            chat_id,
            active_session.id,
        )

        normalized_text = text.strip()
        self.tracer.trace_telegram_message_received(
            session_id=active_session.id,
            chat_id=chat_id,
            text_length=len(normalized_text),
            is_command=normalized_text.startswith("/"),
        )
        if normalized_text.startswith("/"):
            self._handle_command(
                chat_id=chat_id,
                text=normalized_text,
                session_id=active_session.id,
                command_handler=command_handler,
            )
            return

        self._handle_chat_turn(
            chat_id=chat_id,
            text=normalized_text,
            session_id=active_session.id,
            command_handler=command_handler,
        )

    def _activate_chat_session(self, chat_id: int) -> SessionRecord:
        mapped_session_id = self.session_store.get_session_id(chat_id)
        if mapped_session_id is not None:
            mapped_session = self.session_manager.load_session(mapped_session_id)
            if mapped_session is not None:
                active_session = self.session_manager.switch_session(mapped_session.id)
                self.tracer.trace_session_selected(
                    session_id=active_session.id,
                    title=active_session.title,
                    reason="telegram_chat",
                )
                return active_session

        session = self.session_manager.create_session(title=f"Telegram chat {chat_id}")
        self.session_store.bind_chat(chat_id=chat_id, session_id=session.id)
        self.tracer.trace_session_started(
            session_id=session.id,
            title=session.title,
            source="telegram_chat",
        )
        return session

    def _get_command_handler(self, chat_id: int) -> CommandHandler:
        handler = self.command_handlers.get(chat_id)
        if handler is not None:
            return handler

        handler = CommandHandler(
            settings=self.settings,
            session_manager=self.session_manager,
            memory_manager=self.memory_manager,
            tracer=self.tracer,
            allow_exit=False,
        )
        self.command_handlers[chat_id] = handler
        return handler

    def _handle_command(
        self,
        *,
        chat_id: int,
        text: str,
        session_id: str,
        command_handler: CommandHandler,
    ) -> None:
        normalized_command = _normalize_telegram_command(text)
        if normalized_command == "/start" or normalized_command.startswith("/start "):
            normalized_command = "/help"

        result = command_handler.handle(normalized_command)
        bound_session_id = self._resolve_command_session_id(
            fallback_session_id=session_id,
            result=result,
        )
        self._persist_chat_binding(chat_id, session_id=bound_session_id)

        if result.list_tools:
            reply_text = _format_tool_list(self.tool_executor.list_tools())
        elif result.tool_call is not None:
            if is_search_tool_call(result.tool_call):
                reply_text = run_search_command(
                    session_manager=self.session_manager,
                    command_handler=command_handler,
                    tracer=self.tracer,
                    tool_call=result.tool_call,
                    tool_registry=self.tool_executor.registry,
                ).assistant_reply
                _refresh_memory_summary(
                    memory_manager=self.memory_manager,
                    session_id=bound_session_id,
                )
            else:
                self.tracer.trace_tool_started(
                    session_id=bound_session_id,
                    tool_name=result.tool_call.tool_name,
                    arguments=result.tool_call.arguments,
                )
                tool_started_at = time.perf_counter()
                tool_result = self.tool_executor.execute(result.tool_call)
                self.tracer.trace_tool_finished(
                    session_id=bound_session_id,
                    tool_name=result.tool_call.tool_name,
                    success=tool_result.success,
                    output_length=len(tool_result.output_text),
                    error=tool_result.error,
                    tool_duration_ms=elapsed_ms(tool_started_at),
                )
                persist_tool_result(
                    session_manager=self.session_manager,
                    session_id=bound_session_id,
                    result=tool_result,
                    tool_call=result.tool_call,
                )
                reply_text = _format_tool_result(tool_result)
        elif result.should_exit:
            reply_text = "The Telegram bot keeps running. Use /help to see commands."
        else:
            reply_text = _format_command_result(result)

        if reply_text:
            self._send_reply(chat_id=chat_id, text=reply_text)

    def _handle_chat_turn(
        self,
        *,
        chat_id: int,
        text: str,
        session_id: str,
        command_handler: CommandHandler,
    ) -> None:
        self.session_manager.add_message(
            MessageRole.USER,
            text,
            session_id=session_id,
        )
        assistant_reply = run_user_turn(
            session_manager=self.session_manager,
            command_handler=command_handler,
            user_input=text,
            tracer=self.tracer,
        )
        _refresh_memory_summary(
            memory_manager=self.memory_manager,
            session_id=session_id,
        )

        self._persist_chat_binding(chat_id, session_id=session_id)
        self._send_reply(chat_id=chat_id, text=assistant_reply)

    def _persist_chat_binding(self, chat_id: int, *, session_id: str) -> None:
        if not session_id.strip():
            return
        self.session_store.bind_chat(chat_id=chat_id, session_id=session_id)

    def _resolve_command_session_id(
        self,
        *,
        fallback_session_id: str,
        result: CommandResult,
    ) -> str:
        if isinstance(result.session_id, str) and result.session_id.strip():
            return result.session_id

        if fallback_session_id.strip():
            return fallback_session_id

        current_session_id = getattr(self.session_manager, "current_session_id", None)
        if isinstance(current_session_id, str) and current_session_id.strip():
            return current_session_id

        return fallback_session_id

    def _is_rate_limited(
        self,
        *,
        chat_id: int,
        message_timestamp: int | None,
    ) -> bool:
        state = self._get_rate_limit_state(chat_id)
        if (
            message_timestamp is None
            or state.last_reply_sent_at is None
            or message_timestamp >= state.last_reply_sent_at
        ):
            state.pending_messages_since_reply = 0
            return False

        pending_messages = state.pending_messages_since_reply + 1
        if pending_messages <= self.max_pending_messages_per_chat:
            state.pending_messages_since_reply = pending_messages
            return False

        self.tracer.trace_telegram_rate_limited(
            chat_id=chat_id,
            pending_messages=pending_messages,
            max_pending_messages=self.max_pending_messages_per_chat,
        )
        LOGGER.warning(
            "Rejected Telegram burst for chat=%s pending=%s max=%s",
            chat_id,
            pending_messages,
            self.max_pending_messages_per_chat,
        )
        self._send_reply(chat_id=chat_id, text=_RATE_LIMITED_CHAT_MESSAGE)
        return True

    def _get_rate_limit_state(self, chat_id: int) -> TelegramChatRateLimitState:
        state = self.rate_limit_states.get(chat_id)
        if state is None:
            state = TelegramChatRateLimitState()
            self.rate_limit_states[chat_id] = state
        return state

    def _mark_reply_sent(self, chat_id: int) -> None:
        state = self._get_rate_limit_state(chat_id)
        state.last_reply_sent_at = int(self.clock())

    def _send_reply(self, *, chat_id: int, text: str) -> None:
        sent_any = False
        for chunk in _split_message_chunks(text):
            self.api_client.send_message(chat_id=chat_id, text=chunk)
            sent_any = True
        if sent_any:
            self._mark_reply_sent(chat_id)


def main(
    project_root: Path | None = None,
    *,
    command: str = "start",
    chat_id: int | None = None,
) -> int:
    """Run the Telegram polling channel."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if command != "start":
        return _run_management_command(
            project_root=project_root,
            command=command,
            chat_id=chat_id,
            dependencies=TelegramManagementDependencies(
                bootstrap=bootstrap,
                allow_telegram_chat=allow_telegram_chat,
                revoke_telegram_chat=revoke_telegram_chat,
                find_latest_rejected_chat_id=find_latest_rejected_chat_id,
                load_telegram_config=load_telegram_config,
                print_authorized_chat_list=print_authorized_chat_list,
                format_authorized_chat_count=format_authorized_chat_count,
            ),
        )

    return _start_telegram_app(
        project_root=project_root,
        dependencies=TelegramAppDependencies(
            bootstrap=bootstrap,
            load_telegram_config=load_telegram_config,
            build_startup_report=build_startup_report,
            build_banner=build_banner,
            format_startup_report=format_startup_report,
            format_telegram_access_mode=format_telegram_access_mode,
            resolve_telegram_bot_token=resolve_telegram_bot_token,
            session_manager_factory=SessionManager,
            memory_manager_factory=MemoryManager,
            event_bus_factory=EventBus,
            tracer_factory=Tracer,
            tool_executor_factory=ToolExecutor,
            api_client_factory=TelegramApiClient,
            session_store_factory=TelegramChatSessionStore,
            channel_factory=TelegramBotChannel,
        ),
    )


def _refresh_memory_summary(
    *,
    memory_manager: SessionMemorySummaryRefresher,
    session_id: str,
) -> None:
    if not isinstance(memory_manager, SessionMemorySummaryRefresher):
        return

    try:
        memory_manager.build_or_refresh_session_summary(session_id)
    except UnclawError as exc:
        LOGGER.warning("Could not refresh session summary for %s: %s", session_id, exc)


if __name__ == "__main__":
    raise SystemExit(main())
