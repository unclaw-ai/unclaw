"""Polling-based Telegram channel for the Unclaw runtime."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
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
_PENDING_UPDATE_WAIT_INTERVAL_SECONDS = 0.01

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
class _TelegramChatWorker:
    """Serialize one Telegram chat on its own background thread."""

    chat_id: int
    channel_factory: Callable[[], TelegramBotChannel]
    _channel: TelegramBotChannel | None = field(default=None, init=False, repr=False)
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"telegram-chat-{self.chat_id}",
        )

    def submit_update(self, update: dict[str, Any]) -> Future[None]:
        return self._executor.submit(self._run_update, update)

    def close(self) -> None:
        if self._channel is not None:
            try:
                self._executor.submit(self._channel.close).result()
            except Exception:
                LOGGER.exception(
                    "Failed to close Telegram chat worker for chat=%s.",
                    self.chat_id,
                )
        self._executor.shutdown(wait=True)

    def _run_update(self, update: dict[str, Any]) -> None:
        if self._channel is None:
            self._channel = self.channel_factory()
        self._channel._handle_update(update)


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
    event_bus: EventBus | None = None
    session_manager_factory: type[SessionManager] | None = None
    memory_manager_factory: Callable[..., SessionMemoryChannelInterface] | None = None
    tracer_factory: Callable[..., Tracer] | None = None
    tool_executor_factory: Any | None = None
    session_store_factory: Callable[..., TelegramChatSessionStore] | None = None
    command_handlers: dict[int, CommandHandler] = field(default_factory=dict)
    rate_limit_states: dict[int, TelegramChatRateLimitState] = field(
        default_factory=dict
    )
    chat_workers: dict[int, _TelegramChatWorker] = field(default_factory=dict)
    max_pending_messages_per_chat: int = _MAX_PENDING_MESSAGES_PER_CHAT
    clock: Callable[[], float] = time.time
    owns_session_manager: bool = False
    _chat_workers_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _pending_update_futures: set[Future[None]] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _pending_update_futures_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

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

        if self._supports_isolated_chat_workers():
            self._run_polling_loop_concurrent()
            return

        self._run_polling_loop_sync()

    def close(self) -> None:
        if self.chat_workers:
            self._wait_for_pending_updates(timeout_seconds=None)
            self._close_chat_workers()
        if self.owns_session_manager:
            self.session_manager.close()

    def _run_polling_loop_sync(self) -> None:
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

    def _run_polling_loop_concurrent(self) -> None:
        next_update_offset: int | None = None
        try:
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

                    chat_id = self._read_update_chat_id(update)
                    if chat_id is None or not self.config.is_chat_allowed(chat_id):
                        try:
                            self._handle_update(update)
                        except TelegramApiError as exc:
                            LOGGER.error(
                                "Telegram transport failed for one update: %s",
                                exc,
                            )
                        except Exception:
                            LOGGER.exception("Unexpected Telegram update failure.")
                        continue

                    self._submit_update(chat_id=chat_id, update=update)
        finally:
            self._wait_for_pending_updates(timeout_seconds=None)
            self._close_chat_workers()

    def _read_update_chat_id(self, update: dict[str, Any]) -> int | None:
        message = update.get("message")
        if not isinstance(message, dict):
            return None

        chat = message.get("chat")
        if not isinstance(chat, dict):
            return None

        chat_id = chat.get("id")
        return chat_id if isinstance(chat_id, int) else None

    def _submit_update(self, *, chat_id: int, update: dict[str, Any]) -> None:
        worker = self._get_or_create_chat_worker(chat_id)
        future = worker.submit_update(update)
        with self._pending_update_futures_lock:
            self._pending_update_futures.add(future)
        future.add_done_callback(
            lambda completed, chat_id=chat_id: self._finalize_update_future(
                chat_id=chat_id,
                future=completed,
            )
        )

    def _get_or_create_chat_worker(self, chat_id: int) -> _TelegramChatWorker:
        with self._chat_workers_lock:
            worker = self.chat_workers.get(chat_id)
            if worker is not None:
                return worker

            worker = _TelegramChatWorker(
                chat_id=chat_id,
                channel_factory=lambda chat_id=chat_id: self._build_chat_worker_channel(
                    chat_id
                ),
            )
            self.chat_workers[chat_id] = worker
            return worker

    def _build_chat_worker_channel(self, chat_id: int) -> TelegramBotChannel:
        del chat_id

        if not self._supports_isolated_chat_workers():
            return self

        assert self.session_manager_factory is not None
        assert self.memory_manager_factory is not None
        assert self.tracer_factory is not None
        assert self.tool_executor_factory is not None
        assert self.session_store_factory is not None

        session_manager = self.session_manager_factory.from_settings(self.settings)
        memory_manager = self.memory_manager_factory(session_manager=session_manager)
        tracer = self.tracer_factory(
            event_bus=self.event_bus or EventBus(),
            event_repository=session_manager.event_repository,
            include_reasoning_text=self.tracer.include_reasoning_text,
        )
        tracer.runtime_log_path = self.tracer.runtime_log_path
        tool_executor = self.tool_executor_factory.with_default_tools(self.settings)
        session_store = self.session_store_factory(session_manager.connection)
        session_store.initialize()

        return TelegramBotChannel(
            settings=self.settings,
            config=self.config,
            session_manager=session_manager,
            memory_manager=memory_manager,
            tracer=tracer,
            tool_executor=tool_executor,
            api_client=self.api_client,
            session_store=session_store,
            max_pending_messages_per_chat=self.max_pending_messages_per_chat,
            clock=self.clock,
            owns_session_manager=True,
        )

    def _supports_isolated_chat_workers(self) -> bool:
        return all(
            factory is not None
            for factory in (
                self.session_manager_factory,
                self.memory_manager_factory,
                self.tracer_factory,
                self.tool_executor_factory,
                self.session_store_factory,
            )
        )

    def _finalize_update_future(
        self,
        *,
        chat_id: int,
        future: Future[None],
    ) -> None:
        try:
            future.result()
        except TelegramApiError as exc:
            LOGGER.error("Telegram transport failed for chat=%s: %s", chat_id, exc)
        except Exception:
            LOGGER.exception(
                "Unexpected Telegram update failure for chat=%s.",
                chat_id,
            )
        finally:
            with self._pending_update_futures_lock:
                self._pending_update_futures.discard(future)

    def _wait_for_pending_updates(
        self,
        *,
        timeout_seconds: float | None,
    ) -> None:
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        while True:
            with self._pending_update_futures_lock:
                if not self._pending_update_futures:
                    return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out while waiting for Telegram updates.")
            time.sleep(_PENDING_UPDATE_WAIT_INTERVAL_SECONDS)

    def _close_chat_workers(self) -> None:
        with self._chat_workers_lock:
            workers = tuple(self.chat_workers.values())
            self.chat_workers.clear()
        for worker in workers:
            worker.close()

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
