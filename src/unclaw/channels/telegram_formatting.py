"""Telegram-specific formatting helpers."""

from __future__ import annotations

from typing import Any

from unclaw.constants import TELEGRAM_MESSAGE_CHARACTER_LIMIT
from unclaw.core.command_handler import CommandResult, CommandStatus
from unclaw.tools.contracts import ToolDefinition, ToolResult

MESSAGE_LIMIT = TELEGRAM_MESSAGE_CHARACTER_LIMIT
NON_TEXT_MESSAGE_REPLY = "Please send a text message or a slash command."
RATE_LIMITED_CHAT_MESSAGE = (
    "Too many messages arrived before the previous reply finished. "
    "Please wait for the next reply, then send another message."
)


def read_message_timestamp(message: dict[str, Any]) -> int | None:
    value = message.get("date")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def normalize_telegram_command(text: str) -> str:
    command, separator, remainder = text.partition(" ")
    command_name, mention_separator, _mention = command.partition("@")
    if not mention_separator:
        return text
    return f"{command_name}{separator}{remainder}".strip()


def format_command_result(result: CommandResult) -> str:
    if not result.lines:
        return ""

    if result.status is CommandStatus.ERROR:
        first_line, *other_lines = result.lines
        lines = [f"Error: {first_line}", *other_lines]
        return "\n".join(lines)

    return "\n".join(result.lines)


def format_tool_list(tools: list[ToolDefinition]) -> str:
    if not tools:
        return "No built-in tools available."

    lines = ["Built-in tools:"]
    for tool in tools:
        lines.append(
            f"- {tool.name} [{tool.permission_level.value}] {tool.description}"
        )
    return "\n".join(lines)


def format_tool_result(result: ToolResult) -> str:
    if result.success:
        return result.output_text

    if not result.output_text:
        return f"Error: {result.error}"

    lines = result.output_text.splitlines() or [result.output_text]
    first_line, *other_lines = lines
    return "\n".join([f"Error: {first_line}", *other_lines])


def split_message_chunks(text: str, *, limit: int = MESSAGE_LIMIT) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    chunks.append(remaining)
    return chunks
