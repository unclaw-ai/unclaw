"""Synchronous Telegram Bot API client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from unclaw.errors import UnclawError
from unclaw.local_secrets import sanitize_telegram_text

_TELEGRAM_API_BASE_URL = "https://api.telegram.org"


class TelegramApiError(UnclawError):
    """Raised when the Telegram Bot API request fails."""


@dataclass(slots=True)
class TelegramApiClient:
    """Small synchronous Telegram Bot API client."""

    bot_token: str
    request_timeout_seconds: float = 40.0

    @property
    def api_base_url(self) -> str:
        return f"{_TELEGRAM_API_BASE_URL}/bot{self.bot_token}"

    def get_me(self) -> dict[str, Any]:
        result = self._request("getMe", {})
        if not isinstance(result, dict):
            raise TelegramApiError("Telegram returned an invalid bot profile payload.")
        return result

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset

        result = self._request("getUpdates", payload)
        if not isinstance(result, list):
            raise TelegramApiError("Telegram returned an invalid updates payload.")
        return [update for update in result if isinstance(update, dict)]

    def send_message(self, *, chat_id: int, text: str) -> None:
        self._request(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
            },
        )

    def _request(self, method: str, payload: dict[str, Any]) -> Any:
        request = Request(
            url=f"{self.api_base_url}/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise TelegramApiError(
                self._sanitize_error_message(
                    f"Telegram API request failed with HTTP {exc.code}: "
                    f"{_read_http_error_body(exc)}"
                )
            ) from exc
        except URLError as exc:
            raise TelegramApiError(
                self._sanitize_error_message(
                    f"Could not reach the Telegram API: {exc.reason}"
                )
            ) from exc
        except OSError as exc:
            raise TelegramApiError(
                self._sanitize_error_message(
                    f"Telegram API request failed: {exc}"
                )
            ) from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise TelegramApiError("Telegram returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise TelegramApiError("Telegram returned an invalid response payload.")
        if payload.get("ok") is not True:
            description = payload.get("description")
            if isinstance(description, str) and description.strip():
                raise TelegramApiError(
                    self._sanitize_error_message(description.strip())
                )
            raise TelegramApiError("Telegram returned an unsuccessful response.")

        return payload.get("result")

    def _sanitize_error_message(self, message: str) -> str:
        return sanitize_telegram_text(message, known_token=self.bot_token)


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        raw_body = exc.read().decode("utf-8").strip()
    except OSError:
        return exc.reason or "Unknown Telegram error."

    if not raw_body:
        return exc.reason or "Unknown Telegram error."

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body

    if isinstance(payload, dict):
        description = payload.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
    return raw_body
