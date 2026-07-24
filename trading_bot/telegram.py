"""Notifications Telegram sans secret dans le code source."""

from __future__ import annotations

import logging

import requests

LOGGER = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(
        self,
        token: str | None,
        chat_id: str | None,
        timeout_seconds: int = 10,
    ):
        self.token = token
        self.chat_id = chat_id
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled:
            LOGGER.info("Telegram désactivé — %s", message.replace("\n", " | "))
            return False
        try:
            response = self.session.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            LOGGER.exception("Échec de la notification Telegram.")
            return False
