"""Configuration de l'agent, chargée exclusivement depuis l'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class Settings:
    """Paramètres métier de la V4.

    Les valeurs par défaut correspondent au protocole de paper trading validé.
    Elles restent surchargeables par variables d'environnement pour faciliter
    les essais sans modifier le code.
    """

    timezone: str = "Europe/Paris"
    benchmark: str = "^FCHI"

    warmup_time: time = time(9, 5)
    entry_start_time: time = time(9, 15)
    last_scan_time: time = time(16, 0)
    forced_exit_time: time = time(17, 20)
    extended_tp_cutoff: time = time(10, 0)
    scan_interval_minutes: int = 15
    loop_sleep_seconds: int = 30
    position_poll_seconds: int = 60
    scheduled_monitor_window_seconds: int = field(
        default_factory=lambda: _env_int("SCHEDULED_MONITOR_WINDOW_SECONDS", 240)
    )

    watch_threshold: float = 65.0
    signal_threshold: float = 72.0
    strong_threshold: float = 80.0
    max_absolute_gap_pct: float = 4.0
    news_shortlist_size: int = 6

    paper_capital_eur: float = field(
        default_factory=lambda: _env_float("PAPER_CAPITAL_EUR", 1000.0)
    )
    round_trip_fees_eur: float = field(
        default_factory=lambda: _env_float("ROUND_TRIP_FEES_EUR", 2.0)
    )
    base_tp_pct: float = field(default_factory=lambda: _env_float("BASE_TP_PCT", 1.0))
    extended_tp_pct: float = field(
        default_factory=lambda: _env_float("EXTENDED_TP_PCT", 2.0)
    )
    min_sl_pct: float = field(default_factory=lambda: _env_float("MIN_SL_PCT", 0.60))
    max_sl_pct: float = field(default_factory=lambda: _env_float("MAX_SL_PCT", 1.20))
    atr_sl_multiplier: float = field(
        default_factory=lambda: _env_float("ATR_SL_MULTIPLIER", 0.65)
    )
    breakeven_trigger_pct: float = field(
        default_factory=lambda: _env_float("BREAKEVEN_TRIGGER_PCT", 0.65)
    )
    trailing_distance_pct: float = field(
        default_factory=lambda: _env_float("TRAILING_DISTANCE_PCT", 0.45)
    )

    state_path: Path = field(
        default_factory=lambda: Path(os.getenv("STATE_PATH", "data/trading_state.json"))
    )
    journal_path: Path = field(
        default_factory=lambda: Path(os.getenv("JOURNAL_PATH", "data/trades.csv"))
    )
    telegram_token: str | None = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: str | None = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID")
    )
    http_timeout_seconds: int = field(
        default_factory=lambda: _env_int("HTTP_TIMEOUT_SECONDS", 10)
    )

    def validate(self) -> None:
        if self.paper_capital_eur <= 0:
            raise ValueError("PAPER_CAPITAL_EUR doit être strictement positif.")
        if self.round_trip_fees_eur < 0:
            raise ValueError("ROUND_TRIP_FEES_EUR ne peut pas être négatif.")
        if self.scheduled_monitor_window_seconds < 0:
            raise ValueError(
                "SCHEDULED_MONITOR_WINDOW_SECONDS ne peut pas être négatif."
            )
        if not (self.watch_threshold < self.signal_threshold < self.strong_threshold):
            raise ValueError("Les seuils 65/72/80 doivent être croissants.")
        if bool(self.telegram_token) != bool(self.telegram_chat_id):
            raise ValueError(
                "TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID doivent être définis ensemble."
            )
