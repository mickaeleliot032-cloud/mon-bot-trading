"""Persistance locale de l'état et journal CSV du paper trading."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


def new_state(day: date, capital: float) -> dict[str, Any]:
    return {
        "date": day.isoformat(),
        "capital": round(capital, 2),
        "daily_start_capital": round(capital, 2),
        "entry_taken": False,
        "position": None,
        "last_scan_slot": None,
        "score_history": {},
        "alerted_levels": {},
        "last_ranking": [],
        "summary_sent": False,
    }


class StateStore:
    def __init__(self, state_path: Path, journal_path: Path):
        self.state_path = state_path
        self.journal_path = journal_path

    def load(self, day: date, initial_capital: float) -> dict[str, Any]:
        if not self.state_path.exists():
            return new_state(day, initial_capital)
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return new_state(day, initial_capital)
        if state.get("date") == day.isoformat():
            return state
        capital = float(state.get("capital", initial_capital))
        return new_state(day, capital)

    def save(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def append_trade(self, trade: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.journal_path.exists()
        fields = (
            "date",
            "ticker",
            "name",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "score",
            "reason",
            "gross_return_pct",
            "fees_eur",
            "net_pnl_eur",
            "capital_after_eur",
        )
        with self.journal_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({field: trade.get(field, "") for field in fields})
