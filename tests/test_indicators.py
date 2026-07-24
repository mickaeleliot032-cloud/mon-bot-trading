from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_bot.indicators import build_snapshot


def _history() -> pd.DataFrame:
    timezone = "Europe/Paris"
    days = pd.date_range("2026-07-20", periods=5, freq="D", tz=timezone)
    indexes = [
        pd.date_range(
            day.replace(hour=9, minute=0),
            periods=36,
            freq="5min",
        )
        for day in days
    ]
    index = indexes[0].append(indexes[1:])
    close = np.linspace(100.0, 105.0, len(index))
    return pd.DataFrame(
        {
            "Open": close - 0.05,
            "High": close + 0.15,
            "Low": close - 0.15,
            "Close": close,
            "Volume": np.full(len(index), 10_000),
        },
        index=index,
    )


def test_build_snapshot_returns_intraday_metrics():
    now = datetime(2026, 7, 24, 11, 55, tzinfo=ZoneInfo("Europe/Paris"))
    snapshot = build_snapshot(_history(), now)

    assert snapshot is not None
    assert snapshot["bars_today"] == 36
    assert snapshot["price"] > snapshot["open"]
    assert snapshot["ema20"] > snapshot["ema50"]
    assert snapshot["atr_pct"] > 0
    assert 0.9 <= snapshot["volume_ratio"] <= 1.1
