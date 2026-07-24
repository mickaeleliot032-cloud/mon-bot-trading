"""Calcul des indicateurs à partir des chandeliers intraday."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))
    return result.fillna(50.0)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        (
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _localize_index(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize("UTC").tz_convert(timezone)
    return index.tz_convert(timezone)


def _volume_ratio(frame: pd.DataFrame, session: pd.DataFrame) -> float:
    """Compare les dernières barres aux mêmes horaires des jours précédents."""

    if "Volume" not in frame or session.empty:
        return 1.0
    recent = session.tail(3)
    prior = frame.loc[frame.index.date != session.index[-1].date()]
    comparable: list[float] = []
    for timestamp, row in recent.iterrows():
        same_time = prior.loc[
            (prior.index.hour == timestamp.hour)
            & (prior.index.minute == timestamp.minute),
            "Volume",
        ]
        baseline = float(same_time.median()) if not same_time.empty else np.nan
        if baseline > 0:
            comparable.append(float(row["Volume"]) / baseline)
    if comparable:
        return float(np.clip(np.mean(comparable), 0.0, 5.0))
    baseline = float(prior["Volume"].median()) if not prior.empty else 0.0
    if baseline <= 0:
        return 1.0
    return float(np.clip(recent["Volume"].mean() / baseline, 0.0, 5.0))


def build_snapshot(
    frame: pd.DataFrame,
    now: datetime,
    timezone: str = "Europe/Paris",
) -> dict[str, Any] | None:
    """Transforme un historique 5 minutes en photographie exploitable."""

    required = {"Open", "High", "Low", "Close"}
    if frame.empty or not required.issubset(frame.columns):
        return None

    clean = frame.copy().dropna(subset=list(required)).sort_index()
    if len(clean) < 30:
        return None
    clean.index = _localize_index(pd.DatetimeIndex(clean.index), timezone)

    local_now = now.astimezone(ZoneInfo(timezone))
    session_mask = clean.index.date == local_now.date()
    session = clean.loc[session_mask]
    if session.empty:
        return None

    prior = clean.loc[~session_mask]
    current = float(session["Close"].iloc[-1])
    open_price = float(session["Open"].iloc[0])
    previous_close = float(prior["Close"].iloc[-1]) if not prior.empty else open_price

    ema20 = clean["Close"].ewm(span=20, adjust=False).mean()
    ema50 = clean["Close"].ewm(span=50, adjust=False).mean()
    rsi14 = rsi(clean["Close"])
    atr14 = atr(clean)

    volume = session.get("Volume", pd.Series(0.0, index=session.index)).fillna(0)
    typical_price = (session["High"] + session["Low"] + session["Close"]) / 3
    cumulative_volume = volume.cumsum()
    if float(cumulative_volume.iloc[-1]) > 0:
        vwap = float(
            (typical_price * volume).cumsum().iloc[-1] / cumulative_volume.iloc[-1]
        )
    else:
        vwap = current

    reference_15m = (
        float(session["Close"].iloc[-4]) if len(session) >= 4 else open_price
    )
    return {
        "price": current,
        "open": open_price,
        "previous_close": previous_close,
        "return_open_pct": (current / open_price - 1) * 100,
        "gap_pct": (open_price / previous_close - 1) * 100,
        "momentum_15m_pct": (current / reference_15m - 1) * 100,
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "vwap": vwap,
        "rsi14": float(rsi14.iloc[-1]),
        "atr_pct": float(atr14.iloc[-1] / current * 100),
        "volume_ratio": _volume_ratio(clean, session),
        "bars_today": int(len(session)),
        "timestamp": session.index[-1].isoformat(),
    }
