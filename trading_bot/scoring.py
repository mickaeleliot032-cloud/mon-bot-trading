"""Scoring progressif de la V4, conçu pour limiter les faux blocages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Score:
    ticker: str
    name: str
    sector: str
    final: float
    quantitative: float
    market: float
    sector_score: float
    news: float
    level: str
    eligible: bool
    reasons: tuple[str, ...]
    snapshot: dict[str, Any]
    headlines: tuple[str, ...] = ()


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def quantitative_score(
    snapshot: dict[str, Any],
    market_return_pct: float,
    max_absolute_gap_pct: float,
) -> tuple[float, bool, tuple[str, ...]]:
    """Score technique 0–100 avec pénalités graduelles plutôt que filtres durs."""

    reasons: list[str] = []
    score = 0.0

    if snapshot["price"] > snapshot["ema20"]:
        score += 10
        reasons.append("cours > EMA20")
    if snapshot["ema20"] > snapshot["ema50"]:
        score += 8
        reasons.append("EMA20 > EMA50")
    if snapshot["price"] > snapshot["vwap"]:
        score += 7
        reasons.append("cours > VWAP")

    open_return = snapshot["return_open_pct"]
    score += float(
        np.interp(open_return, [-1.0, 0.0, 0.4, 1.2, 2.0], [0, 5, 13, 18, 15])
    )
    momentum = snapshot["momentum_15m_pct"]
    score += float(np.interp(momentum, [-0.8, 0.0, 0.25, 0.7], [0, 3, 8, 10]))
    if open_return > 0:
        reasons.append(f"momentum séance +{open_return:.2f}%")

    relative_strength = open_return - market_return_pct
    score += float(np.interp(relative_strength, [-1.0, 0.0, 0.6, 1.5], [0, 5, 11, 15]))
    if relative_strength > 0.2:
        reasons.append(f"surperformance CAC +{relative_strength:.2f} pt")

    volume_ratio = snapshot["volume_ratio"]
    score += float(
        np.interp(volume_ratio, [0.4, 0.8, 1.0, 1.5, 2.5], [0, 3, 7, 12, 15])
    )
    if volume_ratio >= 1.2:
        reasons.append(f"volume x{volume_ratio:.1f}")

    current_rsi = snapshot["rsi14"]
    if 50 <= current_rsi <= 68:
        score += 10
        reasons.append(f"RSI sain {current_rsi:.0f}")
    elif 42 <= current_rsi < 50 or 68 < current_rsi <= 74:
        score += 6
    elif 35 <= current_rsi < 42 or 74 < current_rsi <= 80:
        score += 3

    atr_pct = snapshot["atr_pct"]
    if 0.35 <= atr_pct <= 2.5:
        score += 10
    elif 0.2 <= atr_pct <= 3.5:
        score += 5
    else:
        score += 2

    gap = abs(snapshot["gap_pct"])
    eligible = gap <= max_absolute_gap_pct
    if 2.5 < gap <= max_absolute_gap_pct:
        score -= 10
        reasons.append(f"gap pénalisé {snapshot['gap_pct']:+.2f}%")
    elif not eligible:
        reasons.insert(0, f"gap excessif {snapshot['gap_pct']:+.2f}%")

    return _clip(score), eligible, tuple(reasons[:5])


def market_context_score(snapshot: dict[str, Any]) -> float:
    score = 50.0 + snapshot["return_open_pct"] * 18
    if snapshot["price"] > snapshot["ema20"]:
        score += 8
    if snapshot["ema20"] > snapshot["ema50"]:
        score += 7
    return _clip(score)


def sector_context_score(sector_return_pct: float, market_return_pct: float) -> float:
    return _clip(50 + (sector_return_pct - market_return_pct) * 25)


def level_for(
    final_score: float,
    eligible: bool,
    watch_threshold: float,
    signal_threshold: float,
    strong_threshold: float,
) -> str:
    if not eligible:
        return "EXCLU"
    if final_score >= strong_threshold:
        return "FORT"
    if final_score >= signal_threshold:
        return "SIGNAL"
    if final_score >= watch_threshold:
        return "SURVEILLANCE"
    return "NEUTRE"


def combine_score(
    *,
    ticker: str,
    name: str,
    sector: str,
    snapshot: dict[str, Any],
    quantitative: float,
    market: float,
    sector_score: float,
    news: float,
    eligible: bool,
    reasons: tuple[str, ...],
    thresholds: tuple[float, float, float],
    headlines: tuple[str, ...] = (),
) -> Score:
    # La composante technique reste dominante, mais aucun sous-score neutre
    # n'interdit à lui seul une opportunité.
    final = _clip(
        0.75 * quantitative + 0.10 * market + 0.10 * sector_score + 0.05 * news
    )
    level = level_for(final, eligible, *thresholds)
    return Score(
        ticker=ticker,
        name=name,
        sector=sector,
        final=round(final, 2),
        quantitative=round(quantitative, 2),
        market=round(market, 2),
        sector_score=round(sector_score, 2),
        news=round(news, 2),
        level=level,
        eligible=eligible,
        reasons=reasons,
        snapshot=snapshot,
        headlines=headlines,
    )
