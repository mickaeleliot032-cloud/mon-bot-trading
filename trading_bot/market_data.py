"""Accès résilient aux données de marché Yahoo Finance."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


class MarketDataClient:
    def download_universe(
        self,
        tickers: Iterable[str],
        period: str = "5d",
        interval: str = "5m",
    ) -> dict[str, pd.DataFrame]:
        symbols = list(dict.fromkeys(tickers))
        if not symbols:
            return {}
        raw = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
            prepost=False,
        )
        if raw.empty:
            return {}
        return {
            ticker: frame
            for ticker in symbols
            if not (frame := self._extract(raw, ticker, len(symbols))).empty
        }

    def latest_price(self, ticker: str) -> float | None:
        try:
            frames = self.download_universe([ticker], period="1d", interval="1m")
            frame = frames.get(ticker)
            if frame is None or frame.empty:
                return None
            return float(frame["Close"].dropna().iloc[-1])
        except Exception:
            LOGGER.exception("Impossible de récupérer le dernier prix de %s", ticker)
            return None

    @staticmethod
    def _extract(raw: pd.DataFrame, ticker: str, symbol_count: int) -> pd.DataFrame:
        if not isinstance(raw.columns, pd.MultiIndex):
            return raw.copy() if symbol_count == 1 else pd.DataFrame()

        first_level = raw.columns.get_level_values(0)
        second_level = raw.columns.get_level_values(1)
        if ticker in first_level:
            return raw[ticker].copy()
        if ticker in second_level:
            return raw.xs(ticker, axis=1, level=1).copy()
        return pd.DataFrame()
