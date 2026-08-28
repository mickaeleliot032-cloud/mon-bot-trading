"""Moteur V4.3 : prix d'entrée paper basé sur la cotation 1 minute."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from trading_bot.engine import TradingEngine
from trading_bot.engine_v42 import TradingEngineV42
from trading_bot.indicators import build_snapshot
from trading_bot.scoring import Score
from trading_bot.universe import CAC40

LOGGER = logging.getLogger(__name__)


class TradingEngineV43(TradingEngineV42):
    """Conserve le scoring V4.2 mais enrichit la collecte pour le futur ML."""

    def _notify_level_change(self, leader: Score) -> None:
        """Journalise un signal enrichi sans modifier la logique de trading."""

        levels = {"NEUTRE": 0, "SURVEILLANCE": 1, "SIGNAL": 2, "FORT": 3}
        current = levels.get(leader.level, 0)
        previous = int(
            self.state.setdefault("alerted_levels", {}).get(leader.ticker, 0)
        )
        will_notify = current > previous and current > 0

        # On conserve exactement la notification Telegram et la gestion des
        # niveaux du moteur de base, sans déclencher l'écriture Sheets de V4.2.
        TradingEngine._notify_level_change(self, leader)

        if not will_notify:
            return

        alert_time = datetime.now(self.timezone)
        signal_id = (
            f"{alert_time:%Y%m%d-%H%M%S}-{leader.ticker}-{leader.level}"
        )
        snapshot = leader.snapshot

        # Le benchmark est relu uniquement lors d'un changement de niveau,
        # donc sans ajouter de charge à chaque scan. En cas d'indisponibilité,
        # la collecte du signal continue avec des champs CAC laissés vides.
        perf_cac40: float | str = ""
        surperf_cac40: float | str = ""
        try:
            benchmark_frame = self.market_data.download_universe(
                [self.settings.benchmark]
            ).get(self.settings.benchmark)
            if benchmark_frame is not None:
                benchmark_snapshot = build_snapshot(
                    benchmark_frame,
                    alert_time,
                    self.settings.timezone,
                )
                if benchmark_snapshot is not None:
                    perf_cac40 = round(
                        float(benchmark_snapshot["return_open_pct"]), 3
                    )
                    surperf_cac40 = round(
                        float(snapshot["return_open_pct"]) - float(perf_cac40),
                        3,
                    )
        except Exception as exc:
            LOGGER.warning(
                "Contexte CAC 40 indisponible pour le signal %s : %s",
                leader.ticker,
                exc,
            )

        event = {
            "id_signal": signal_id,
            "time": alert_time.isoformat(),
            "ticker": leader.ticker,
            "name": leader.name,
            "score": round(leader.final, 2),
            "quantitative": round(leader.quantitative, 2),
            "level": leader.level,
            "price": round(float(snapshot["price"]), 4),
            "return_open_pct": round(float(snapshot["return_open_pct"]), 3),
            "ema20": round(float(snapshot["ema20"]), 4),
            "ema50": round(float(snapshot["ema50"]), 4),
            "vwap": round(float(snapshot["vwap"]), 4),
            "volume_ratio": round(float(snapshot["volume_ratio"]), 3),
            "eligible": bool(leader.eligible),
            "reasons": ", ".join(leader.reasons),
            # Variables supplémentaires historisées pour le futur ML.
            "perf_cac40": perf_cac40,
            "surperf_cac40": surperf_cac40,
            "momentum_15m": round(float(snapshot["momentum_15m_pct"]), 3),
            "rsi14": round(float(snapshot["rsi14"]), 2),
            "atr_pct": round(float(snapshot["atr_pct"]), 3),
            "gap_pct": round(float(snapshot["gap_pct"]), 3),
            "secteur": leader.sector,
            "score_marche": round(float(leader.market), 2),
            "score_secteur": round(float(leader.sector_score), 2),
            "score_news": round(float(leader.news), 2),
        }
        self.state.setdefault("alert_history", []).append(event)
        self.state["alert_history"] = self.state["alert_history"][-80:]
        self.state.setdefault("last_signal_id_by_ticker", {})[
            leader.ticker
        ] = signal_id
        self._remember_signal_for_follow_up(alert_time, leader, signal_id)
        self.store.save(self.state)
        self.google_sheets.send("alert", **event)

    def _finalize_signal_tracking(self, now: datetime) -> None:
        """Ajoute au suivi le rang CAC40 et la perf max de la journée.

        Ces deux informations sont calculées une seule fois, juste avant le bilan
        journalier, puis injectées dans chaque ligne SUIVI produite par V4.2.
        Elles n'interviennent jamais dans le scoring ni dans la décision de trade.
        """

        self._daily_ml_ranking_cache = self._build_daily_ml_ranking(now)
        try:
            super()._finalize_signal_tracking(now)
        finally:
            self._daily_ml_ranking_cache = {}

    def _build_daily_ml_ranking(
        self, now: datetime
    ) -> dict[str, dict[str, float | int]]:
        """Classe le CAC40 sur la performance maximale intraday depuis l'ouverture."""

        tickers = [item.ticker for item in CAC40]
        try:
            frames = self.market_data.download_universe(
                tickers,
                period="1d",
                interval="5m",
            )
        except Exception as exc:
            LOGGER.warning(
                "Classement CAC40 de fin de journée indisponible : %s",
                exc,
            )
            return {}

        performances: list[tuple[str, float]] = []
        for ticker in tickers:
            frame = frames.get(ticker)
            if frame is None or frame.empty:
                continue
            try:
                localized = self._localize_market_frame(frame)
                session = localized.loc[localized.index.date == now.date()]
                if session.empty or "Open" not in session or "High" not in session:
                    continue

                opens = pd.to_numeric(session["Open"], errors="coerce").dropna()
                highs = pd.to_numeric(session["High"], errors="coerce").dropna()
                if opens.empty or highs.empty:
                    continue

                open_price = float(opens.iloc[0])
                if open_price <= 0:
                    continue
                max_price = float(highs.max())
                perf_max = (max_price / open_price - 1) * 100
                performances.append((ticker, perf_max))
            except Exception as exc:
                LOGGER.debug(
                    "Perf max journalière non calculable pour %s : %s",
                    ticker,
                    exc,
                )

        performances.sort(key=lambda item: item[1], reverse=True)
        return {
            ticker: {
                "rang_fin_journee": rank,
                "perf_max_journee": round(perf_max, 3),
            }
            for rank, (ticker, perf_max) in enumerate(performances, start=1)
        }

    def _build_suivi_payload(
        self,
        item: dict[str, Any],
        frame: pd.DataFrame | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        payload = super()._build_suivi_payload(item, frame, now)
        if payload is None:
            return None

        daily = getattr(self, "_daily_ml_ranking_cache", {}).get(item["ticker"], {})
        payload["rang_fin_journee"] = daily.get("rang_fin_journee", "")
        payload["perf_max_journee"] = daily.get("perf_max_journee", "")
        return payload

    def _open_position(self, now: datetime, score: Score) -> None:
        """Ouvre le paper trade avec le dernier cours 1 minute disponible.

        Les indicateurs, le classement et la décision d'entrée restent calculés
        à partir du snapshot 5 minutes. Seul le prix utilisé pour l'entrée, puis
        pour les TP/SL et le suivi du trade, est rafraîchi au dernier moment.
        En cas d'indisponibilité du flux 1 minute, le moteur retombe sur le prix
        5 minutes existant afin de ne pas bloquer l'agent.
        """

        snapshot_price = float(score.snapshot["price"])
        latest_entry = self.market_data.latest_price(score.ticker)

        if latest_entry is None or latest_entry <= 0:
            LOGGER.warning(
                "Prix 1 min indisponible pour %s : repli sur le snapshot 5 min %.2f.",
                score.ticker,
                snapshot_price,
            )
            super()._open_position(now, score)
            return

        latest_entry = float(latest_entry)
        LOGGER.info(
            "Prix d'entrée rafraîchi pour %s : snapshot 5 min %.2f -> 1 min %.2f.",
            score.ticker,
            snapshot_price,
            latest_entry,
        )

        # Le moteur V4.2 et son écriture Google Sheets doivent voir le même prix
        # d'entrée. La mutation reste strictement locale à l'ouverture du trade.
        score.snapshot["price"] = latest_entry
        try:
            super()._open_position(now, score)
        finally:
            score.snapshot["price"] = snapshot_price
