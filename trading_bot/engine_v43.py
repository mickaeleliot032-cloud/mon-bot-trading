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
        """Ajoute au suivi le rang CAC40 et les variables de timing intraday.

        Ces informations sont calculées une seule fois, juste avant le bilan
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

        # Les nouvelles variables sont purement analytiques. Elles permettent de
        # mesurer si le moteur détecte une action avant ou après l'essentiel de
        # son mouvement haussier journalier.
        if frame is None or frame.empty:
            return payload

        try:
            signal_time = datetime.fromisoformat(item["signal_time"]).astimezone(
                self.timezone
            )
            signal_price = float(item["price_signal"])
            session = frame.loc[frame.index.date == signal_time.date()].copy()
            if session.empty or signal_price <= 0:
                return payload

            opens = (
                pd.to_numeric(session["Open"], errors="coerce").dropna()
                if "Open" in session
                else pd.Series(dtype=float)
            )
            highs = (
                pd.to_numeric(session["High"], errors="coerce").dropna()
                if "High" in session
                else pd.Series(dtype=float)
            )
            if opens.empty or highs.empty:
                return payload

            open_price = float(opens.iloc[0])
            if open_price <= 0:
                return payload

            # Une bougie 1 minute est horodatée au début de sa minute. Pour
            # PLUS_HAUT_AVANT_SIGNAL, on exclut volontairement la minute du
            # signal afin de ne pas utiliser un plus haut éventuellement atteint
            # quelques secondes après le déclenchement du signal.
            signal_minute = signal_time.replace(second=0, microsecond=0)
            before = session.loc[session.index < signal_minute]
            after = session.loc[session.index >= signal_time]

            before_highs = (
                pd.to_numeric(before["High"], errors="coerce").dropna()
                if not before.empty and "High" in before
                else pd.Series(dtype=float)
            )
            after_highs = (
                pd.to_numeric(after["High"], errors="coerce").dropna()
                if not after.empty and "High" in after
                else pd.Series(dtype=float)
            )

            plus_haut_avant = (
                float(before_highs.max()) if not before_highs.empty else open_price
            )
            plus_haut_apres = (
                float(after_highs.max()) if not after_highs.empty else ""
            )
            daily_high = float(highs.max())
            high_time = highs.idxmax()

            perf_ouv_signal = (signal_price / open_price - 1) * 100
            perf_max_avant = (plus_haut_avant / open_price - 1) * 100
            perf_max_apres: float | str = ""
            if plus_haut_apres != "":
                perf_max_apres = (float(plus_haut_apres) / signal_price - 1) * 100

            # On calcule ici le mouvement journalier sur les mêmes données 1 min
            # que les variables avant/après signal. Cela évite un ratio incohérent
            # si le cache CAC40 de fin de journée provient d'une série 5 minutes.
            perf_max_journee_1m = (daily_high / open_price - 1) * 100
            mouvement_consomme: float | str = ""
            if perf_max_journee_1m > 0:
                mouvement_consomme = min(
                    100.0,
                    max(0.0, perf_max_avant) / perf_max_journee_1m * 100,
                )

            payload.update(
                {
                    "prix_ouverture": round(open_price, 4),
                    "perf_ouv_signal": round(perf_ouv_signal, 3),
                    "plus_haut_avant_signal": round(plus_haut_avant, 4),
                    "perf_max_avant_signal": round(perf_max_avant, 3),
                    "plus_haut_apres_signal": (
                        round(float(plus_haut_apres), 4)
                        if plus_haut_apres != ""
                        else ""
                    ),
                    "perf_max_apres_signal": (
                        round(float(perf_max_apres), 3)
                        if perf_max_apres != ""
                        else ""
                    ),
                    "heure_plus_haut": high_time.strftime("%H:%M:%S"),
                    "mouvement_consomme_pct": (
                        round(float(mouvement_consomme), 1)
                        if mouvement_consomme != ""
                        else ""
                    ),
                }
            )
        except Exception as exc:
            LOGGER.warning(
                "Variables ML intraday non calculables pour %s : %s",
                item.get("ticker", "?"),
                exc,
            )

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
