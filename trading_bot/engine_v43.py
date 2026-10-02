"""Moteur V4.3 : prix d'entrée paper basé sur la cotation 1 minute."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from trading_bot.engine import TradingEngine
from trading_bot.engine_v42 import TradingEngineV42
from trading_bot.indicators import build_snapshot
from trading_bot.ml_shadow import MLShadowScorer
from trading_bot.scoring import Score
from trading_bot.universe import CAC40

LOGGER = logging.getLogger(__name__)


class TradingEngineV43(TradingEngineV42):
    """Conserve le scoring V4.2 mais enrichit la collecte pour le futur ML."""

    def _prepare_shadow_ml(
        self, ranking: list[Score], now: datetime, market_return: float
    ) -> None:
        """Classe les candidats avec les modèles ML sans influencer le trade réel."""

        scorer = getattr(self, "_ml_shadow_scorer", None)
        if scorer is None:
            scorer = MLShadowScorer()
            self._ml_shadow_scorer = scorer
        self._ml_shadow_current = {}
        if not scorer.enabled:
            LOGGER.info("ML shadow : aucun modèle disponible pour ce passage.")
            return

        # Une seule requête 1 minute pour tous les candidats afin de reconstruire
        # les quatre variables de timing disponibles au moment du signal.
        minute_frames: dict[str, pd.DataFrame] = {}
        try:
            minute_frames = self.market_data.download_universe(
                [item.ticker for item in ranking], period="1d", interval="1m"
            )
        except Exception as exc:
            LOGGER.warning("Données 1 min ML shadow indisponibles : %s", exc)

        rows: list[tuple[Score, dict[str, float | str]]] = []
        signal_minute = now.replace(second=0, microsecond=0)
        heure_decimale = now.hour + now.minute / 60 + now.second / 3600

        for item in ranking:
            snap = item.snapshot
            open_price: float | str = ""
            high_before: float | str = ""
            perf_open_signal: float | str = ""
            perf_max_before: float | str = ""
            frame = minute_frames.get(item.ticker)
            if frame is not None and not frame.empty:
                try:
                    localized = self._localize_market_frame(frame)
                    session = localized.loc[localized.index.date == now.date()]
                    opens = pd.to_numeric(session["Open"], errors="coerce").dropna()
                    before = session.loc[session.index < signal_minute]
                    highs = pd.to_numeric(before["High"], errors="coerce").dropna()
                    if not opens.empty:
                        open_price = float(opens.iloc[0])
                        signal_price = float(snap["price"])
                        if open_price > 0:
                            perf_open_signal = (signal_price / open_price - 1) * 100
                            high_before = (
                                float(highs.max()) if not highs.empty else open_price
                            )
                            perf_max_before = (
                                float(high_before) / open_price - 1
                            ) * 100
                except Exception as exc:
                    LOGGER.debug(
                        "Timing ML indisponible pour %s : %s",
                        item.ticker,
                        exc,
                    )

            features = {
                "SCORE_GLOBAL": item.final,
                "SCORE_QUANTITATIF": item.quantitative,
                "PRIX": snap.get("price", ""),
                "VARIATION_SEANCE": snap.get("return_open_pct", ""),
                "EMA20": snap.get("ema20", ""),
                "EMA50": snap.get("ema50", ""),
                "VWAP": snap.get("vwap", ""),
                "VOLUME_RELATIF": snap.get("volume_ratio", ""),
                "PERF_CAC40": market_return,
                "SURPERF_CAC40": (
                    float(snap.get("return_open_pct", 0)) - market_return
                ),
                "MOMENTUM_15M": snap.get("momentum_15m_pct", ""),
                "RSI14": snap.get("rsi14", ""),
                "ATR_PCT": snap.get("atr_pct", ""),
                "GAP_PCT": snap.get("gap_pct", ""),
                "SCORE_MARCHE": item.market,
                "SCORE_SECTEUR": item.sector_score,
                "SCORE_NEWS": item.news,
                "HEURE_DECIMALE": heure_decimale,
                "PRIX_OUVERTURE": open_price,
                "PERF_OUV_SIGNAL": perf_open_signal,
                "PLUS_HAUT_AVANT_SIGNAL": high_before,
                "PERF_MAX_AVANT_SIGNAL": perf_max_before,
                "NIVEAU": item.level,
                "SECTEUR": item.sector,
            }
            prediction = scorer.predict(features)
            rows.append((item, prediction))

        valid = [
            (item, pred)
            for item, pred in rows
            if pred.get("score_ml_combine", "") != ""
        ]
        valid.sort(
            key=lambda pair: float(pair[1]["score_ml_combine"]), reverse=True
        )
        ml_choice = valid[0][0] if valid else None
        rank_by_ticker = {
            item.ticker: rank for rank, (item, _pred) in enumerate(valid, start=1)
        }

        agent_choice = ranking[0]
        for item, prediction in rows:
            self._ml_shadow_current[item.ticker] = {
                **prediction,
                "rang_ml": rank_by_ticker.get(item.ticker, ""),
                "choix_ml": ml_choice.name if ml_choice else "",
                "choix_agent": agent_choice.name,
                "ml_accord_agent": (
                    "OUI"
                    if ml_choice is not None and ml_choice.ticker == agent_choice.ticker
                    else "NON" if ml_choice is not None else ""
                ),
                "resultat_choix_ml": "",
            }

        if ml_choice is not None:
            self._remember_v5_shadow_choice(ml_choice, now)
            LOGGER.info(
                "ML shadow : choix=%s, score=%.2f, choix agent=%s "
                "(aucun impact sur le trade).",
                ml_choice.name,
                float(
                    self._ml_shadow_current[ml_choice.ticker][
                        "score_ml_combine"
                    ]
                ),
                agent_choice.name,
            )

    def _remember_v5_shadow_choice(self, choice: Score, now: datetime) -> None:
        """Ouvre le premier paper trade V5 éligible, sans influencer V4.3."""

        if self.state.get("v5_shadow_trade"):
            return
        if now.time() < self.settings.entry_start_time:
            return
        if not self._entry_confirmed(choice):
            return

        entry = self.market_data.latest_price(choice.ticker)
        if entry is None or entry <= 0:
            entry = float(choice.snapshot["price"])
        entry = float(entry)

        atr_distance = (
            float(choice.snapshot["atr_pct"]) * self.settings.atr_sl_multiplier
        )
        stop_distance = min(
            self.settings.max_sl_pct,
            max(self.settings.min_sl_pct, atr_distance),
        )
        ml = self._ml_shadow_current.get(choice.ticker, {})
        capital = float(
            self.state.setdefault(
                "v5_shadow_capital",
                self.state.get("daily_start_capital", self.state["capital"]),
            )
        )
        self.state["v5_shadow_daily_start_capital"] = capital
        self.state["v5_shadow_trade"] = {
            "ticker": choice.ticker,
            "name": choice.name,
            "entry_time": now.isoformat(),
            "entry_price": entry,
            "shares": capital / entry,
            "capital_before": capital,
            "score_ml": ml.get("score_ml_combine", ""),
            "proba_top3_ml": ml.get("proba_top3_ml", ""),
            "proba_potentiel_1pct_ml": ml.get("proba_potentiel_1pct_ml", ""),
            "stop_price": entry * (1 - stop_distance / 100),
            "base_target_price": entry * (1 + self.settings.base_tp_pct / 100),
            "extended_target_price": entry * (1 + self.settings.extended_tp_pct / 100),
            "extended_mode": False,
            "peak_price": entry,
        }
        self.store.save(self.state)

    def _finalize_v5_shadow(self) -> dict[str, Any] | None:
        """Rejoue la logique de sortie V4.3 sur le choix V5 avec les cours 1 min."""

        trade = self.state.get("v5_shadow_trade")
        if not trade:
            return None
        if trade.get("result"):
            return trade

        try:
            frame = self.market_data.download_universe(
                [trade["ticker"]], period="1d", interval="1m"
            ).get(trade["ticker"])
            if frame is None or frame.empty:
                return trade
            frame = self._localize_market_frame(frame)
            entry_time = datetime.fromisoformat(trade["entry_time"]).astimezone(
                self.timezone
            )
            after = frame.loc[frame.index >= entry_time]
            if after.empty:
                return trade

            entry = float(trade["entry_price"])
            stop_price = float(trade["stop_price"])
            base_target = float(trade["base_target_price"])
            extended_target = float(trade["extended_target_price"])
            extended_mode = False
            peak_price = entry
            reason = "SORTIE_HORAIRE"
            exit_price = float(pd.to_numeric(after["Close"], errors="coerce").dropna().iloc[-1])
            exit_time = after.index[-1]

            for stamp, row in after.iterrows():
                high = float(row["High"])
                low = float(row["Low"])
                peak_price = max(peak_price, high)

                # Convention prudente pour une bougie 1 min ambiguë.
                if low <= stop_price:
                    reason = (
                        "TRAILING_STOP"
                        if extended_mode
                        else ("BREAKEVEN" if stop_price >= entry else "STOP_LOSS")
                    )
                    exit_price, exit_time = stop_price, stamp
                    break

                if high >= base_target:
                    if (
                        stamp.time() < self.settings.extended_tp_cutoff
                        and not extended_mode
                    ):
                        extended_mode = True
                        stop_price = max(stop_price, entry)
                    elif not extended_mode:
                        reason, exit_price, exit_time = "TP_1", base_target, stamp
                        break

                if extended_mode:
                    trailing = peak_price * (
                        1 - self.settings.trailing_distance_pct / 100
                    )
                    stop_price = max(stop_price, trailing)
                    if high >= extended_target:
                        reason, exit_price, exit_time = "TP_2", extended_target, stamp
                        break

            gross_value = float(trade["shares"]) * exit_price
            capital_after = gross_value - self.settings.round_trip_fees_eur
            capital_before = float(trade["capital_before"])
            net_pnl = capital_after - capital_before
            gross_return = (exit_price / entry - 1) * 100
            trade.update(
                {
                    "exit_time": exit_time.isoformat(),
                    "exit_price": round(exit_price, 4),
                    "reason": reason,
                    "gross_return_pct": round(gross_return, 3),
                    "net_pnl_eur": round(net_pnl, 2),
                    "capital_after_eur": round(capital_after, 2),
                    "result": "GAGNE" if net_pnl > 0 else "PERDU",
                }
            )
            self.state["v5_shadow_capital"] = round(capital_after, 2)
            history = self.state.setdefault("v5_shadow_history", [])
            history.append(
                {
                    "date": self.state.get("date"),
                    **{key: trade.get(key) for key in (
                        "ticker", "name", "entry_time", "entry_price", "exit_time",
                        "exit_price", "score_ml", "reason", "gross_return_pct",
                        "net_pnl_eur", "capital_before", "capital_after_eur", "result"
                    )},
                }
            )
            self.state["v5_shadow_history"] = history[-250:]
            self.store.save(self.state)
        except Exception as exc:
            LOGGER.warning("Résultat V5 shadow non calculable : %s", exc)
        return trade

    def _send_daily_summary(self) -> None:
        """Ajoute le résultat et le capital V5 shadow au bilan Telegram V4.3."""

        v5 = self._finalize_v5_shadow()
        start = float(self.state["daily_start_capital"])
        end = float(self.state["capital"])
        trade = self.state.get("last_trade")
        ranking = self.state.get("last_ranking", [])
        scans = self.state.get("scan_history", [])
        alerts = self.state.get("alert_history", [])

        leader = ranking[0] if ranking else None
        leader_text = (
            f"{leader['name']} {leader['score']:.1f}/100"
            if leader
            else "aucun classement disponible"
        )
        successful_scans = sum(1 for scan in scans if scan.get("status") == "ok")
        scan_errors = sum(1 for scan in scans if scan.get("status") == "error")
        strong_alerts = sum(1 for alert in alerts if alert.get("level") == "FORT")

        lines = ["📊 BILAN DU JOUR V4.3 + V5 SHADOW"]
        if trade:
            lines.extend(
                [
                    "",
                    "V4.3 — trade réel paper",
                    f"Action : {trade['name']}",
                    f"Motif : {trade['reason']}",
                    f"Performance brute : {trade['gross_return_pct']:+.2f}%",
                    f"Résultat net : {trade['net_pnl_eur']:+.2f} €",
                ]
            )
        else:
            lines.extend(["", "V4.3 — trade réel paper", "Trade simulé : non"])

        lines.extend(["", "V5 — Shadow ML"])
        if v5 and v5.get("result"):
            lines.extend(
                [
                    f"Action : {v5['name']}",
                    f"Score ML : {float(v5['score_ml']):.3f}" if v5.get("score_ml") != "" else "Score ML : n/d",
                    f"Motif : {v5['reason']}",
                    f"Performance brute : {v5['gross_return_pct']:+.2f}%",
                    f"Résultat net : {v5['net_pnl_eur']:+.2f} €",
                    f"Capital V5 : {v5['capital_after_eur']:.2f} €",
                ]
            )
        elif v5:
            lines.append(f"Action : {v5['name']} — résultat indisponible")
        else:
            lines.append("Aucun trade V5 éligible aujourd'hui")

        lines.extend(
            [
                "",
                f"Capital V4.3 : {end:.2f} € ({end - start:+.2f} €)",
                f"Dernier leader observé : {leader_text}",
                f"Scans réussis : {successful_scans}",
                f"Erreurs de scan : {scan_errors}",
                f"Alertes fortes : {strong_alerts}",
            ]
        )
        self.notifier.send("\n".join(lines))

        summary = {
            "date": self.state.get("date"),
            "capital_start_eur": start,
            "capital_end_eur": end,
            "net_pnl_eur": end - start,
            "trade_taken": bool(trade),
            "trade": trade,
            "v5_shadow": v5,
            "v5_shadow_capital_eur": self.state.get("v5_shadow_capital"),
            "leader": leader,
            "successful_scans": successful_scans,
            "scan_errors": scan_errors,
            "strong_alerts": strong_alerts,
        }
        self.google_sheets.send("summary", **summary)
        self.state["summary_sent"] = True
        self.store.save(self.state)

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
            # Shadow mode : ces champs sont purement analytiques et n'entrent
            # jamais dans la décision d'ouverture de position.
            **getattr(self, "_ml_shadow_current", {}).get(leader.ticker, {}),
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
