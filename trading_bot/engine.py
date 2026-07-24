"""Orchestration de la V4 : scans, sélection, position et notifications."""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime, timedelta
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

from trading_bot.config import Settings
from trading_bot.indicators import build_snapshot
from trading_bot.market_data import MarketDataClient
from trading_bot.news import NewsClient
from trading_bot.scoring import (
    Score,
    combine_score,
    market_context_score,
    quantitative_score,
    sector_context_score,
)
from trading_bot.state import StateStore
from trading_bot.telegram import TelegramNotifier
from trading_bot.universe import BY_TICKER, CAC40

LOGGER = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        market_data: MarketDataClient | None = None,
        news: NewsClient | None = None,
        notifier: TelegramNotifier | None = None,
        store: StateStore | None = None,
    ):
        self.settings = settings or Settings()
        self.settings.validate()
        self.timezone = ZoneInfo(self.settings.timezone)
        self.market_data = market_data or MarketDataClient()
        self.news = news or NewsClient(self.settings.http_timeout_seconds)
        self.notifier = notifier or TelegramNotifier(
            self.settings.telegram_token,
            self.settings.telegram_chat_id,
            self.settings.http_timeout_seconds,
        )
        self.store = store or StateStore(
            self.settings.state_path, self.settings.journal_path
        )
        now = datetime.now(self.timezone)
        self.state = self.store.load(now.date(), self.settings.paper_capital_eur)

    def run_forever(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )
        LOGGER.info("Agent V4 démarré en mode paper trading.")
        self.notifier.send(
            "🤖 Agent V4 démarré\n"
            "CAC 40 • scan toutes les 15 min • 1 entrée maximum/jour\n"
            "Mode : paper trading (aucun ordre broker)"
        )
        while True:
            try:
                now = datetime.now(self.timezone)
                self.run_once(now)
                active = (
                    now.weekday() < 5
                    and self.settings.warmup_time
                    <= now.time()
                    <= self.settings.forced_exit_time
                )
                delay = self.settings.loop_sleep_seconds if active else 60
                time_module.sleep(delay)
            except KeyboardInterrupt:
                LOGGER.info("Arrêt demandé.")
                return
            except Exception:
                LOGGER.exception("Erreur non bloquante dans la boucle principale.")
                time_module.sleep(60)

    def run_scheduled(self) -> None:
        """Exécute un passage court adapté à GitHub Actions.

        Sans position, le job réalise au plus le scan logique courant puis quitte.
        Avec une position ouverte, il reste actif quelques minutes et conserve
        ainsi la surveillance à la minute entre deux déclenchements planifiés.
        """

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )
        LOGGER.info("Passage planifié de l'agent V4.")
        self.run_once(datetime.now(self.timezone))

        follow_up_polls = (
            self.settings.scheduled_monitor_window_seconds
            // self.settings.position_poll_seconds
        )
        for _ in range(follow_up_polls):
            if not self.state.get("position"):
                break
            time_module.sleep(self.settings.position_poll_seconds)
            self.run_once(datetime.now(self.timezone))

        # Garantit la présence du répertoire mis en cache, même si ce passage
        # a eu lieu hors des heures de marché.
        self.store.save(self.state)

    def run_once(self, now: datetime) -> None:
        now = now.astimezone(self.timezone)
        if self.state.get("date") != now.date().isoformat():
            self.state = self.store.load(
                now.date(),
                float(self.state.get("capital", self.settings.paper_capital_eur)),
            )
            self.store.save(self.state)

        if now.weekday() >= 5:
            return

        position = self.state.get("position")
        if position and self._position_poll_due(now):
            self._monitor_position(now)
            position = self.state.get("position")

        if position and now.time() >= self.settings.forced_exit_time:
            price = self.market_data.latest_price(position["ticker"])
            if price is not None:
                self._close_position(now, price, "SORTIE_HORAIRE")
            position = self.state.get("position")

        slot = self.scan_slot(now)
        if slot and not position and self.state.get("last_scan_slot") != slot:
            self._run_scan(
                now,
                allow_entry=now.time() >= self.settings.entry_start_time,
            )
            self.state["last_scan_slot"] = slot
            self.store.save(self.state)

        if (
            now.time() >= self.settings.forced_exit_time
            and not self.state.get("summary_sent")
            and not self.state.get("position")
        ):
            self._send_daily_summary()

    def scan_slot(self, now: datetime) -> str | None:
        """Retourne le créneau logique à exécuter, même après un léger retard."""

        current = now.time().replace(second=0, microsecond=0)
        if current < self.settings.warmup_time:
            return None
        if current < self.settings.entry_start_time:
            return self.settings.warmup_time.strftime("%H:%M")
        if current > self.settings.last_scan_time:
            return None

        anchor = now.replace(
            hour=self.settings.entry_start_time.hour,
            minute=self.settings.entry_start_time.minute,
            second=0,
            microsecond=0,
        )
        elapsed = int((now - anchor).total_seconds() // 60)
        bucket = elapsed // self.settings.scan_interval_minutes
        slot = anchor + timedelta(minutes=bucket * self.settings.scan_interval_minutes)
        if slot.time() > self.settings.last_scan_time:
            return None
        return slot.strftime("%H:%M")

    def _run_scan(self, now: datetime, allow_entry: bool) -> list[Score]:
        tickers = [item.ticker for item in CAC40]
        frames = self.market_data.download_universe([self.settings.benchmark, *tickers])
        benchmark_frame = frames.get(self.settings.benchmark)
        if benchmark_frame is None:
            LOGGER.warning("Scan annulé : données CAC 40 indisponibles.")
            return []
        benchmark_snapshot = build_snapshot(
            benchmark_frame, now, self.settings.timezone
        )
        if benchmark_snapshot is None:
            LOGGER.warning("Scan annulé : séance CAC 40 absente ou incomplète.")
            return []

        snapshots: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            frame = frames.get(ticker)
            if frame is None:
                continue
            snapshot = build_snapshot(frame, now, self.settings.timezone)
            if snapshot is not None:
                snapshots[ticker] = snapshot

        market_return = float(benchmark_snapshot["return_open_pct"])
        market_score = market_context_score(benchmark_snapshot)
        sector_returns: dict[str, float] = {}
        sectors = {item.sector for item in CAC40}
        for sector in sectors:
            values = [
                snapshots[item.ticker]["return_open_pct"]
                for item in CAC40
                if item.sector == sector and item.ticker in snapshots
            ]
            sector_returns[sector] = mean(values) if values else market_return

        raw: dict[str, dict[str, Any]] = {}
        for ticker, snapshot in snapshots.items():
            instrument = BY_TICKER[ticker]
            quantitative, eligible, reasons = quantitative_score(
                snapshot,
                market_return,
                self.settings.max_absolute_gap_pct,
            )
            raw[ticker] = {
                "quantitative": quantitative,
                "eligible": eligible,
                "reasons": reasons,
                "sector_score": sector_context_score(
                    sector_returns[instrument.sector], market_return
                ),
            }

        shortlist = sorted(
            raw,
            key=lambda ticker: raw[ticker]["quantitative"],
            reverse=True,
        )[: self.settings.news_shortlist_size]
        news_by_ticker = {
            ticker: self.news.score(BY_TICKER[ticker].name) for ticker in shortlist
        }

        thresholds = (
            self.settings.watch_threshold,
            self.settings.signal_threshold,
            self.settings.strong_threshold,
        )
        ranking: list[Score] = []
        for ticker, values in raw.items():
            instrument = BY_TICKER[ticker]
            news_result = news_by_ticker.get(ticker)
            ranking.append(
                combine_score(
                    ticker=ticker,
                    name=instrument.name,
                    sector=instrument.sector,
                    snapshot=snapshots[ticker],
                    quantitative=values["quantitative"],
                    market=market_score,
                    sector_score=values["sector_score"],
                    news=news_result.score if news_result else 50.0,
                    eligible=values["eligible"],
                    reasons=values["reasons"],
                    thresholds=thresholds,
                    headlines=news_result.headlines if news_result else (),
                )
            )
        ranking.sort(key=lambda item: item.final, reverse=True)
        self._remember_ranking(ranking)

        if not ranking:
            LOGGER.warning("Aucune action exploitable pendant ce scan.")
            return ranking
        leader = ranking[0]
        LOGGER.info(
            "Leader %s : %.2f (%s), quantitatif %.2f.",
            leader.name,
            leader.final,
            leader.level,
            leader.quantitative,
        )
        self._notify_level_change(leader)
        if allow_entry and not self.state.get("entry_taken"):
            if self._entry_confirmed(leader):
                self._open_position(now, leader)
        return ranking

    def _remember_ranking(self, ranking: list[Score]) -> None:
        history = self.state.setdefault("score_history", {})
        for score in ranking:
            values = history.setdefault(score.ticker, [])
            values.append(score.final)
            del values[:-12]
        self.state["last_ranking"] = [
            {
                "ticker": item.ticker,
                "name": item.name,
                "score": item.final,
                "level": item.level,
            }
            for item in ranking[:5]
        ]

    def _entry_confirmed(self, leader: Score) -> bool:
        if not leader.eligible or leader.final < self.settings.signal_threshold:
            return False
        if leader.final >= self.settings.strong_threshold:
            return True
        history = self.state.get("score_history", {}).get(leader.ticker, [])
        previous = history[-2] if len(history) >= 2 else None
        return previous is not None and previous >= self.settings.watch_threshold

    def _notify_level_change(self, leader: Score) -> None:
        levels = {"NEUTRE": 0, "SURVEILLANCE": 1, "SIGNAL": 2, "FORT": 3}
        current = levels.get(leader.level, 0)
        alerted = self.state.setdefault("alerted_levels", {})
        previous = int(alerted.get(leader.ticker, 0))
        if current <= previous or current == 0:
            return
        alerted[leader.ticker] = current
        reasons = ", ".join(leader.reasons[:3]) or "convergence des indicateurs"
        self.notifier.send(
            f"👀 {leader.level} — {leader.name}\n"
            f"Score {leader.final:.1f}/100 (quantitatif {leader.quantitative:.1f})\n"
            f"Prix {leader.snapshot['price']:.2f} € • "
            f"séance {leader.snapshot['return_open_pct']:+.2f}%\n"
            f"Motifs : {reasons}"
        )

    def _open_position(self, now: datetime, score: Score) -> None:
        entry = float(score.snapshot["price"])
        atr_distance = (
            float(score.snapshot["atr_pct"]) * self.settings.atr_sl_multiplier
        )
        stop_distance = min(
            self.settings.max_sl_pct,
            max(self.settings.min_sl_pct, atr_distance),
        )
        capital = float(self.state["capital"])
        self.state["position"] = {
            "ticker": score.ticker,
            "name": score.name,
            "entry_time": now.isoformat(),
            "entry_price": entry,
            "shares": capital / entry,
            "capital_before": capital,
            "score": score.final,
            "stop_price": entry * (1 - stop_distance / 100),
            "base_target_price": entry * (1 + self.settings.base_tp_pct / 100),
            "extended_target_price": entry * (1 + self.settings.extended_tp_pct / 100),
            "extended_mode": False,
            "peak_price": entry,
            "last_poll": None,
        }
        self.state["entry_taken"] = True
        self.store.save(self.state)
        position = self.state["position"]
        self.notifier.send(
            f"🚀 SIGNAL PAPER — {score.name}\n"
            f"Entrée simulée : {entry:.2f} € • score {score.final:.1f}/100\n"
            f"TP : {position['base_target_price']:.2f} € "
            f"(+{self.settings.base_tp_pct:.1f}%)\n"
            f"SL dynamique : {position['stop_price']:.2f} € "
            f"(-{stop_distance:.2f}%)\n"
            "Une seule entrée sera autorisée aujourd'hui."
        )

    def _position_poll_due(self, now: datetime) -> bool:
        position = self.state["position"]
        last_poll = position.get("last_poll")
        if not last_poll:
            return True
        previous = datetime.fromisoformat(last_poll)
        return (now - previous).total_seconds() >= self.settings.position_poll_seconds

    def _monitor_position(self, now: datetime) -> None:
        position = self.state["position"]
        price = self.market_data.latest_price(position["ticker"])
        position["last_poll"] = now.isoformat()
        if price is None:
            self.store.save(self.state)
            return

        position["peak_price"] = max(float(position["peak_price"]), price)
        entry = float(position["entry_price"])
        gain_pct = (price / entry - 1) * 100

        if (
            not position["extended_mode"]
            and gain_pct >= self.settings.breakeven_trigger_pct
        ):
            position["stop_price"] = max(float(position["stop_price"]), entry)

        if price >= float(position["base_target_price"]):
            if (
                now.time() < self.settings.extended_tp_cutoff
                and not position["extended_mode"]
            ):
                position["extended_mode"] = True
                position["stop_price"] = max(float(position["stop_price"]), entry)
                self.notifier.send(
                    f"🎯 +1 % atteint tôt — {position['name']}\n"
                    f"Objectif étendu à {position['extended_target_price']:.2f} €.\n"
                    "Stop remonté au prix d'entrée, puis suivi progressif."
                )
            elif not position["extended_mode"]:
                self._close_position(now, float(position["base_target_price"]), "TP_1")
                return

        if position["extended_mode"]:
            trailing = float(position["peak_price"]) * (
                1 - self.settings.trailing_distance_pct / 100
            )
            position["stop_price"] = max(float(position["stop_price"]), trailing)
            if price >= float(position["extended_target_price"]):
                self._close_position(
                    now, float(position["extended_target_price"]), "TP_2"
                )
                return

        if price <= float(position["stop_price"]):
            reason = (
                "TRAILING_STOP"
                if position["extended_mode"]
                else ("BREAKEVEN" if position["stop_price"] >= entry else "STOP_LOSS")
            )
            self._close_position(now, price, reason)
            return
        self.store.save(self.state)

    def _close_position(self, now: datetime, exit_price: float, reason: str) -> None:
        position = self.state["position"]
        if not position:
            return
        gross_value = float(position["shares"]) * exit_price
        capital_after = gross_value - self.settings.round_trip_fees_eur
        capital_before = float(position["capital_before"])
        pnl = capital_after - capital_before
        gross_return = (exit_price / float(position["entry_price"]) - 1) * 100
        self.state["capital"] = round(capital_after, 2)
        self.store.append_trade(
            {
                "date": self.state["date"],
                "ticker": position["ticker"],
                "name": position["name"],
                "entry_time": position["entry_time"],
                "exit_time": now.isoformat(),
                "entry_price": round(float(position["entry_price"]), 4),
                "exit_price": round(exit_price, 4),
                "score": position["score"],
                "reason": reason,
                "gross_return_pct": round(gross_return, 3),
                "fees_eur": self.settings.round_trip_fees_eur,
                "net_pnl_eur": round(pnl, 2),
                "capital_after_eur": round(capital_after, 2),
            }
        )
        self.state["position"] = None
        self.store.save(self.state)
        emoji = "✅" if pnl >= 0 else "🛡️"
        self.notifier.send(
            f"{emoji} POSITION CLOSE — {position['name']}\n"
            f"Motif : {reason} • sortie {exit_price:.2f} €\n"
            f"Performance brute : {gross_return:+.2f}%\n"
            f"Résultat net frais : {pnl:+.2f} €\n"
            f"Capital paper : {capital_after:.2f} €"
        )

    def _send_daily_summary(self) -> None:
        start = float(self.state["daily_start_capital"])
        end = float(self.state["capital"])
        ranking = self.state.get("last_ranking", [])
        leader = ranking[0] if ranking else None
        leader_text = (
            f"{leader['name']} {leader['score']:.1f}/100"
            if leader
            else "aucun classement disponible"
        )
        traded = "oui" if self.state.get("entry_taken") else "non"
        self.notifier.send(
            "📊 BILAN DU JOUR V4\n"
            f"Trade simulé : {traded}\n"
            f"Meilleur dernier score : {leader_text}\n"
            f"Capital : {end:.2f} € ({end - start:+.2f} €)"
        )
        self.state["summary_sent"] = True
        self.store.save(self.state)
