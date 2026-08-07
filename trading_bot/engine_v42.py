"""Moteur V4.2 : bilan fiable et traçabilité des scans et alertes."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from trading_bot.engine import TradingEngine
from trading_bot.google_sheets import GoogleSheetsWebhook
from trading_bot.scoring import Score


class TradingEngineV42(TradingEngine):
    """Version 4.2 avec traçabilité locale et synchronisation Google Sheets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.google_sheets = GoogleSheetsWebhook(
            self.settings.google_sheets_url,
            self.settings.google_sheets_token,
            self.settings.http_timeout_seconds,
        )

    def run_once(self, now: datetime) -> None:
        """Dissocie la sortie forcée de 17 h 20 du bilan de 17 h 30."""

        now = now.astimezone(self.timezone)
        before_summary = now.time() < self.settings.daily_summary_time
        summary_already_sent = bool(self.state.get("summary_sent"))

        # Le suivi post-signal est finalisé juste avant le bilan, afin que
        # l'onglet SUIVI reçoive une seule ligne complète par signal.
        if (
            self.state.get("date") == now.date().isoformat()
            and now.time() >= self.settings.daily_summary_time
        ):
            self._finalize_signal_tracking(now)

        if before_summary and not summary_already_sent:
            # Le moteur parent utilise summary_sent pour empêcher son bilan
            # immédiat à 17 h 20. La valeur temporaire n'est pas sauvegardée.
            self.state["summary_sent"] = True

        try:
            super().run_once(now)
        finally:
            if before_summary and not summary_already_sent:
                self.state["summary_sent"] = False

    def _run_scan(self, now: datetime, allow_entry: bool) -> list[Score]:
        started = time.monotonic()
        slot = self.scan_slot(now)
        try:
            ranking = super()._run_scan(now, allow_entry)
            leader = ranking[0] if ranking else None
            event = {
                "time": now.isoformat(),
                "slot": slot,
                "status": "ok" if ranking else "empty",
                "actions_ranked": len(ranking),
                "leader_ticker": leader.ticker if leader else None,
                "leader_name": leader.name if leader else None,
                "leader_score": round(leader.final, 2) if leader else None,
                "duration_seconds": round(time.monotonic() - started, 2),
            }
            self.state.setdefault("scan_history", []).append(event)
            self.state["scan_history"] = self.state["scan_history"][-80:]
            self.store.save(self.state)
            return ranking
        except Exception as exc:
            event = {
                "time": now.isoformat(),
                "slot": slot,
                "status": "error",
                "error": type(exc).__name__,
                "duration_seconds": round(time.monotonic() - started, 2),
            }
            self.state.setdefault("scan_history", []).append(event)
            self.state["scan_history"] = self.state["scan_history"][-80:]
            self.store.save(self.state)
            raise

    def _notify_level_change(self, leader: Score) -> None:
        levels = {"NEUTRE": 0, "SURVEILLANCE": 1, "SIGNAL": 2, "FORT": 3}
        current = levels.get(leader.level, 0)
        previous = int(
            self.state.setdefault("alerted_levels", {}).get(leader.ticker, 0)
        )
        will_notify = current > previous and current > 0

        super()._notify_level_change(leader)

        if will_notify:
            alert_time = datetime.now(self.timezone)
            signal_id = (
                f"{alert_time:%Y%m%d-%H%M%S}-{leader.ticker}-{leader.level}"
            )
            snapshot = leader.snapshot
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
            }
            self.state.setdefault("alert_history", []).append(event)
            self.state["alert_history"] = self.state["alert_history"][-80:]
            self.state.setdefault("last_signal_id_by_ticker", {})[
                leader.ticker
            ] = signal_id
            self._remember_signal_for_follow_up(alert_time, leader, signal_id)
            self.store.save(self.state)
            self.google_sheets.send("alert", **event)

    def _remember_signal_for_follow_up(
        self, alert_time: datetime, leader: Score, signal_id: str
    ) -> None:
        snapshot = leader.snapshot
        atr_distance = float(snapshot["atr_pct"]) * self.settings.atr_sl_multiplier
        stop_distance = min(
            self.settings.max_sl_pct,
            max(self.settings.min_sl_pct, atr_distance),
        )
        tracking = self.state.setdefault("signal_tracking", [])
        tracking.append(
            {
                "id_signal": signal_id,
                "date": alert_time.date().isoformat(),
                "signal_time": alert_time.isoformat(),
                "ticker": leader.ticker,
                "name": leader.name,
                "price_signal": float(snapshot["price"]),
                "premier_niveau": leader.level,
                "tp_pct": float(self.settings.base_tp_pct),
                "sl_pct": float(stop_distance),
                "sent_to_sheets": False,
            }
        )
        self.state["signal_tracking"] = tracking[-80:]

    def _open_position(self, now: datetime, score: Score) -> None:
        """Crée la ligne TRADES au moment exact de l'entrée simulée."""

        super()._open_position(now, score)
        position = self.state.get("position")
        if not position:
            return

        signal_id = self.state.setdefault("last_signal_id_by_ticker", {}).get(
            score.ticker
        )
        if not signal_id:
            signal_id = f"{now:%Y%m%d-%H%M%S}-{score.ticker}-ENTRY"
        trade_id = f"T-{now:%Y%m%d-%H%M%S}-{score.ticker}"
        position["id_trade"] = trade_id
        position["id_signal"] = signal_id
        self.store.save(self.state)

        self.google_sheets.send(
            "trade",
            id_trade=trade_id,
            id_signal=signal_id,
            date=now.date().isoformat(),
            ticker=position["ticker"],
            name=position["name"],
            entry_time=position["entry_time"],
            entry_price=float(position["entry_price"]),
            shares=float(position["shares"]),
            base_target_price=float(position["base_target_price"]),
            stop_price=float(position["stop_price"]),
        )

    def _close_position(self, now: datetime, exit_price: float, reason: str) -> None:
        position = self.state.get("position")
        if not position:
            return

        trade: dict[str, Any] = {
            "id_trade": position.get("id_trade", ""),
            "id_signal": position.get("id_signal", ""),
            "ticker": position["ticker"],
            "name": position["name"],
            "entry_time": position["entry_time"],
            "entry_price": float(position["entry_price"]),
            "exit_time": now.isoformat(),
            "exit_price": float(exit_price),
            "score": float(position["score"]),
            "reason": reason,
            "capital_before": float(position["capital_before"]),
        }

        super()._close_position(now, exit_price, reason)

        capital_after = float(self.state["capital"])
        trade["gross_return_pct"] = (
            float(exit_price) / float(position["entry_price"]) - 1
        ) * 100
        trade["net_pnl_eur"] = capital_after - float(position["capital_before"])
        trade["capital_after_eur"] = capital_after
        self.state["last_trade"] = trade
        self.store.save(self.state)
        self.google_sheets.send("trade", **trade)

    def _finalize_signal_tracking(self, now: datetime) -> None:
        tracking = self.state.get("signal_tracking", [])
        pending = [item for item in tracking if not item.get("sent_to_sheets")]
        if not pending:
            return

        frames: dict[str, pd.DataFrame] = {}
        for ticker in {item["ticker"] for item in pending}:
            try:
                frame = self.market_data.download_universe(
                    [ticker], period="1d", interval="1m"
                ).get(ticker)
            except Exception:
                frame = None
            if frame is not None and not frame.empty:
                frames[ticker] = self._localize_market_frame(frame)

        changed = False
        for item in pending:
            frame = frames.get(item["ticker"])
            payload = self._build_suivi_payload(item, frame, now)
            if payload is None:
                continue
            if self.google_sheets.send("suivi", **payload):
                item["sent_to_sheets"] = True
                changed = True

        if changed:
            self.store.save(self.state)

    def _localize_market_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        localized = frame.copy().sort_index()
        index = pd.DatetimeIndex(localized.index)
        if index.tz is None:
            index = index.tz_localize("UTC").tz_convert(self.timezone)
        else:
            index = index.tz_convert(self.timezone)
        localized.index = index
        return localized

    def _build_suivi_payload(
        self,
        item: dict[str, Any],
        frame: pd.DataFrame | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        signal_time = datetime.fromisoformat(item["signal_time"]).astimezone(
            self.timezone
        )
        signal_price = float(item["price_signal"])
        if signal_price <= 0:
            return None

        if frame is None or frame.empty:
            latest = self.market_data.latest_price(item["ticker"])
            if latest is None:
                return None
            return {
                "id_signal": item["id_signal"],
                "date": item["date"],
                "name": item["name"],
                "price_signal": signal_price,
                "perf_cloture": round((latest / signal_price - 1) * 100, 3),
                "premier_niveau": item["premier_niveau"],
            }

        after = frame.loc[frame.index >= signal_time]
        if after.empty or "Close" not in after:
            return None
        close = after["Close"].dropna()
        if close.empty:
            return None

        def performance_at(minutes: int) -> float | str:
            target = signal_time + timedelta(minutes=minutes)
            values = close.loc[close.index >= target]
            if values.empty:
                return ""
            price = float(values.iloc[0])
            return round((price / signal_price - 1) * 100, 3)

        high_series = after["High"].dropna() if "High" in after else close
        low_series = after["Low"].dropna() if "Low" in after else close
        max_price = float(high_series.max())
        min_price = float(low_series.min())
        close_price = float(close.iloc[-1])
        tp_price = signal_price * (1 + float(item["tp_pct"]) / 100)
        sl_price = signal_price * (1 - float(item["sl_pct"]) / 100)

        return {
            "id_signal": item["id_signal"],
            "date": item["date"],
            "name": item["name"],
            "price_signal": signal_price,
            "perf_15_min": performance_at(15),
            "perf_30_min": performance_at(30),
            "perf_1h": performance_at(60),
            "perf_2h": performance_at(120),
            "perf_cloture": round((close_price / signal_price - 1) * 100, 3),
            "max_apres_signal": round((max_price / signal_price - 1) * 100, 3),
            "min_apres_signal": round((min_price / signal_price - 1) * 100, 3),
            "tp_atteint": max_price >= tp_price,
            "sl_atteint": min_price <= sl_price,
            "premier_niveau": item["premier_niveau"],
            "time": now.isoformat(),
        }

    def _send_daily_summary(self) -> None:
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

        lines = ["📊 BILAN DU JOUR V4.2"]
        if trade:
            entry_time = datetime.fromisoformat(trade["entry_time"]).astimezone(
                self.timezone
            )
            exit_time = datetime.fromisoformat(trade["exit_time"]).astimezone(
                self.timezone
            )
            duration_minutes = max(
                0, int((exit_time - entry_time).total_seconds() // 60)
            )
            lines.extend(
                [
                    "Trade simulé : oui",
                    f"Action tradée : {trade['name']}",
                    f"Entrée : {trade['entry_price']:.2f} € à {entry_time:%H:%M}",
                    f"Sortie : {trade['exit_price']:.2f} € à {exit_time:%H:%M}",
                    f"Durée : {duration_minutes} min",
                    f"Motif : {trade['reason']}",
                    f"Performance brute : {trade['gross_return_pct']:+.2f}%",
                    f"Résultat net : {trade['net_pnl_eur']:+.2f} €",
                ]
            )
        else:
            lines.append("Trade simulé : non")

        lines.extend(
            [
                f"Capital : {end:.2f} € ({end - start:+.2f} €)",
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
            "leader": leader,
            "successful_scans": successful_scans,
            "scan_errors": scan_errors,
            "strong_alerts": strong_alerts,
        }
        self.google_sheets.send("summary", **summary)
        self.state["summary_sent"] = True
        self.store.save(self.state)
