"""Moteur V4.2 : bilan fiable et traçabilité des scans et alertes."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

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
            self.google_sheets.send("scan", **event)
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
            self.google_sheets.send("scan", **event)
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
            event = {
                "time": datetime.now(self.timezone).isoformat(),
                "ticker": leader.ticker,
                "name": leader.name,
                "score": round(leader.final, 2),
                "level": leader.level,
                "price": round(float(leader.snapshot["price"]), 4),
                "return_open_pct": round(
                    float(leader.snapshot["return_open_pct"]), 3
                ),
            }
            self.state.setdefault("alert_history", []).append(event)
            self.state["alert_history"] = self.state["alert_history"][-80:]
            self.store.save(self.state)
            self.google_sheets.send("alert", **event)

    def _close_position(self, now: datetime, exit_price: float, reason: str) -> None:
        position = self.state.get("position")
        if not position:
            return

        trade: dict[str, Any] = {
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
