"""Moteur V4.2 : bilan fiable et traçabilité des scans et alertes."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from trading_bot.engine import TradingEngine
from trading_bot.scoring import Score


class TradingEngineV42(TradingEngine):
    """Version 4.2 sans duplication du point d'entrée.

    La stratégie de trading reste celle de la V4. La V4.2 ajoute uniquement
    la traçabilité nécessaire pour relier le bilan au trade réellement exécuté.
    """

    def _run_scan(self, now: datetime, allow_entry: bool) -> list[Score]:
        started = time.monotonic()
        slot = self.scan_slot(now)
        try:
            ranking = super()._run_scan(now, allow_entry)
            leader = ranking[0] if ranking else None
            self.state.setdefault("scan_history", []).append(
                {
                    "time": now.isoformat(),
                    "slot": slot,
                    "status": "ok" if ranking else "empty",
                    "actions_ranked": len(ranking),
                    "leader_ticker": leader.ticker if leader else None,
                    "leader_name": leader.name if leader else None,
                    "leader_score": round(leader.final, 2) if leader else None,
                    "duration_seconds": round(time.monotonic() - started, 2),
                }
            )
            self.state["scan_history"] = self.state["scan_history"][-80:]
            self.store.save(self.state)
            return ranking
        except Exception as exc:
            self.state.setdefault("scan_history", []).append(
                {
                    "time": now.isoformat(),
                    "slot": slot,
                    "status": "error",
                    "error": type(exc).__name__,
                    "duration_seconds": round(time.monotonic() - started, 2),
                }
            )
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
            self.state.setdefault("alert_history", []).append(
                {
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
            )
            self.state["alert_history"] = self.state["alert_history"][-80:]
            self.store.save(self.state)

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
        self.state["summary_sent"] = True
        self.store.save(self.state)
