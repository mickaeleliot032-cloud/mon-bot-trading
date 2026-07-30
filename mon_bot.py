"""Point d'entrée du bot de trading V4.2."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

from trading_bot.engine import TradingEngine


class TradingEngineV42(TradingEngine):
    """V4.2 : bilan relié au trade réellement exécuté."""

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
        leader = ranking[0] if ranking else None
        leader_text = (
            f"{leader['name']} {leader['score']:.1f}/100"
            if leader
            else "aucun classement disponible"
        )

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
            ]
        )
        self.notifier.send("\n".join(lines))
        self.state["summary_sent"] = True
        self.store.save(self.state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent de paper trading CAC 40 V4.2")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Exécute un passage GitHub Actions puis quitte.",
    )
    return parser.parse_args()


def main() -> None:
    engine = TradingEngineV42()
    if parse_args().scheduled:
        engine.run_scheduled()
    else:
        engine.run_forever()


if __name__ == "__main__":
    main()
