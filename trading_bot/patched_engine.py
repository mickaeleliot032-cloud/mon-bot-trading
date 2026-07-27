"""Correctifs ciblés du moteur V4."""

from __future__ import annotations

import logging

from trading_bot.engine import TradingEngine as BaseTradingEngine

LOGGER = logging.getLogger(__name__)


class TradingEngine(BaseTradingEngine):
    """Moteur V4 avec confirmation réelle des notifications Telegram."""

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
        message = (
            "📊 BILAN DU JOUR V4\n"
            f"Trade simulé : {traded}\n"
            f"Meilleur dernier score : {leader_text}\n"
            f"Capital : {end:.2f} € ({end - start:+.2f} €)"
        )

        sent = self.notifier.send(message)
        if sent:
            self.state["summary_sent"] = True
            LOGGER.info("Bilan quotidien envoyé sur Telegram.")
        else:
            self.state["summary_sent"] = False
            LOGGER.warning(
                "Bilan quotidien non envoyé : nouvelle tentative au prochain passage."
            )
        self.store.save(self.state)
