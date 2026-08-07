"""Améliorations V4.1 : fiabilité Telegram et diagnostics GitHub Actions."""

from __future__ import annotations

import logging
from datetime import datetime

from trading_bot.engine import TradingEngine as BaseTradingEngine
from trading_bot.scoring import Score

LOGGER = logging.getLogger(__name__)


class TradingEngine(BaseTradingEngine):
    """Moteur V4.1 rétrocompatible avec l'algorithme V4."""

    def run_scheduled(self) -> None:
        now = datetime.now(self.timezone)
        LOGGER.info(
            "V4.1 — passage à %s, position=%s, dernier_scan=%s, bilan_envoyé=%s.",
            now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "oui" if self.state.get("position") else "non",
            self.state.get("last_scan_slot") or "aucun",
            "oui" if self.state.get("summary_sent") else "non",
        )
        super().run_scheduled()
        LOGGER.info(
            "V4.1 — passage terminé, position=%s, capital=%.2f €.",
            "oui" if self.state.get("position") else "non",
            float(self.state.get("capital", self.settings.paper_capital_eur)),
        )

    def test_telegram(self) -> bool:
        """Envoie un message de contrôle sans lancer l'analyse de marché."""
        now = datetime.now(self.timezone)
        LOGGER.info("Test Telegram V4.1 demandé.")
        sent = self.notifier.send(
            "✅ TEST TELEGRAM RÉUSSI — AGENT V4.1\n"
            "GitHub Actions communique correctement avec le bot.\n"
            f"Heure de Paris : {now.strftime('%d/%m/%Y %H:%M:%S')}\n"
            "Mode : paper trading"
        )
        if sent:
            LOGGER.info("Notification de test Telegram envoyée avec succès.")
        else:
            LOGGER.error("Échec de la notification de test Telegram.")
        return sent

    def _run_scan(self, now: datetime, allow_entry: bool) -> list[Score]:
        ranking = super()._run_scan(now, allow_entry)
        slot = self.scan_slot(now) or now.strftime("%H:%M")

        LOGGER.info(
            "RAPPORT SCAN %s — %d actions classées — entrée autorisée=%s.",
            slot,
            len(ranking),
            "oui" if allow_entry else "non",
        )

        if not ranking:
            LOGGER.info("Aucun classement exploitable pour ce scan.")
            return ranking

        for index, score in enumerate(ranking[:5], start=1):
            reasons = ", ".join(score.reasons[:3]) or "aucun motif dominant"
            LOGGER.info(
                "TOP %d — %s (%s) : %.1f/100, niveau=%s, éligible=%s, motifs=%s.",
                index,
                score.name,
                score.ticker,
                score.final,
                score.level,
                "oui" if score.eligible else "non",
                reasons,
            )

        leader = ranking[0]
        if self.state.get("entry_taken"):
            decision = "aucune nouvelle entrée : limite quotidienne déjà utilisée"
        elif not allow_entry:
            decision = "observation uniquement : fenêtre d'entrée non ouverte"
        elif not leader.eligible:
            decision = "aucune entrée : meilleur candidat non éligible"
        elif leader.final < self.settings.watch_threshold:
            decision = "aucune entrée : score inférieur au seuil de surveillance"
        elif leader.final < self.settings.signal_threshold:
            decision = "aucune entrée : surveillance sans signal confirmé"
        elif self.state.get("position"):
            decision = "position ouverte"
        else:
            decision = (
                "signal potentiel en attente de confirmation ou entrée déclenchée"
            )

        LOGGER.info("DÉCISION SCAN %s — %s.", slot, decision)
        return ranking

    def _notify_level_change(self, leader: Score) -> None:
        levels = {"NEUTRE": 0, "SURVEILLANCE": 1, "SIGNAL": 2, "FORT": 3}
        current = levels.get(leader.level, 0)
        alerted = self.state.setdefault("alerted_levels", {})
        previous = int(alerted.get(leader.ticker, 0))
        if current <= previous or current == 0:
            return

        reasons = ", ".join(leader.reasons[:3]) or "convergence des indicateurs"
        sent = self.notifier.send(
            f"👀 {leader.level} — {leader.name}\n"
            f"Score {leader.final:.1f}/100 (quantitatif {leader.quantitative:.1f})\n"
            f"Prix {leader.snapshot['price']:.2f} € • "
            f"séance {leader.snapshot['return_open_pct']:+.2f}%\n"
            f"Motifs : {reasons}"
        )
        if sent:
            alerted[leader.ticker] = current
            self.store.save(self.state)
            LOGGER.info("Alerte %s envoyée pour %s.", leader.level, leader.name)
        else:
            LOGGER.warning(
                "Alerte %s non envoyée pour %s : elle pourra être retentée.",
                leader.level,
                leader.name,
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
        message = (
            "📊 BILAN DU JOUR V4.1\n"
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
