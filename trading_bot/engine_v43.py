"""Moteur V4.3 : prix d'entrée paper basé sur la cotation 1 minute."""

from __future__ import annotations

import logging
from datetime import datetime

from trading_bot.engine_v42 import TradingEngineV42
from trading_bot.scoring import Score

LOGGER = logging.getLogger(__name__)


class TradingEngineV43(TradingEngineV42):
    """Conserve le scoring V4.2 mais rafraîchit le prix au moment de l'entrée."""

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
