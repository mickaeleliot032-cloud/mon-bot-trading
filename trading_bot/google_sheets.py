"""Client non bloquant pour journaliser les événements dans Google Sheets."""

from __future__ import annotations

import logging
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class GoogleSheetsWebhook:
    """Envoie des événements JSON à une application Web Google Apps Script.

    Une indisponibilité de Google Sheets ne doit jamais interrompre le moteur de
    paper trading. Les erreurs sont donc journalisées puis ignorées.
    """

    def __init__(self, url: str | None, token: str | None, timeout_seconds: int = 10):
        self.url = url
        self.token = token
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)

    @staticmethod
    def _apps_script_action(action: str, payload: dict[str, Any]) -> str:
        """Traduit les événements V4 vers les actions attendues par Code.gs."""
        normalized = action.strip().lower()

        if normalized == "alert":
            return "ajouter_signal"
        if normalized == "scan":
            return "ajouter_suivi"
        if normalized == "trade":
            closing_fields = {
                "heure_sortie",
                "prix_sortie",
                "motif_sortie",
                "performance_brute",
                "resultat_net",
                "capital_apres",
            }
            status = str(payload.get("statut", payload.get("status", ""))).upper()
            is_closing = bool(closing_fields.intersection(payload)) or status in {
                "FERME",
                "FERMÉ",
                "CLOSED",
            }
            return "fermer_trade" if is_closing else "ouvrir_trade"

        return normalized

    def send(self, action: str, **payload: Any) -> bool:
        if not self.enabled:
            return False

        apps_script_action = self._apps_script_action(action, payload)
        body = {"token": self.token, "action": apps_script_action, **payload}
        try:
            response = requests.post(
                str(self.url),
                json=body,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("success"):
                LOGGER.warning(
                    "Google Sheets a refusé l'événement %s (%s) : %s",
                    action,
                    apps_script_action,
                    result.get("error", "erreur inconnue"),
                )
                return False
            LOGGER.info(
                "Événement Google Sheets enregistré : %s (%s)",
                action,
                apps_script_action,
            )
            return True
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning(
                "Échec non bloquant de l'envoi Google Sheets (%s/%s) : %s",
                action,
                apps_script_action,
                exc,
            )
            return False
