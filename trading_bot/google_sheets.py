"""Client non bloquant pour journaliser les événements dans Google Sheets."""

from __future__ import annotations

import logging
from datetime import datetime
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
    def _first(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None and value != "":
                return value
        return default

    @classmethod
    def _date_and_time(cls, payload: dict[str, Any]) -> tuple[str, str]:
        date_value = cls._first(payload, "date")
        time_value = cls._first(payload, "heure", "time")
        raw = cls._first(
            payload,
            "timestamp",
            "datetime",
            "entry_time",
            "exit_time",
            "heure_entree",
            "heure_sortie",
        )
        if raw:
            try:
                parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                date_value = date_value or parsed.date().isoformat()
                time_value = time_value or parsed.strftime("%H:%M:%S")
            except ValueError:
                pass
        return str(date_value or ""), str(time_value or "")

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
                "exit_time",
                "exit_price",
                "reason",
                "gross_return_pct",
                "net_pnl_eur",
                "capital_after_eur",
            }
            status = str(payload.get("statut", payload.get("status", ""))).upper()
            is_closing = bool(closing_fields.intersection(payload)) or status in {
                "FERME",
                "FERMÉ",
                "CLOSED",
            }
            return "fermer_trade" if is_closing else "ouvrir_trade"

        return normalized

    @classmethod
    def _apps_script_payload(
        cls, action: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Convertit les noms de champs V4 vers ceux attendus par Code.gs."""
        converted = dict(payload)
        date_value, time_value = cls._date_and_time(payload)

        if action == "ajouter_signal":
            ticker = cls._first(payload, "ticker")
            converted.update(
                {
                    "id_signal": cls._first(
                        payload,
                        "id_signal",
                        "signal_id",
                        "id",
                        default=f"{date_value}-{time_value}-{ticker}",
                    ),
                    "date": date_value,
                    "heure": time_value,
                    "action_nom": cls._first(payload, "action_nom", "name", "nom"),
                    "ticker": ticker,
                    "niveau": cls._first(payload, "niveau", "level"),
                    "score_global": cls._first(
                        payload, "score_global", "score", "final"
                    ),
                    "score_quantitatif": cls._first(
                        payload,
                        "score_quantitatif",
                        "quantitative",
                        "quantitative_score",
                    ),
                    "prix": cls._first(payload, "prix", "price"),
                    "variation_seance": cls._first(
                        payload,
                        "variation_seance",
                        "return_open_pct",
                        "variation",
                    ),
                    "ema20": cls._first(payload, "ema20"),
                    "ema50": cls._first(payload, "ema50"),
                    "vwap": cls._first(payload, "vwap"),
                    "volume_relatif": cls._first(
                        payload, "volume_relatif", "relative_volume", "volume_ratio"
                    ),
                    "decision": cls._first(payload, "decision", "eligible"),
                    "motifs": cls._first(payload, "motifs", "reasons"),
                }
            )

        elif action == "ouvrir_trade":
            entry_time = cls._first(payload, "heure_entree", "entry_time")
            converted.update(
                {
                    "id_trade": cls._first(payload, "id_trade", "trade_id", "id"),
                    "id_signal": cls._first(payload, "id_signal", "signal_id"),
                    "date": date_value,
                    "action_nom": cls._first(payload, "action_nom", "name", "nom"),
                    "ticker": cls._first(payload, "ticker"),
                    "heure_entree": entry_time or time_value,
                    "prix_entree": cls._first(payload, "prix_entree", "entry_price"),
                    "quantite": cls._first(payload, "quantite", "shares", "quantity"),
                    "take_profit": cls._first(
                        payload, "take_profit", "base_target_price", "target_price"
                    ),
                    "stop_loss": cls._first(payload, "stop_loss", "stop_price"),
                }
            )

        elif action == "fermer_trade":
            converted.update(
                {
                    "id_trade": cls._first(payload, "id_trade", "trade_id", "id"),
                    "heure_sortie": cls._first(
                        payload, "heure_sortie", "exit_time", default=time_value
                    ),
                    "prix_sortie": cls._first(payload, "prix_sortie", "exit_price"),
                    "motif_sortie": cls._first(payload, "motif_sortie", "reason"),
                    "performance_brute": cls._first(
                        payload, "performance_brute", "gross_return_pct"
                    ),
                    "resultat_net": cls._first(
                        payload, "resultat_net", "net_pnl_eur", "pnl"
                    ),
                    "capital_apres": cls._first(
                        payload, "capital_apres", "capital_after_eur"
                    ),
                }
            )

        elif action == "ajouter_suivi":
            converted.update(
                {
                    "id_signal": cls._first(
                        payload, "id_signal", "signal_id", "id"
                    ),
                    "date": date_value,
                    "action_nom": cls._first(payload, "action_nom", "name", "nom"),
                    "prix_signal": cls._first(
                        payload, "prix_signal", "price", "entry_price"
                    ),
                    "perf_15_min": cls._first(payload, "perf_15_min"),
                    "perf_30_min": cls._first(payload, "perf_30_min"),
                    "perf_1h": cls._first(payload, "perf_1h"),
                    "perf_2h": cls._first(payload, "perf_2h"),
                    "perf_cloture": cls._first(payload, "perf_cloture"),
                    "max_apres_signal": cls._first(payload, "max_apres_signal"),
                    "min_apres_signal": cls._first(payload, "min_apres_signal"),
                    "tp_atteint": cls._first(payload, "tp_atteint"),
                    "sl_atteint": cls._first(payload, "sl_atteint"),
                    "premier_niveau": cls._first(payload, "premier_niveau", "level"),
                }
            )

        return converted

    def send(self, action: str, **payload: Any) -> bool:
        if not self.enabled:
            return False

        apps_script_action = self._apps_script_action(action, payload)
        apps_script_payload = self._apps_script_payload(apps_script_action, payload)
        body = {
            "token": self.token,
            "action": apps_script_action,
            **apps_script_payload,
        }
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
