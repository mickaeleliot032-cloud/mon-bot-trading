"""Prédictions ML en shadow mode : elles n'influencent jamais l'entrée réelle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

LOGGER = logging.getLogger(__name__)

TOP3_PATH = Path("data/ml/top3_model.joblib")
REMAINING_PATH = Path("data/ml/remaining_potential_model.joblib")

TOP3_NUMERIC = [
    "SCORE_GLOBAL",
    "SCORE_QUANTITATIF",
    "PRIX",
    "VARIATION_SEANCE",
    "EMA20",
    "EMA50",
    "VWAP",
    "VOLUME_RELATIF",
    "PERF_CAC40",
    "SURPERF_CAC40",
    "MOMENTUM_15M",
    "RSI14",
    "ATR_PCT",
    "GAP_PCT",
    "SCORE_MARCHE",
    "SCORE_SECTEUR",
    "SCORE_NEWS",
    "HEURE_DECIMALE",
]
REMAINING_EXTRA = [
    "PRIX_OUVERTURE",
    "PERF_OUV_SIGNAL",
    "PLUS_HAUT_AVANT_SIGNAL",
    "PERF_MAX_AVANT_SIGNAL",
]
CATEGORICAL = ["NIVEAU", "SECTEUR"]


class MLShadowScorer:
    """Charge les deux modèles hebdomadaires et calcule des probabilités indicatives."""

    def __init__(self) -> None:
        self.top3: Any | None = None
        self.remaining: Any | None = None
        try:
            if TOP3_PATH.exists():
                self.top3 = joblib.load(TOP3_PATH)
            if REMAINING_PATH.exists():
                loaded = joblib.load(REMAINING_PATH)
                self.remaining = (
                    loaded.get("model") if isinstance(loaded, dict) else loaded
                )
        except Exception as exc:
            LOGGER.warning("Modèles ML shadow indisponibles : %s", exc)

    @property
    def enabled(self) -> bool:
        return self.top3 is not None or self.remaining is not None

    @staticmethod
    def _frame(features: dict[str, Any], columns: list[str]) -> pd.DataFrame:
        row = {column: features.get(column, "") for column in columns}
        return pd.DataFrame([row])

    def predict(self, features: dict[str, Any]) -> dict[str, float | str]:
        result: dict[str, float | str] = {
            "proba_top3_ml": "",
            "proba_potentiel_1pct_ml": "",
            "score_ml_combine": "",
        }
        try:
            if self.top3 is not None:
                cols = TOP3_NUMERIC + CATEGORICAL
                result["proba_top3_ml"] = round(
                    float(
                        self.top3.predict_proba(self._frame(features, cols))[0, 1]
                    )
                    * 100,
                    2,
                )
            if self.remaining is not None:
                cols = TOP3_NUMERIC + REMAINING_EXTRA + CATEGORICAL
                result["proba_potentiel_1pct_ml"] = round(
                    float(
                        self.remaining.predict_proba(
                            self._frame(features, cols)
                        )[0, 1]
                    )
                    * 100,
                    2,
                )
        except Exception as exc:
            LOGGER.warning("Prédiction ML shadow impossible : %s", exc)
            return result

        top3 = result["proba_top3_ml"]
        remaining = result["proba_potentiel_1pct_ml"]
        if top3 != "" and remaining != "":
            # Le modèle potentiel restant est actuellement mieux validé : poids 65 %.
            result["score_ml_combine"] = round(
                0.35 * float(top3) + 0.65 * float(remaining), 2
            )
        elif remaining != "":
            result["score_ml_combine"] = remaining
        elif top3 != "":
            result["score_ml_combine"] = top3
        return result
