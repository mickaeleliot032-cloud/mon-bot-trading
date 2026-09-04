"""Entraînement ML pour estimer si un signal conserve au moins +1 % de potentiel.

Ce modèle est volontairement séparé du modèle Top 3 existant. Il n'utilise comme
features que des informations connues au moment du signal. Les colonnes calculées
après le signal servent uniquement de cible ou de diagnostic afin d'éviter toute
fuite d'information.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTPUT_DIR = Path("data/ml")
MODEL_PATH = OUTPUT_DIR / "remaining_potential_model.joblib"
METRICS_PATH = OUTPUT_DIR / "remaining_potential_metrics.json"
DATASET_PATH = OUTPUT_DIR / "remaining_potential_dataset.csv"

TARGET_THRESHOLD_PCT = float(os.environ.get("ML_REMAINING_POTENTIAL_THRESHOLD", "1.0"))
TARGET_COLUMN = "POTENTIEL_RESTANT_1PCT"

# Uniquement des informations disponibles au moment du signal.
NUMERIC_FEATURES = [
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
    # Nouvelles variables disponibles au moment du signal.
    "PRIX_OUVERTURE",
    "PERF_OUV_SIGNAL",
    "PLUS_HAUT_AVANT_SIGNAL",
    "PERF_MAX_AVANT_SIGNAL",
]
CATEGORICAL_FEATURES = ["NIVEAU", "SECTEUR"]

# Colonnes de résultat futur : jamais utilisées comme features.
OUTCOME_COLUMNS = [
    "PERF_MAX_APRES_SIGNAL",
    "PLUS_HAUT_APRES_SIGNAL",
    "HEURE_PLUS_HAUT",
    "MOUVEMENT_CONSOMME_PCT",
    "PERF_MAX_JOURNEE",
    "RANG_FIN_JOURNEE",
]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def _load_via_apps_script() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    url = os.environ.get("GOOGLE_SHEETS_URL", "").strip()
    token = os.environ.get("GOOGLE_SHEETS_TOKEN", "").strip()
    if not url:
        raise RuntimeError("Secret GOOGLE_SHEETS_URL absent.")
    if not token:
        raise RuntimeError("Secret GOOGLE_SHEETS_TOKEN absent.")

    response = requests.post(
        url,
        json={"token": token, "action": "export_ml"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(
            f"Export Apps Script refusé : {payload.get('error', 'réponse invalide') if isinstance(payload, dict) else 'réponse invalide'}"
        )

    signals = payload.get("signaux", [])
    follow_up = payload.get("suivi", [])
    trades = payload.get("trades", [])

    if not isinstance(signals, list) or not isinstance(follow_up, list):
        raise RuntimeError("Export Apps Script incomplet : SIGNAUX/SUIVI absents ou invalides.")
    if not isinstance(trades, list):
        trades = []

    print(
        "Export ML reçu : "
        f"SIGNAUX={len(signals)}, SUIVI={len(follow_up)}, TRADES={len(trades)}."
    )
    if not trades:
        print(
            "Information : l'onglet TRADES n'est pas actuellement fourni par export_ml. "
            "Il n'est pas utilisé comme feature afin d'éviter une fuite d'information post-trade."
        )

    return (
        _normalize_columns(pd.DataFrame(signals)),
        _normalize_columns(pd.DataFrame(follow_up)),
        _normalize_columns(pd.DataFrame(trades)),
    )


def _hour_to_decimal(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour + parsed.minute / 60 + parsed.second / 3600)


def _clean_id_signal(frame: pd.DataFrame) -> pd.DataFrame:
    if "ID_SIGNAL" not in frame.columns:
        return frame.iloc[0:0].copy()
    frame = frame.copy()
    frame["ID_SIGNAL"] = frame["ID_SIGNAL"].fillna("").astype(str).str.strip()
    return frame[frame["ID_SIGNAL"] != ""].copy()


def build_dataset(signals: pd.DataFrame, follow_up: pd.DataFrame) -> pd.DataFrame:
    required_signal = {"ID_SIGNAL", "DATE"}
    required_follow = {"ID_SIGNAL", "PERF_MAX_APRES_SIGNAL"}
    missing_signal = required_signal.difference(signals.columns)
    missing_follow = required_follow.difference(follow_up.columns)
    if missing_signal:
        raise RuntimeError(f"Colonnes SIGNAUX manquantes : {sorted(missing_signal)}")
    if missing_follow:
        raise RuntimeError(
            "Les nouvelles colonnes de SUIVI ne sont pas encore disponibles dans l'export : "
            f"{sorted(missing_follow)}"
        )

    signals = _clean_id_signal(signals).drop_duplicates("ID_SIGNAL", keep="last")
    follow_up = _clean_id_signal(follow_up)
    follow_up["_ROW_ORDER"] = np.arange(len(follow_up))
    follow_up = (
        follow_up.sort_values(["ID_SIGNAL", "_ROW_ORDER"])
        .drop_duplicates("ID_SIGNAL", keep="last")
        .drop(columns=["_ROW_ORDER"])
    )

    wanted_follow = ["ID_SIGNAL"]
    for column in [
        "PRIX_OUVERTURE",
        "PERF_OUV_SIGNAL",
        "PLUS_HAUT_AVANT_SIGNAL",
        "PERF_MAX_AVANT_SIGNAL",
        *OUTCOME_COLUMNS,
    ]:
        if column in follow_up.columns:
            wanted_follow.append(column)

    dataset = signals.merge(
        follow_up[wanted_follow],
        on="ID_SIGNAL",
        how="inner",
        validate="one_to_one",
    )

    dataset["PERF_MAX_APRES_SIGNAL"] = pd.to_numeric(
        dataset["PERF_MAX_APRES_SIGNAL"], errors="coerce"
    )
    dataset = dataset.dropna(subset=["PERF_MAX_APRES_SIGNAL"]).copy()
    dataset[TARGET_COLUMN] = (
        dataset["PERF_MAX_APRES_SIGNAL"] >= TARGET_THRESHOLD_PCT
    ).astype(int)

    if "HEURE" in dataset.columns:
        dataset["HEURE_DECIMALE"] = dataset["HEURE"].map(_hour_to_decimal)
    else:
        dataset["HEURE_DECIMALE"] = np.nan

    dataset["DATE"] = pd.to_datetime(dataset["DATE"], errors="coerce")
    dataset = dataset.dropna(subset=["DATE"]).sort_values(["DATE", "ID_SIGNAL"])

    for column in NUMERIC_FEATURES:
        if column not in dataset.columns:
            dataset[column] = np.nan
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column not in dataset.columns:
            dataset[column] = ""
        dataset[column] = dataset[column].fillna("").astype(str)

    return dataset.reset_index(drop=True)


def _time_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(dataset["DATE"].dt.date.unique())
    if len(dates) < 5:
        raise RuntimeError("Au moins 5 journées distinctes sont nécessaires.")
    split_index = max(1, int(len(dates) * 0.8))
    split_index = min(split_index, len(dates) - 1)
    validation_dates = set(dates[split_index:])
    return (
        dataset[~dataset["DATE"].dt.date.isin(validation_dates)].copy(),
        dataset[dataset["DATE"].dt.date.isin(validation_dates)].copy(),
    )


def _build_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def train_or_report(dataset: pd.DataFrame) -> dict[str, Any]:
    min_rows = int(os.environ.get("ML_MIN_REMAINING_ROWS", "60"))
    min_class = int(os.environ.get("ML_MIN_REMAINING_CLASS", "5"))
    positives = int(dataset[TARGET_COLUMN].sum()) if not dataset.empty else 0
    negatives = int(len(dataset) - positives)

    metrics: dict[str, Any] = {
        "target": f"PERF_MAX_APRES_SIGNAL >= {TARGET_THRESHOLD_PCT:.2f}%",
        "rows_total": int(len(dataset)),
        "positive_total": positives,
        "negative_total": negatives,
        "positive_rate": round(float(dataset[TARGET_COLUMN].mean()), 4) if len(dataset) else None,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "future_columns_excluded_from_features": OUTCOME_COLUMNS,
        "status": "waiting_for_data",
    }

    if len(dataset) < min_rows or positives < min_class or negatives < min_class:
        metrics["reason"] = (
            f"Données insuffisantes : lignes={len(dataset)}/{min_rows}, "
            f"positifs={positives}/{min_class}, négatifs={negatives}/{min_class}."
        )
        return metrics

    train_set, validation_set = _time_split(dataset)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = _build_pipeline()
    model.fit(train_set[features], train_set[TARGET_COLUMN])

    probabilities = model.predict_proba(validation_set[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y_true = validation_set[TARGET_COLUMN].to_numpy()

    metrics.update(
        {
            "status": "trained",
            "rows_train": int(len(train_set)),
            "rows_validation": int(len(validation_set)),
            "days_total": int(dataset["DATE"].dt.date.nunique()),
            "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
            "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
            "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4)
            if len(np.unique(y_true)) > 1
            else None,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN,
            "threshold_pct": TARGET_THRESHOLD_PCT,
        },
        MODEL_PATH,
    )
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals, follow_up, _trades = _load_via_apps_script()

    try:
        dataset = build_dataset(signals, follow_up)
    except RuntimeError as exc:
        pd.DataFrame().to_csv(DATASET_PATH, index=False)
        metrics = {
            "status": "waiting_for_columns",
            "reason": str(exc),
            "target": f"PERF_MAX_APRES_SIGNAL >= {TARGET_THRESHOLD_PCT:.2f}%",
        }
        METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(metrics["reason"])
        return

    dataset.to_csv(DATASET_PATH, index=False)
    metrics = train_or_report(dataset)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
