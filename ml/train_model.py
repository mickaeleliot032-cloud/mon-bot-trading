"""Entraînement ML hors ligne pour estimer la probabilité d'un Top 3 CAC 40."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gspread
import joblib
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUTPUT_DIR = Path("data/ml")
MODEL_PATH = OUTPUT_DIR / "top3_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
DATASET_PATH = OUTPUT_DIR / "dataset.csv"

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
]
CATEGORICAL_FEATURES = ["NIVEAU", "SECTEUR"]
TARGET_COLUMN = "TOP3"


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def _load_credentials() -> Credentials:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        raise RuntimeError("Secret GOOGLE_SERVICE_ACCOUNT_JSON absent.")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON n'est pas un JSON valide.") from exc

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    return Credentials.from_service_account_info(info, scopes=scopes)


def _load_worksheet(client: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    worksheet = client.open_by_key(sheet_id).worksheet(name)
    rows = worksheet.get_all_records(default_blank="")
    return _normalize_columns(pd.DataFrame(rows))


def _hour_to_decimal(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour + parsed.minute / 60 + parsed.second / 3600)


def build_dataset(signals: pd.DataFrame, follow_up: pd.DataFrame) -> pd.DataFrame:
    required_signal = {"ID_SIGNAL", "DATE"}
    required_follow = {"ID_SIGNAL", "RANG_FIN_JOURNEE"}
    missing_signal = required_signal.difference(signals.columns)
    missing_follow = required_follow.difference(follow_up.columns)
    if missing_signal:
        raise RuntimeError(f"Colonnes SIGNAUX manquantes : {sorted(missing_signal)}")
    if missing_follow:
        raise RuntimeError(f"Colonnes SUIVI manquantes : {sorted(missing_follow)}")

    suivi = follow_up[["ID_SIGNAL", "RANG_FIN_JOURNEE", "PERF_MAX_JOURNEE"]].copy()
    dataset = signals.merge(suivi, on="ID_SIGNAL", how="inner", validate="one_to_one")
    dataset["RANG_FIN_JOURNEE"] = pd.to_numeric(
        dataset["RANG_FIN_JOURNEE"], errors="coerce"
    )
    dataset = dataset.dropna(subset=["RANG_FIN_JOURNEE"]).copy()
    dataset[TARGET_COLUMN] = (dataset["RANG_FIN_JOURNEE"] <= 3).astype(int)

    if "HEURE" in dataset:
        dataset["HEURE_DECIMALE"] = dataset["HEURE"].map(_hour_to_decimal)
    else:
        dataset["HEURE_DECIMALE"] = np.nan

    dataset["DATE"] = pd.to_datetime(dataset["DATE"], errors="coerce")
    dataset = dataset.dropna(subset=["DATE"]).sort_values(["DATE", "ID_SIGNAL"])

    for column in NUMERIC_FEATURES:
        if column not in dataset:
            dataset[column] = np.nan
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column not in dataset:
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
    train = dataset[~dataset["DATE"].dt.date.isin(validation_dates)].copy()
    validation = dataset[dataset["DATE"].dt.date.isin(validation_dates)].copy()
    return train, validation


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


def train(dataset: pd.DataFrame) -> dict[str, Any]:
    min_rows = int(os.environ.get("ML_MIN_ROWS", "60"))
    min_positive = int(os.environ.get("ML_MIN_TOP3", "5"))
    positives = int(dataset[TARGET_COLUMN].sum())
    negatives = int(len(dataset) - positives)
    if len(dataset) < min_rows:
        raise RuntimeError(
            f"Dataset insuffisant : {len(dataset)} lignes, minimum {min_rows}."
        )
    if positives < min_positive or negatives < min_positive:
        raise RuntimeError(
            "Classes insuffisamment représentées : "
            f"Top3={positives}, hors Top3={negatives}."
        )

    train_set, validation_set = _time_split(dataset)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = _build_pipeline()
    model.fit(train_set[feature_columns], train_set[TARGET_COLUMN])

    probabilities = model.predict_proba(validation_set[feature_columns])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y_true = validation_set[TARGET_COLUMN].to_numpy()

    metrics: dict[str, Any] = {
        "rows_total": int(len(dataset)),
        "rows_train": int(len(train_set)),
        "rows_validation": int(len(validation_set)),
        "days_total": int(dataset["DATE"].dt.date.nunique()),
        "top3_total": positives,
        "top3_rate": round(float(dataset[TARGET_COLUMN].mean()), 4),
        "validation_accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "validation_precision": round(
            float(precision_score(y_true, predictions, zero_division=0)), 4
        ),
        "validation_recall": round(
            float(recall_score(y_true, predictions, zero_division=0)), 4
        ),
        "validation_brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "validation_baseline_top3_rate": round(float(np.mean(y_true)), 4),
        "validation_start": validation_set["DATE"].min().date().isoformat(),
        "validation_end": validation_set["DATE"].max().date().isoformat(),
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "target": "RANG_FIN_JOURNEE <= 3",
        "decision_threshold": 0.5,
    }
    if len(np.unique(y_true)) == 2:
        metrics["validation_roc_auc"] = round(
            float(roc_auc_score(y_true, probabilities)), 4
        )
    else:
        metrics["validation_roc_auc"] = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    dataset.to_csv(DATASET_PATH, index=False)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main() -> None:
    sheet_id = os.environ.get("ML_GOOGLE_SHEET_ID", "").strip()
    if not sheet_id:
        raise RuntimeError("Secret ML_GOOGLE_SHEET_ID absent.")

    client = gspread.authorize(_load_credentials())
    signals = _load_worksheet(client, sheet_id, "SIGNAUX")
    follow_up = _load_worksheet(client, sheet_id, "SUIVI")
    dataset = build_dataset(signals, follow_up)
    metrics = train(dataset)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Modèle enregistré : {MODEL_PATH}")


if __name__ == "__main__":
    main()
