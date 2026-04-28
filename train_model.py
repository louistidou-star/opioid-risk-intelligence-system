"""Train the NORIP risk prediction model.

This script separates model training from the Streamlit app.
It creates a saved Random Forest model that the app can load from:

    models/risk_model.pkl
    models/feature_columns.json

The training pipeline uses available project data when present and augments it
with synthetic medication-behavior scenarios so the prototype has a stable,
reproducible model for demonstration purposes.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
BASE_CSV_PATH = Path("data/sample_events.csv")
DB_PATH = Path("opioid_risk.db")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "risk_model.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
TRAINING_DATA_PATH = MODEL_DIR / "training_dataset.csv"

FEATURE_COLUMNS = [
    "dosage_mg",
    "refill_count",
    "days_supply",
    "adherence_pct",
    "on_time_pct",
    "late_dose_count",
    "recent_missed_7d",
    "recent_missed_3d",
    "missed_streak",
    "late_streak",
    "timing_variability_hours",
    "early_redose_count",
    "frequent_dose_6h_count",
    "frequent_dose_24h_count",
    "nighttime_use_count",
    "long_gap_count",
    "dose_timing_alert_count",
    "medium_alert_count",
    "high_alert_count",
    "ml_alert_count",
    "anomaly_severity",
]


def load_base_csv() -> pd.DataFrame:
    if not BASE_CSV_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(BASE_CSV_PATH)
    except Exception:
        return pd.DataFrame()


def load_sqlite_events() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )["name"].tolist()
            if "events" not in tables:
                return pd.DataFrame()
            return pd.read_sql_query("SELECT * FROM events", conn)
    except Exception:
        return pd.DataFrame()


def create_project_feature_rows() -> pd.DataFrame:
    """Use available project data as part of the training set.

    The app creates many behavior features dynamically. If they are not present
    in raw CSV/database rows, they are filled with neutral defaults. The larger
    synthetic dataset below provides most of the behavioral variety.
    """
    frames = []
    base_df = load_base_csv()
    db_df = load_sqlite_events()

    for frame in [base_df, db_df]:
        if frame.empty:
            continue
        normalized = pd.DataFrame()
        for col in FEATURE_COLUMNS:
            normalized[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else 0
        frames.append(normalized.fillna(0))

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    return pd.concat(frames, ignore_index=True)[FEATURE_COLUMNS]


def generate_synthetic_behavior_rows(n_rows: int = 5000) -> pd.DataFrame:
    """Generate realistic prototype scenarios for model training.

    This is not clinical ground truth. It creates controlled demonstration data
    representing common behavioral patterns: normal adherence, missed doses,
    late doses, early redosing, frequent dosing, nighttime use, alert escalation,
    and ML anomaly signals.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    profiles = ["stable", "missed", "late", "redose", "frequent", "high_rx", "combined"]
    profile_probs = [0.35, 0.15, 0.12, 0.10, 0.08, 0.10, 0.10]

    for _ in range(n_rows):
        profile = rng.choice(profiles, p=profile_probs)

        dosage_mg = max(5, rng.normal(70, 35))
        refill_count = max(0, int(rng.poisson(1)))
        days_supply = max(1, int(rng.normal(20, 8)))
        adherence_pct = float(np.clip(rng.normal(92, 8), 40, 100))
        on_time_pct = float(np.clip(rng.normal(90, 10), 35, 100))
        late_dose_count = int(rng.poisson(0.5))
        recent_missed_7d = int(rng.poisson(0.3))
        recent_missed_3d = int(rng.binomial(2, 0.08))
        missed_streak = int(rng.binomial(3, 0.08))
        late_streak = int(rng.binomial(3, 0.08))
        timing_variability_hours = float(np.clip(rng.normal(1.2, 1.0), 0, 12))
        early_redose_count = 0
        frequent_dose_6h_count = 0
        frequent_dose_24h_count = 0
        nighttime_use_count = int(rng.poisson(0.2))
        long_gap_count = int(rng.binomial(2, 0.08))
        dose_timing_alert_count = 0
        medium_alert_count = int(rng.poisson(0.5))
        high_alert_count = 0
        ml_alert_count = int(rng.binomial(1, 0.08))
        anomaly_severity = float(rng.uniform(0, 0.35)) if ml_alert_count else 0.0

        if profile == "missed":
            adherence_pct = float(np.clip(rng.normal(58, 14), 10, 85))
            recent_missed_7d = int(rng.integers(1, 5))
            recent_missed_3d = int(rng.integers(0, 3))
            missed_streak = int(rng.integers(1, 4))
            medium_alert_count += int(rng.integers(1, 3))
        elif profile == "late":
            on_time_pct = float(np.clip(rng.normal(55, 18), 10, 85))
            late_dose_count = int(rng.integers(2, 8))
            late_streak = int(rng.integers(1, 4))
            timing_variability_hours = float(rng.uniform(4, 11))
            medium_alert_count += int(rng.integers(1, 3))
        elif profile == "redose":
            early_redose_count = int(rng.integers(1, 4))
            frequent_dose_6h_count = int(rng.integers(0, 3))
            dose_timing_alert_count = early_redose_count + frequent_dose_6h_count
            anomaly_severity = float(rng.uniform(0.45, 0.95))
            ml_alert_count = 1
        elif profile == "frequent":
            frequent_dose_6h_count = int(rng.integers(1, 4))
            frequent_dose_24h_count = int(rng.integers(1, 5))
            nighttime_use_count = int(rng.integers(1, 5))
            dose_timing_alert_count = frequent_dose_6h_count + frequent_dose_24h_count + nighttime_use_count
            anomaly_severity = float(rng.uniform(0.35, 0.9))
            ml_alert_count = 1
        elif profile == "high_rx":
            dosage_mg = float(rng.uniform(125, 280))
            refill_count = int(rng.integers(2, 7))
            days_supply = int(rng.integers(3, 12))
            medium_alert_count += int(rng.integers(1, 4))
            high_alert_count = int(rng.binomial(1, 0.55))
        elif profile == "combined":
            dosage_mg = float(rng.uniform(120, 260))
            refill_count = int(rng.integers(2, 7))
            adherence_pct = float(np.clip(rng.normal(50, 18), 5, 80))
            on_time_pct = float(np.clip(rng.normal(50, 20), 5, 80))
            late_dose_count = int(rng.integers(2, 8))
            recent_missed_7d = int(rng.integers(1, 5))
            missed_streak = int(rng.integers(1, 4))
            early_redose_count = int(rng.integers(0, 4))
            frequent_dose_24h_count = int(rng.integers(0, 4))
            nighttime_use_count = int(rng.integers(0, 5))
            long_gap_count = int(rng.integers(0, 3))
            dose_timing_alert_count = early_redose_count + frequent_dose_24h_count + nighttime_use_count + long_gap_count
            medium_alert_count = int(rng.integers(2, 6))
            high_alert_count = int(rng.binomial(1, 0.7))
            ml_alert_count = 1
            anomaly_severity = float(rng.uniform(0.55, 1.0))

        rows.append({
            "dosage_mg": dosage_mg,
            "refill_count": refill_count,
            "days_supply": days_supply,
            "adherence_pct": adherence_pct,
            "on_time_pct": on_time_pct,
            "late_dose_count": late_dose_count,
            "recent_missed_7d": recent_missed_7d,
            "recent_missed_3d": recent_missed_3d,
            "missed_streak": missed_streak,
            "late_streak": late_streak,
            "timing_variability_hours": timing_variability_hours,
            "early_redose_count": early_redose_count,
            "frequent_dose_6h_count": frequent_dose_6h_count,
            "frequent_dose_24h_count": frequent_dose_24h_count,
            "nighttime_use_count": nighttime_use_count,
            "long_gap_count": long_gap_count,
            "dose_timing_alert_count": dose_timing_alert_count,
            "medium_alert_count": medium_alert_count,
            "high_alert_count": high_alert_count,
            "ml_alert_count": ml_alert_count,
            "anomaly_severity": anomaly_severity,
        })

    return pd.DataFrame(rows)[FEATURE_COLUMNS]


def create_training_label(row: pd.Series) -> int:
    """Create prototype high-risk labels from known behavioral risk factors.

    This gives the model supervised examples for the public demo. In a clinical
    setting, this should be replaced by real outcome labels or expert-reviewed
    case labels.
    """
    score = 0

    if row["dosage_mg"] > 200:
        score += 20
    elif row["dosage_mg"] > 120:
        score += 10

    if row["refill_count"] > 3:
        score += 20
    elif row["refill_count"] >= 2:
        score += 10

    if row["days_supply"] < 10:
        score += 10

    if row["adherence_pct"] < 80:
        score += 10
    if row["adherence_pct"] < 60:
        score += 15
    if row["adherence_pct"] < 40:
        score += 15

    if row["on_time_pct"] < 80:
        score += 10
    if row["on_time_pct"] < 50:
        score += 10
    if row["late_dose_count"] >= 2:
        score += 10
    if row["late_streak"] >= 2:
        score += 10
    if row["timing_variability_hours"] >= 4:
        score += 10
    if row["timing_variability_hours"] >= 8:
        score += 10

    if row["early_redose_count"] >= 1:
        score += 20
    if row["early_redose_count"] >= 2:
        score += 10
    if row["frequent_dose_6h_count"] >= 1:
        score += 20
    if row["frequent_dose_24h_count"] >= 1:
        score += 15
    if row["nighttime_use_count"] >= 2:
        score += 10
    if row["long_gap_count"] >= 1:
        score += 10
    if row["dose_timing_alert_count"] >= 3:
        score += 10

    if row["recent_missed_7d"] >= 1:
        score += 10
    if row["recent_missed_7d"] >= 3:
        score += 10
    if row["recent_missed_3d"] >= 2:
        score += 10
    if row["missed_streak"] >= 2:
        score += 15

    if row["medium_alert_count"] >= 2:
        score += 15
    if row["medium_alert_count"] >= 3:
        score += 10
    if row["high_alert_count"] >= 1:
        score += 20
    if row["ml_alert_count"] >= 1:
        score += 10

    score += int(float(row["anomaly_severity"]) * 20)

    if row["adherence_pct"] < 80 and row["ml_alert_count"] >= 1:
        score += 10
    if row["early_redose_count"] >= 1 and row["anomaly_severity"] >= 0.5:
        score += 10

    return int(min(score, 100) >= 50)


def build_training_dataset() -> pd.DataFrame:
    project_rows = create_project_feature_rows()
    synthetic_rows = generate_synthetic_behavior_rows()
    training_df = pd.concat([project_rows, synthetic_rows], ignore_index=True)
    training_df[FEATURE_COLUMNS] = training_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    training_df["target_high_risk"] = training_df.apply(create_training_label, axis=1)
    return training_df


def train_and_save_model() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    training_df = build_training_dataset()

    X = training_df[FEATURE_COLUMNS]
    y = training_df["target_high_risk"]

    stratify = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "training_rows": int(len(training_df)),
        "positive_high_risk_rows": int(y.sum()),
        "negative_lower_risk_rows": int((1 - y).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "note": (
            "Prototype model trained on project/synthetic data with rule-derived labels. "
            "Replace target_high_risk with real outcome or expert-reviewed labels for clinical validation."
        ),
    }

    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    training_df.to_csv(TRAINING_DATA_PATH, index=False)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved features: {FEATURE_COLUMNS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Rows trained: {metrics['training_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    train_and_save_model()
