import pandas as pd
from sklearn.ensemble import IsolationForest


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    alerts = []

    for _, row in df.iterrows():
        if row["refill_count"] > 3:
            alerts.append({
                "rule": "Excessive Refills",
                "patient_id": row["patient_id"],
                "drug": row["drug"],
                "severity": "High"
            })

        if row["dosage_mg"] > 200:
            alerts.append({
                "rule": "High Dosage",
                "patient_id": row["patient_id"],
                "drug": row["drug"],
                "severity": "Medium"
            })

        if row["days_supply"] < 10 and row["refill_count"] > 2:
            alerts.append({
                "rule": "Short Supply Abuse Pattern",
                "patient_id": row["patient_id"],
                "drug": row["drug"],
                "severity": "High"
            })

    return pd.DataFrame(alerts)


def run_ml_detector(df: pd.DataFrame, contamination: float = 0.15) -> pd.DataFrame:
    working_df = df.copy()

    features = working_df[["dosage_mg", "refill_count", "days_supply"]]

    model = IsolationForest(contamination=contamination, random_state=42)

    working_df["anomaly"] = model.fit_predict(features)
    working_df["anomaly_score"] = model.decision_function(features)

    anomalies = working_df[working_df["anomaly"] == -1].copy()

    return anomalies[
        ["patient_id", "drug", "dosage_mg", "refill_count", "days_supply", "anomaly_score"]
    ]