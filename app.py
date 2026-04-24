from datetime import date, datetime, time
import json

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from database import (
    init_db,
    insert_event,
    load_audit_log,
    load_events,
    log_audit,
    update_event,
)
from detection import run_ml_detector, run_rules


# ------------------------------------------------------------
# Page configuration and styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Opioid Risk Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
body {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.block-container > div {
    margin-bottom: 14px;
}

div[data-testid="stPlotlyChart"] {
    margin-bottom: 24px !important;
}

section[data-testid="stSidebar"] {
    background-color: #f5f7fb;
}

div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e6e9ef;
    padding: 14px;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 600;
}

.kpi-card {
    background: white;
    border: 1px solid #e8ecf2;
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    min-height: 105px;
}

.kpi-title {
    font-size: 0.85rem;
    color: #5b6575;
    margin-bottom: 8px;
    font-weight: 600;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #6b7280;
    margin-top: 8px;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    margin-right: 8px;
    margin-bottom: 8px;
}

.badge-stable {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}

.badge-monitor {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

.badge-high {
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.badge-critical {
    background: #3f0d12;
    color: #fecaca;
    border: 1px solid #7f1d1d;
}

.subtle {
    color: #6b7280;
    font-size: 0.92rem;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: #111827;
}

.section-sub {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def color_row(row):
    if row.get("severity") == "High":
        return ["background-color: red; color: white"] * len(row)
    if row.get("severity") == "Medium":
        return ["background-color: orange"] * len(row)
    return [""] * len(row)


def render_kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_status_label(risk_level: str) -> str:
    if risk_level == "Low":
        return "Stable"
    if risk_level == "Moderate":
        return "Monitor"
    if risk_level == "High":
        return "High Risk"
    if risk_level == "Critical":
        return "Critical"
    return "Unknown"


def render_status_badge(risk_level: str):
    badge_class = {
        "Low": "badge-stable",
        "Moderate": "badge-monitor",
        "High": "badge-high",
        "Critical": "badge-critical",
    }.get(risk_level, "badge-monitor")

    st.markdown(
        f'<span class="badge {badge_class}">{get_status_label(risk_level)}</span>',
        unsafe_allow_html=True,
    )


def validate_event_input(
    patient_id: str,
    submitted_by: str,
    drug: str,
    dose: float,
    status: str,
    zip_code: int,
    scheduled_date,
    scheduled_clock,
):
    errors = []
    warnings = []

    cleaned_patient_id = str(patient_id).strip()
    cleaned_submitted_by = str(submitted_by).strip()
    cleaned_drug = str(drug).strip()

    if not cleaned_patient_id:
        errors.append("Patient ID cannot be blank.")

    if not cleaned_drug:
        errors.append("Drug name cannot be blank.")

    if submitted_by and not cleaned_submitted_by:
        errors.append("Submitted By cannot contain only whitespace.")

    if status == "Taken" and dose <= 0:
        errors.append("Dose must be greater than 0 when status is Taken.")

    if status == "Missed" and dose > 0:
        warnings.append("Dose will be stored as 0 because status is Missed.")

    try:
        scheduled_dt = datetime.combine(scheduled_date, scheduled_clock)
    except Exception:
        errors.append("Scheduled date/time is invalid.")
        scheduled_dt = None

    if zip_code < 0:
        errors.append("ZIP Code cannot be negative.")
    elif zip_code == 0:
        warnings.append("ZIP Code looks unusual.")
    elif len(str(int(zip_code))) not in [5, 9]:
        warnings.append("ZIP Code does not look like a standard 5- or 9-digit ZIP code.")

    return {
        "errors": errors,
        "warnings": warnings,
        "cleaned_patient_id": cleaned_patient_id,
        "cleaned_submitted_by": cleaned_submitted_by if cleaned_submitted_by else "unknown",
        "cleaned_drug": cleaned_drug,
        "scheduled_dt": scheduled_dt,
    }


def build_patient_behavior_features(events_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "patient_id",
        "total_events",
        "taken_count",
        "missed_count",
        "on_time_count",
        "late_dose_count",
        "adherence_pct",
        "on_time_pct",
        "recent_missed_7d",
        "recent_missed_3d",
        "missed_streak",
        "late_streak",
        "timing_variability_hours",
        "last_reported_intake",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=output_columns)

    ev = events_df.copy()
    ev["patient_id"] = ev["patient_id"].astype(str).str.strip()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev["scheduled_time"] = pd.to_datetime(ev["scheduled_time"], errors="coerce")
    ev["taken_on_time"] = pd.to_numeric(ev.get("taken_on_time", 0), errors="coerce").fillna(0)

    now = pd.Timestamp.now()

    ev["recent_missed_7d_flag"] = (
        (ev["status"] == "Missed") & (ev["event_time"] >= now - pd.Timedelta(days=7))
    ).astype(int)

    ev["recent_missed_3d_flag"] = (
        (ev["status"] == "Missed") & (ev["event_time"] >= now - pd.Timedelta(days=3))
    ).astype(int)

    ev["timing_diff_hours"] = (
        (ev["event_time"] - ev["scheduled_time"]).dt.total_seconds() / 3600.0
    )
    ev.loc[ev["status"] != "Taken", "timing_diff_hours"] = pd.NA

    summary = ev.groupby("patient_id", dropna=True).agg(
        total_events=("patient_id", "size"),
        taken_count=("status", lambda s: (s == "Taken").sum()),
        missed_count=("status", lambda s: (s == "Missed").sum()),
        on_time_count=("taken_on_time", "sum"),
        recent_missed_7d=("recent_missed_7d_flag", "sum"),
        recent_missed_3d=("recent_missed_3d_flag", "sum"),
        timing_variability_hours=("timing_diff_hours", "std"),
        last_reported_intake=("event_time", "max"),
    ).reset_index()

    summary["late_dose_count"] = (summary["taken_count"] - summary["on_time_count"]).clip(lower=0)
    summary["adherence_pct"] = ((summary["taken_count"] / summary["total_events"].replace(0, 1)) * 100).round(1)
    summary["on_time_pct"] = ((summary["on_time_count"] / summary["taken_count"].replace(0, 1)) * 100).round(1)
    summary["timing_variability_hours"] = summary["timing_variability_hours"].fillna(0).round(2)

    def calc_missed_streak(patient_df):
        patient_df = patient_df.sort_values("event_time", ascending=False)
        streak = 0
        for status in patient_df["status"]:
            if status == "Missed":
                streak += 1
            else:
                break
        return streak

    def calc_late_streak(patient_df):
        patient_df = patient_df.sort_values("event_time", ascending=False)
        streak = 0
        for _, row in patient_df.iterrows():
            if row["status"] == "Taken" and row["taken_on_time"] == 0:
                streak += 1
            else:
                break
        return streak

    missed_streak_df = ev.groupby("patient_id", dropna=True).apply(calc_missed_streak).reset_index(name="missed_streak")
    late_streak_df = ev.groupby("patient_id", dropna=True).apply(calc_late_streak).reset_index(name="late_streak")

    summary = summary.merge(missed_streak_df, on="patient_id", how="left")
    summary = summary.merge(late_streak_df, on="patient_id", how="left")
    summary["missed_streak"] = summary["missed_streak"].fillna(0).astype(int)
    summary["late_streak"] = summary["late_streak"].fillna(0).astype(int)

    return summary[output_columns]


def build_risk_trend(events_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "patient_id",
        "date",
        "event_count",
        "missed_count",
        "late_dose_count",
        "daily_behavior_score",
        "risk_score",
        "rolling_risk_score",
        "risk_delta",
        "risk_spike_flag",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=output_columns)

    if "patient_id" not in events_df.columns or "event_time" not in events_df.columns:
        return pd.DataFrame(columns=output_columns)

    ev = events_df.copy()
    ev["patient_id"] = ev["patient_id"].astype(str).str.strip()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev["scheduled_time"] = pd.to_datetime(ev.get("scheduled_time"), errors="coerce")
    ev["taken_on_time"] = pd.to_numeric(ev.get("taken_on_time", 0), errors="coerce").fillna(0)
    ev = ev.dropna(subset=["event_time"])

    if ev.empty:
        return pd.DataFrame(columns=output_columns)

    ev["date"] = ev["event_time"].dt.date
    ev["is_missed"] = (ev["status"] == "Missed").astype(int)
    ev["is_late"] = ((ev["status"] == "Taken") & (ev["taken_on_time"] == 0)).astype(int)

    daily = ev.groupby(["patient_id", "date"], as_index=False).agg(
        event_count=("patient_id", "size"),
        missed_count=("is_missed", "sum"),
        late_dose_count=("is_late", "sum"),
    )

    daily["daily_behavior_score"] = (
        daily["missed_count"] * 20
        + daily["late_dose_count"] * 12
        + daily["event_count"].clip(lower=0).sub(1).clip(lower=0) * 5
    ).clip(upper=100)

    if not scored_df.empty and "patient_id" in scored_df.columns and "risk_score" in scored_df.columns:
        scored = scored_df[["patient_id", "risk_score"]].copy()
        scored["patient_id"] = scored["patient_id"].astype(str).str.strip()
        scored["risk_score"] = pd.to_numeric(scored["risk_score"], errors="coerce").fillna(0)
        patient_current_risk = (
            scored.groupby("patient_id", as_index=False)["risk_score"]
            .max()
            .rename(columns={"risk_score": "current_risk_score"})
        )
        daily = daily.merge(patient_current_risk, on="patient_id", how="left")
    else:
        daily["current_risk_score"] = 0

    daily["current_risk_score"] = daily["current_risk_score"].fillna(0)

    daily["risk_score"] = (
        daily["current_risk_score"] * 0.55 + daily["daily_behavior_score"] * 0.45
    ).round(1).clip(upper=100)

    daily = daily.sort_values(["patient_id", "date"])
    daily["rolling_risk_score"] = (
        daily.groupby("patient_id")["risk_score"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        .round(1)
    )
    daily["risk_delta"] = (
        daily.groupby("patient_id")["rolling_risk_score"]
        .diff()
        .fillna(0)
        .round(1)
    )

    daily["risk_spike_flag"] = (
        (daily["risk_delta"] >= 12)
        | ((daily["daily_behavior_score"] >= 40) & ((daily["missed_count"] > 0) | (daily["late_dose_count"] > 0)))
    )

    return daily[output_columns]


def build_alert_features(rules_alerts: pd.DataFrame, ml_alerts: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if not rules_alerts.empty:
        rules_summary = rules_alerts.groupby("patient_id", dropna=True).agg(
            total_rule_alerts=("patient_id", "size"),
            high_alert_count=("severity", lambda s: (s == "High").sum()),
            medium_alert_count=("severity", lambda s: (s == "Medium").sum()),
        ).reset_index()

        rules_summary["medium_alert_escalation"] = rules_summary["medium_alert_count"].apply(
            lambda x: 0 if x < 2 else 1 if x < 4 else 2
        )
        frames.append(rules_summary)

    if not ml_alerts.empty:
        ml = ml_alerts.copy()
        if "anomaly_score" in ml.columns:
            score_min = ml["anomaly_score"].min()
            score_max = ml["anomaly_score"].max()
            denom = (score_max - score_min) if score_max != score_min else 1
            ml["anomaly_severity"] = ((score_max - ml["anomaly_score"]) / denom).clip(0, 1)
        else:
            ml["anomaly_severity"] = 0.5

        ml_summary = ml.groupby("patient_id", dropna=True).agg(
            ml_alert_count=("patient_id", "size"),
            anomaly_severity=("anomaly_severity", "max"),
        ).reset_index()
        frames.append(ml_summary)

    if not frames:
        return pd.DataFrame(columns=[
            "patient_id",
            "total_rule_alerts",
            "high_alert_count",
            "medium_alert_count",
            "medium_alert_escalation",
            "ml_alert_count",
            "anomaly_severity",
        ])

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="patient_id", how="outer")

    return result.fillna(0)


def compute_enhanced_risk_scores(
    df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    alert_features_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["patient_id"] = enriched["patient_id"].astype(str).str.strip()

    if not behavior_df.empty:
        enriched = enriched.merge(behavior_df, on="patient_id", how="left")
    if not alert_features_df.empty:
        enriched = enriched.merge(alert_features_df, on="patient_id", how="left")

    fill_defaults = {
        "total_events": 0,
        "taken_count": 0,
        "missed_count": 0,
        "on_time_count": 0,
        "late_dose_count": 0,
        "adherence_pct": 100,
        "on_time_pct": 100,
        "recent_missed_7d": 0,
        "recent_missed_3d": 0,
        "missed_streak": 0,
        "late_streak": 0,
        "timing_variability_hours": 0,
        "total_rule_alerts": 0,
        "high_alert_count": 0,
        "medium_alert_count": 0,
        "medium_alert_escalation": 0,
        "ml_alert_count": 0,
        "anomaly_severity": 0,
    }

    for col, default in fill_defaults.items():
        if col not in enriched.columns:
            enriched[col] = default
        else:
            enriched[col] = enriched[col].fillna(default)

    numeric_cols = ["dosage_mg", "refill_count", "days_supply"] + list(fill_defaults.keys())
    for col in numeric_cols:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0)

    def score_row(row):
        score = 0
        reasons = []
        components = {}

        def add_points(name, pts, reason):
            nonlocal score
            if pts > 0:
                score += pts
                components[name] = components.get(name, 0) + pts
                reasons.append(reason)

        if row["dosage_mg"] > 200:
            add_points("dosage", 20, "high dosage")
        elif row["dosage_mg"] > 120:
            add_points("dosage", 10, "elevated dosage")

        if row["refill_count"] > 3:
            add_points("refills", 20, "excessive refills")
        elif row["refill_count"] >= 2:
            add_points("refills", 10, "repeat refills")

        if row["days_supply"] < 10:
            add_points("supply_pattern", 10, "short supply pattern")

        if row["adherence_pct"] < 80:
            add_points("adherence", 10, "adherence below 80%")
        if row["adherence_pct"] < 60:
            add_points("adherence", 15, "adherence below 60%")
        if row["adherence_pct"] < 40:
            add_points("adherence", 15, "severely low adherence")

        if row["late_dose_count"] >= 2:
            add_points("timing", 10, "repeated late doses")
        if row["late_streak"] >= 2:
            add_points("timing", 10, "consecutive late doses")
        if row["on_time_pct"] < 80:
            add_points("timing", 10, "low on-time adherence")
        if row["on_time_pct"] < 50:
            add_points("timing", 10, "poor timing consistency")
        if row["timing_variability_hours"] >= 4:
            add_points("timing", 10, "high dose timing variability")
        if row["timing_variability_hours"] >= 8:
            add_points("timing", 10, "extreme dose timing variability")

        if row["missed_count"] >= 2:
            add_points("missed_doses", 10, "multiple missed doses")
        if row["recent_missed_7d"] >= 1:
            add_points("missed_doses", 10, "recent missed dose")
        if row["recent_missed_7d"] >= 3:
            add_points("missed_doses", 10, "frequent missed doses in last 7 days")
        if row["recent_missed_3d"] >= 2:
            add_points("missed_doses", 10, "clustered recent missed doses")
        if row["missed_streak"] >= 2:
            add_points("missed_doses", 15, "consecutive missed-dose streak")
        if row["missed_streak"] >= 3:
            add_points("missed_doses", 10, "prolonged missed-dose streak")

        if row["medium_alert_count"] >= 2:
            add_points("alerts", 15, "repeated medium alerts")
        if row["medium_alert_count"] >= 3:
            add_points("alerts", 10, "medium alerts escalating toward high risk")
        if row["medium_alert_escalation"] == 2:
            add_points("alerts", 10, "persistent medium-alert escalation")
        if row["high_alert_count"] >= 1:
            add_points("alerts", 20, "existing high-severity alert")

        if row["ml_alert_count"] >= 1:
            add_points("anomaly", 10, "ml anomaly detected")

        anomaly_points = int(float(row["anomaly_severity"]) * 20)
        if anomaly_points > 0:
            add_points("anomaly", anomaly_points, "anomaly severity contribution")

        if row["adherence_pct"] < 80 and row["ml_alert_count"] >= 1:
            add_points("interaction", 10, "low adherence + anomaly combination")
        if row["adherence_pct"] < 60 and row["ml_alert_count"] >= 1:
            add_points("interaction", 10, "high-risk adherence + anomaly combination")
        if row["recent_missed_7d"] >= 2 and row["anomaly_severity"] >= 0.5:
            add_points("interaction", 10, "missed-dose cluster + anomaly combination")
        if row["medium_alert_count"] >= 3 and row["on_time_pct"] < 80:
            add_points("interaction", 10, "alert escalation + poor timing combination")

        recency_boost = 0
        if row["recent_missed_3d"] >= 1:
            recency_boost += 5
        if row["missed_streak"] >= 2:
            recency_boost += 5
        if row["late_streak"] >= 2:
            recency_boost += 5

        if recency_boost > 0:
            add_points("recency", recency_boost, "recent behavior risk weighting")

        score = min(score, 100)

        if score >= 75:
            level = "Critical"
        elif score >= 50:
            level = "High"
        elif score >= 25:
            level = "Moderate"
        else:
            level = "Low"

        return pd.Series({
            "risk_score": score,
            "risk_level": level,
            "risk_reasons": ", ".join(sorted(set(reasons))) if reasons else "baseline risk",
            "risk_components": ", ".join([f"{k}:{v}" for k, v in sorted(components.items())]) if components else "baseline:0",
        })

    scored = enriched.apply(score_row, axis=1)
    enriched["risk_score"] = scored["risk_score"]
    enriched["risk_level"] = scored["risk_level"]
    enriched["risk_reasons"] = scored["risk_reasons"]
    enriched["risk_components"] = scored["risk_components"]

    return enriched


def run_random_forest_risk_model(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()

    feature_cols = [
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
        "medium_alert_count",
        "high_alert_count",
        "ml_alert_count",
        "anomaly_severity",
    ]

    available_features = [col for col in feature_cols if col in model_df.columns]
    if not available_features:
        model_df["rf_risk_probability"] = 0.0
        model_df["rf_prediction"] = "Unavailable"
        return model_df

    model_df[available_features] = model_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    model_df["rf_target"] = model_df["risk_level"].isin(["High", "Critical"]).astype(int)

    if model_df["rf_target"].nunique() < 2:
        model_df["rf_risk_probability"] = 0.0
        model_df["rf_prediction"] = "Unavailable"
        return model_df

    X = model_df[available_features]
    y = model_df["rf_target"]

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    rf_model.fit(X, y)

    probabilities = rf_model.predict_proba(X)[:, 1]
    model_df["rf_risk_probability"] = probabilities.round(3)
    model_df["rf_prediction"] = model_df["rf_risk_probability"].apply(
        lambda p: "High Risk Likely" if p >= 0.5 else "Lower Risk Likely"
    )

    feature_importance = pd.DataFrame({
        "feature": available_features,
        "importance": rf_model.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    model_df.attrs["rf_feature_importance"] = feature_importance
    return model_df


def get_top_risk_factors(risk_components: str, top_n: int = 3):
    factors = []

    if not isinstance(risk_components, str) or risk_components.strip() in ["", "baseline:0"]:
        return factors

    for item in risk_components.split(","):
        item = item.strip()
        if ":" not in item:
            continue
        factor, points = item.split(":", 1)
        try:
            factors.append({"factor": factor.strip(), "points": float(points.strip())})
        except ValueError:
            continue

    return sorted(factors, key=lambda x: x["points"], reverse=True)[:top_n]


def parse_risk_components_series(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if "risk_components" not in df.columns:
        return pd.DataFrame(columns=["patient_id", "factor", "points"])

    for _, row in df.iterrows():
        components = row.get("risk_components", "")
        if not isinstance(components, str) or components.strip() in ["", "baseline:0"]:
            continue

        for item in components.split(","):
            item = item.strip()
            if ":" not in item:
                continue
            factor, points = item.split(":", 1)
            try:
                rows.append({
                    "patient_id": row.get("patient_id", ""),
                    "factor": factor.strip(),
                    "points": float(points.strip()),
                })
            except ValueError:
                continue

    return pd.DataFrame(rows)


def safe_parse_json(value):
    if pd.isna(value) or value in ["", None]:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def normalize_value(value):
    if pd.isna(value):
        return None
    return value


def get_changed_fields(prev_dict, new_dict):
    all_keys = sorted(set(prev_dict.keys()) | set(new_dict.keys()))
    changed = []
    for key in all_keys:
        if normalize_value(prev_dict.get(key)) != normalize_value(new_dict.get(key)):
            changed.append(key)
    return changed


def highlight_changes(row_style):
    return [
        "background-color: #fee2e2; color: #991b1b;" if col == "previous"
        else "background-color: #dcfce7; color: #166534;" if col == "new"
        else ""
        for col in row_style.index
    ]


# ------------------------------------------------------------
# Data loading and detection pipeline
# ------------------------------------------------------------
init_db()

st.sidebar.title("System Controls")
st.sidebar.markdown("SIEM-based Risk Monitoring")

mode = st.sidebar.selectbox(
    "Application Mode",
    ["Demo Overview", "Patient View", "Analyst View"],
)

uploaded_file = st.sidebar.file_uploader("Upload Base CSV", type=["csv"])
risk_threshold = st.sidebar.slider("ML Sensitivity", 0.05, 0.5, 0.15)

st.sidebar.subheader("System Status")
st.sidebar.success("SIEM Engine: ACTIVE")

st.title("National Opioid Risk Intelligence Platform (NORIP)")
st.caption("Cybersecurity-driven anomaly detection and public health monitoring platform")

try:
    if uploaded_file is not None:
        base_df = pd.read_csv(uploaded_file)
    else:
        base_df = pd.read_csv("data/sample_events.csv")
except Exception as e:
    st.error(f"Error loading base data: {e}")
    st.stop()

required_columns = [
    "patient_id",
    "prescriber_id",
    "drug",
    "dosage_mg",
    "days_supply",
    "refill_count",
    "zip",
]

missing_columns = [col for col in required_columns if col not in base_df.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

base_df["patient_id"] = base_df["patient_id"].astype(str).str.strip()

events_df = load_events()

if not events_df.empty:
    events_df["patient_id"] = events_df["patient_id"].astype(str).str.strip()
    if "submitted_by" in events_df.columns:
        events_df["submitted_by"] = events_df["submitted_by"].replace("", pd.NA).fillna("legacy record")

if not events_df.empty:
    db_analysis_df = events_df[[
        "patient_id",
        "prescriber_id",
        "drug",
        "dosage_mg",
        "days_supply",
        "refill_count",
        "zip",
    ]].copy()
else:
    db_analysis_df = pd.DataFrame(columns=required_columns)

df = pd.concat([base_df, db_analysis_df], ignore_index=True)

try:
    rules_alerts = run_rules(df)
    ml_alerts = run_ml_detector(df, contamination=risk_threshold)
except Exception as e:
    st.error(f"Error running detection: {e}")
    st.stop()

behavior_df = build_patient_behavior_features(events_df)
alert_features_df = build_alert_features(rules_alerts, ml_alerts)
df = compute_enhanced_risk_scores(df, behavior_df, alert_features_df)
df = run_random_forest_risk_model(df)
risk_trend_df = build_risk_trend(events_df, df)

if not events_df.empty:
    events_df["event_time"] = pd.to_datetime(events_df["event_time"], errors="coerce")
    events_df["scheduled_time"] = pd.to_datetime(events_df["scheduled_time"], errors="coerce")
    taken_events = events_df[events_df["status"] == "Taken"]
    missed_events = events_df[events_df["status"] == "Missed"]

    on_time_count = int(events_df["taken_on_time"].fillna(0).sum()) if "taken_on_time" in events_df.columns else 0
    missed_count = len(missed_events)
    total_expected = len(events_df)
    adherence_percentage = round((len(taken_events) / total_expected) * 100, 1) if total_expected > 0 else 0.0

    last_reported_intake_raw = events_df["event_time"].max()
    last_reported_intake = (
        last_reported_intake_raw.strftime("%Y-%m-%d %H:%M")
        if pd.notna(last_reported_intake_raw)
        else "No events recorded"
    )
else:
    on_time_count = 0
    missed_count = 0
    adherence_percentage = 0.0
    last_reported_intake = "No events recorded"

high_risk_alerts = len(rules_alerts[rules_alerts["severity"] == "High"]) if not rules_alerts.empty else 0
high_risk_patients = len(df[df["risk_level"].isin(["High", "Critical"])]) if not df.empty else 0

st.sidebar.info(f"Records Loaded: {len(df)}")
st.sidebar.info(f"Saved Patient Events: {len(events_df)}")
st.sidebar.warning(f"Alerts Generated: {len(rules_alerts)}")


# ------------------------------------------------------------
# Demo Overview
# ------------------------------------------------------------
if mode == "Demo Overview":
    st.subheader("System Overview")
    st.write(
        "This prototype demonstrates a two-sided healthcare risk intelligence workflow: "
        "patients submit medication intake events, and analysts review alerts, anomalies, "
        "adherence trends, and risk patterns through a monitoring dashboard."
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Records", len(df), "All combined records")
    with k2:
        render_kpi_card("Saved Events", len(events_df), "Logged by patients")
    with k3:
        render_kpi_card("High-Risk Alerts", high_risk_alerts, "Rule-based severe alerts")
    with k4:
        render_kpi_card("ML Anomalies", len(ml_alerts), "Detected anomalous cases")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        render_kpi_card("Doses On Time", on_time_count, "Recorded on-time doses")
    with k6:
        render_kpi_card("Missed Doses", missed_count, "All missed dose events")
    with k7:
        render_kpi_card("Adherence %", f"{adherence_percentage}%", "Across saved events")
    with k8:
        render_kpi_card("High-Risk Patients", high_risk_patients, "High + Critical")

    st.markdown("---")
    st.markdown('<div class="section-title">Top High-Risk Patients</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Highest priority patients based on the enhanced behavioral risk model.</div>', unsafe_allow_html=True)

    top_risk = df.sort_values(by="risk_score", ascending=False).head(10)[
        ["patient_id", "drug", "risk_score", "risk_level", "risk_reasons", "risk_components"]
    ]
    st.dataframe(top_risk, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Recent Submitted Medication Events</div>', unsafe_allow_html=True)
    if events_df.empty:
        st.info("No medication events have been submitted yet.")
    else:
        st.dataframe(events_df.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Risk Analytics Snapshot</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(df, x="dosage_mg", title="Dosage Distribution", labels={"dosage_mg": "Dosage (mg)"})
        fig1.update_layout(height=380)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(df, x="risk_score", nbins=10, title="Risk Score Distribution", labels={"risk_score": "Risk Score"})
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------------------
# Patient View
# ------------------------------------------------------------
elif mode == "Patient View":
    st.subheader("Medication Intake Logging")
    st.write("Submit a medication intake event to update adherence tracking and patient risk visibility.")

    with st.form("patient_form"):
        p_id = st.text_input("Patient ID")
        submitted_by = st.text_input("Submitted By")
        drug = st.text_input("Drug")
        dose = st.number_input("Dose Taken (mg)", min_value=0.0, step=1.0)
        status = st.selectbox("Dose Status", ["Taken", "Missed"])
        zip_code = st.number_input("ZIP Code", min_value=0, step=1)
        scheduled_date = st.date_input("Scheduled Dose Date", value=date.today())
        scheduled_clock = st.time_input("Scheduled Dose Time", value=time(8, 0))
        submit = st.form_submit_button("Submit Event")

    if submit:
        validation = validate_event_input(
            patient_id=p_id,
            submitted_by=submitted_by,
            drug=drug,
            dose=dose,
            status=status,
            zip_code=int(zip_code),
            scheduled_date=scheduled_date,
            scheduled_clock=scheduled_clock,
        )

        for warning in validation["warnings"]:
            st.warning(warning)

        if validation["errors"]:
            for error in validation["errors"]:
                st.error(error)
        else:
            cleaned_patient_id = validation["cleaned_patient_id"]
            cleaned_drug = validation["cleaned_drug"]
            cleaned_submitter = validation["cleaned_submitted_by"]
            scheduled_dt = validation["scheduled_dt"].strftime("%Y-%m-%d %H:%M:%S")
            final_dose = dose if status == "Taken" else 0

            event_payload = {
                "drug": cleaned_drug,
                "dosage_mg": final_dose,
                "days_supply": 1,
                "refill_count": 0,
                "zip_code": int(zip_code),
                "status": status,
                "scheduled_time": scheduled_dt,
            }

            event_id = insert_event(
                patient_id=cleaned_patient_id,
                prescriber_id="D_SIM",
                drug=cleaned_drug,
                dosage_mg=final_dose,
                days_supply=1,
                refill_count=0,
                zip_code=int(zip_code),
                status=status,
                scheduled_time=scheduled_dt,
                submitted_by=cleaned_submitter,
            )

            log_audit(
                event_id=event_id,
                patient_id=cleaned_patient_id,
                action="CREATE",
                submitted_by=cleaned_submitter,
                previous_value={},
                new_value=event_payload,
            )

            st.success("Medication event recorded successfully.")
            st.rerun()

    st.markdown("---")
    st.subheader("Your Medication History")

    if events_df.empty:
        st.info("No records yet.")
    else:
        patient_id_input = p_id.strip()

        if patient_id_input:
            patient_history = events_df[events_df["patient_id"] == patient_id_input].copy()

            if patient_history.empty:
                st.info("No records found for this patient.")
            else:
                patient_history["event_time"] = pd.to_datetime(patient_history["event_time"], errors="coerce")
                patient_history["scheduled_time"] = pd.to_datetime(patient_history["scheduled_time"], errors="coerce")
                patient_history = patient_history.sort_values("event_time", ascending=False)

                taken = patient_history[patient_history["status"] == "Taken"]
                missed = patient_history[patient_history["status"] == "Missed"]
                adherence = round((len(taken) / len(patient_history)) * 100, 1)

                patient_last_intake_raw = patient_history["event_time"].max()
                patient_last_intake = (
                    patient_last_intake_raw.strftime("%Y-%m-%d %H:%M")
                    if pd.notna(patient_last_intake_raw)
                    else "No events recorded"
                )

                patient_risk_profile = df[df["patient_id"] == patient_id_input].copy()
                if not patient_risk_profile.empty:
                    patient_risk_row = patient_risk_profile.sort_values("risk_score", ascending=False).iloc[0]
                    patient_risk_label = get_status_label(patient_risk_row["risk_level"])
                else:
                    patient_risk_label = "Unknown"

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Adherence %", adherence)
                s2.metric("Missed Doses", len(missed))
                s3.metric("Last Intake", patient_last_intake)
                s4.metric("Current Risk", patient_risk_label)

                st.dataframe(patient_history, use_container_width=True)

                fig_patient_timeline = px.scatter(
                    patient_history.sort_values("event_time"),
                    x="event_time",
                    y="drug",
                    color="status",
                    hover_data=["dosage_mg", "scheduled_time", "taken_on_time"],
                    title=f"Medication Timeline for {patient_id_input}",
                    labels={"event_time": "Event Time", "drug": "Medication"},
                )
                fig_patient_timeline.update_layout(height=420)
                st.plotly_chart(fig_patient_timeline, use_container_width=True)
        else:
            st.info("Enter your Patient ID above to view your history.")


# ------------------------------------------------------------
# Analyst View
# ------------------------------------------------------------
elif mode == "Analyst View":
    st.subheader("Analyst Dashboard")
    st.write("Monitor risk, adherence behavior, alerts, and patient-level summaries.")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Total Records", len(df), "All analyzed rows")
    with k2:
        render_kpi_card("Active Alerts", len(rules_alerts), "Rule-based alerts")
    with k3:
        render_kpi_card("High-Risk Patients", high_risk_patients, "High + Critical")
    with k4:
        render_kpi_card("Adherence", f"{adherence_percentage}%", "Global adherence")
    with k5:
        render_kpi_card("ML Anomalies", len(ml_alerts), "Anomaly engine output")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Patient Monitoring",
        "Alerts",
        "Risk Analytics",
        "Reports",
        "Audit Trail",
    ])

    with tab1:
        st.markdown('<div class="section-title">Operational Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">High-level monitoring of risk posture and current system activity.</div>', unsafe_allow_html=True)

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Doses On Time", on_time_count)
        o2.metric("Missed Doses", missed_count)
        o3.metric("High Severity Alerts", high_risk_alerts)
        o4.metric("Last Intake", last_reported_intake)

        st.markdown("#### Risk Spike Watchlist")
        if risk_trend_df.empty or "risk_spike_flag" not in risk_trend_df.columns:
            st.info("No risk spike data available yet.")
        else:
            spike_watchlist = risk_trend_df[risk_trend_df["risk_spike_flag"] == True].sort_values(
                ["date", "risk_delta"], ascending=[False, False]
            )
            if spike_watchlist.empty:
                st.success("No active risk spikes detected.")
            else:
                st.dataframe(
                    spike_watchlist[
                        [
                            "patient_id",
                            "date",
                            "event_count",
                            "missed_count",
                            "late_dose_count",
                            "daily_behavior_score",
                            "rolling_risk_score",
                            "risk_delta",
                        ]
                    ].head(10),
                    use_container_width=True,
                )

        st.markdown("#### Highest-Risk Patients")
        top_risk = df.sort_values(by="risk_score", ascending=False).head(10)[
            ["patient_id", "drug", "risk_score", "risk_level", "risk_reasons", "risk_components"]
        ]
        st.dataframe(top_risk, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="risk_score", nbins=10, title="Risk Score Distribution", labels={"risk_score": "Risk Score"})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            risk_counts["Risk Level"] = pd.Categorical(
                risk_counts["Risk Level"],
                categories=["Low", "Moderate", "High", "Critical"],
                ordered=True,
            )
            risk_counts = risk_counts.sort_values("Risk Level")

            fig = px.bar(risk_counts, x="Risk Level", y="Count", title="Patients by Risk Level", labels={"Count": "Patient Count"})
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Patient Monitoring</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Select a patient to review adherence, timing behavior, and patient-specific risk details.</div>',
            unsafe_allow_html=True,
        )

        if events_df.empty:
            st.info("No medication events have been submitted yet.")
        else:
            patient_options = sorted(events_df["patient_id"].replace("", pd.NA).dropna().unique().tolist())

            if not patient_options:
                st.warning("Patient events exist but patient IDs are missing or invalid.")
                st.dataframe(events_df.head(), use_container_width=True)
            else:
                selected_patient = st.selectbox(
                    "Select Patient",
                    patient_options,
                    help="Choose a patient to inspect event history, adherence, and risk profile.",
                )

                patient_events = events_df[events_df["patient_id"] == selected_patient].copy()
                patient_profile = df[df["patient_id"] == selected_patient].copy()

                if patient_events.empty:
                    st.info("No events found for this patient.")
                else:
                    patient_events["event_time"] = pd.to_datetime(patient_events["event_time"], errors="coerce")
                    patient_events["scheduled_time"] = pd.to_datetime(patient_events["scheduled_time"], errors="coerce")
                    patient_events = patient_events.sort_values("event_time", ascending=False)

                    patient_taken = patient_events[patient_events["status"] == "Taken"]
                    patient_missed = patient_events[patient_events["status"] == "Missed"]
                    patient_on_time = int(patient_events["taken_on_time"].fillna(0).sum())
                    patient_total = len(patient_events)
                    patient_adherence = round((len(patient_taken) / patient_total) * 100, 1) if patient_total > 0 else 0.0

                    patient_last_intake_raw = patient_events["event_time"].max()
                    patient_last_intake = (
                        patient_last_intake_raw.strftime("%Y-%m-%d %H:%M")
                        if pd.notna(patient_last_intake_raw)
                        else "No events recorded"
                    )

                    if not patient_profile.empty:
                        patient_summary = patient_profile.sort_values("risk_score", ascending=False).iloc[0]
                        risk_level = patient_summary["risk_level"]
                        risk_score = patient_summary["risk_score"]
                        risk_reasons = patient_summary["risk_reasons"]
                        risk_components = patient_summary["risk_components"]
                    else:
                        patient_summary = pd.Series(dtype="object")
                        risk_level = "Moderate"
                        risk_score = 0
                        risk_reasons = "No risk profile available"
                        risk_components = "baseline:0"

                    left, right = st.columns([1.3, 2.2])

                    with left:
                        st.markdown(f"### Patient {selected_patient}")
                        render_status_badge(risk_level)
                        st.caption(f"Model Risk Level: {risk_level}")
                        st.metric("Risk Score", int(risk_score))
                        st.metric("Adherence %", patient_adherence)
                        st.metric("Missed Doses", len(patient_missed))
                        st.metric("Taken On Time", patient_on_time)

                        if "rf_risk_probability" in patient_summary.index:
                            st.metric(
                                "RF Risk Probability",
                                f"{patient_summary['rf_risk_probability'] * 100:.1f}%",
                            )
                            st.caption(f"ML Prediction: {patient_summary.get('rf_prediction', 'Unavailable')}")

                        st.caption(f"Last reported intake: {patient_last_intake}")
                        st.markdown("**Primary Risk Reasons**")
                        st.write(risk_reasons)
                        st.markdown("**Risk Components**")
                        st.code(risk_components)

                        top_factors = get_top_risk_factors(risk_components)
                        st.markdown("**Top 3 Risk Factors**")
                        if top_factors:
                            for factor in top_factors:
                                st.write(f"- {factor['factor']}: +{int(factor['points'])} points")
                        else:
                            st.caption("No elevated risk factors detected.")

                    with right:
                        history_tab, timeline_tab = st.tabs(["Event History", "Timeline & Status"])

                        with history_tab:
                            display_cols = [c for c in [
                                "id",
                                "patient_id",
                                "drug",
                                "status",
                                "dosage_mg",
                                "days_supply",
                                "refill_count",
                                "zip",
                                "event_time",
                                "scheduled_time",
                                "taken_on_time",
                                "submitted_by",
                            ] if c in patient_events.columns]
                            st.dataframe(patient_events[display_cols], use_container_width=True)

                        with timeline_tab:
                            tc1, tc2 = st.columns(2)

                            with tc1:
                                fig_timeline = px.scatter(
                                    patient_events.sort_values("event_time"),
                                    x="event_time",
                                    y="drug",
                                    color="status",
                                    hover_data=["dosage_mg", "scheduled_time", "taken_on_time"],
                                    title=f"Medication Timeline for {selected_patient}",
                                    labels={"event_time": "Event Time", "drug": "Medication"},
                                )
                                fig_timeline.update_layout(height=420)
                                st.plotly_chart(fig_timeline, use_container_width=True)

                            with tc2:
                                status_counts = patient_events["status"].value_counts().reset_index()
                                status_counts.columns = ["Dose Status", "Count"]
                                fig_status = px.bar(
                                    status_counts,
                                    x="Dose Status",
                                    y="Count",
                                    title="Dose Status Breakdown",
                                    labels={"Count": "Number of Events"},
                                )
                                fig_status.update_layout(height=420)
                                st.plotly_chart(fig_status, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Risk Trend Over Time")

                    patient_trend = risk_trend_df[
                        risk_trend_df["patient_id"] == selected_patient
                    ].copy()

                    if patient_trend.empty:
                        st.info("No risk trend data available for this patient yet.")
                    else:
                        fig_risk_trend = px.line(
                            patient_trend,
                            x="date",
                            y=["risk_score", "rolling_risk_score"],
                            markers=True,
                            title=f"Risk Trend Over Time for {selected_patient}",
                            labels={"date": "Date", "value": "Risk Score", "variable": "Metric"},
                        )
                        fig_risk_trend.add_hline(y=25, line_dash="dash")
                        fig_risk_trend.add_hline(y=50, line_dash="dash")
                        fig_risk_trend.add_hline(y=75, line_dash="dash")
                        fig_risk_trend.update_layout(height=420)
                        st.plotly_chart(fig_risk_trend, use_container_width=True)

                        spike_days = patient_trend[patient_trend["risk_spike_flag"] == True].copy()
                        if spike_days.empty:
                            st.success("No risk spikes detected for this patient.")
                        else:
                            st.error(f"{len(spike_days)} risk spike day(s) detected for this patient.")
                            st.dataframe(
                                spike_days[
                                    [
                                        "date",
                                        "event_count",
                                        "missed_count",
                                        "late_dose_count",
                                        "daily_behavior_score",
                                        "rolling_risk_score",
                                        "risk_delta",
                                        "risk_spike_flag",
                                    ]
                                ],
                                use_container_width=True,
                            )

                    st.markdown("---")
                    st.markdown("#### Patient Alert History")

                    patient_rule_alerts = (
                        rules_alerts[rules_alerts["patient_id"] == selected_patient]
                        if not rules_alerts.empty else pd.DataFrame()
                    )

                    patient_ml_alerts = (
                        ml_alerts[ml_alerts["patient_id"] == selected_patient]
                        if not ml_alerts.empty else pd.DataFrame()
                    )

                    alert_k1, alert_k2, alert_k3 = st.columns(3)
                    with alert_k1:
                        st.metric("Rule Alerts", len(patient_rule_alerts))
                    with alert_k2:
                        high_patient_alerts = int((patient_rule_alerts["severity"] == "High").sum()) if not patient_rule_alerts.empty else 0
                        st.metric("High Severity", high_patient_alerts)
                    with alert_k3:
                        st.metric("ML Anomalies", len(patient_ml_alerts))

                    alert_col1, alert_col2 = st.columns(2)

                    with alert_col1:
                        st.markdown("**Rule-Based Alerts for Selected Patient**")
                        if patient_rule_alerts.empty:
                            st.success("No rule-based alerts for this patient.")
                        else:
                            st.dataframe(patient_rule_alerts, use_container_width=True)

                    with alert_col2:
                        st.markdown("**ML Anomaly Alerts for Selected Patient**")
                        if patient_ml_alerts.empty:
                            st.success("No ML anomaly alerts for this patient.")
                        else:
                            st.dataframe(patient_ml_alerts, use_container_width=True)

                    st.markdown("#### Alert Severity Breakdown")
                    if patient_rule_alerts.empty:
                        st.info("No alert severity data to display for this patient.")
                    else:
                        severity_counts = patient_rule_alerts["severity"].value_counts().reset_index()
                        severity_counts.columns = ["Severity", "Count"]

                        fig_patient_alerts = px.bar(
                            severity_counts,
                            x="Severity",
                            y="Count",
                            title=f"Alert Severity Distribution for {selected_patient}",
                            labels={"Severity": "Severity", "Count": "Alert Count"},
                        )
                        fig_patient_alerts.update_layout(height=350)
                        st.plotly_chart(fig_patient_alerts, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Edit Existing Event")

                    editable_events = patient_events.copy()
                    editable_events["event_label"] = editable_events.apply(
                        lambda row: f"#{row['id']} | {row['drug']} | {row['status']} | {row['event_time']}",
                        axis=1,
                    )

                    selected_event_label = st.selectbox(
                        "Select Event to Edit",
                        editable_events["event_label"].tolist(),
                        key=f"edit_event_{selected_patient}",
                    )

                    selected_event = editable_events[editable_events["event_label"] == selected_event_label].iloc[0]
                    selected_sched = pd.to_datetime(selected_event["scheduled_time"], errors="coerce")
                    default_date = selected_sched.date() if pd.notna(selected_sched) else date.today()
                    default_time = selected_sched.time() if pd.notna(selected_sched) else time(8, 0)

                    with st.form(f"edit_event_form_{selected_patient}"):
                        edited_submitted_by = st.text_input("Edited By", value="")
                        edited_drug = st.text_input("Drug", value=selected_event["drug"])
                        edited_dose = st.number_input("Dose Taken (mg)", min_value=0.0, step=1.0, value=float(selected_event["dosage_mg"]))
                        edited_status = st.selectbox("Dose Status", ["Taken", "Missed"], index=0 if selected_event["status"] == "Taken" else 1)
                        edited_zip = st.number_input("ZIP Code", min_value=0, step=1, value=int(selected_event["zip"]))
                        edited_days_supply = st.number_input("Days Supply", min_value=1, step=1, value=int(selected_event["days_supply"]))
                        edited_refill_count = st.number_input("Refill Count", min_value=0, step=1, value=int(selected_event["refill_count"]))
                        edited_sched_date = st.date_input("Scheduled Dose Date", value=default_date, key=f"edit_sched_date_{selected_patient}")
                        edited_sched_time = st.time_input("Scheduled Dose Time", value=default_time, key=f"edit_sched_time_{selected_patient}")
                        save_edit = st.form_submit_button("Save Changes")

                    if save_edit:
                        edit_validation = validate_event_input(
                            patient_id=str(selected_event["patient_id"]),
                            submitted_by=edited_submitted_by,
                            drug=edited_drug,
                            dose=edited_dose,
                            status=edited_status,
                            zip_code=int(edited_zip),
                            scheduled_date=edited_sched_date,
                            scheduled_clock=edited_sched_time,
                        )

                        for warning in edit_validation["warnings"]:
                            st.warning(warning)

                        if edit_validation["errors"]:
                            for error in edit_validation["errors"]:
                                st.error(error)
                        else:
                            edited_scheduled_dt = edit_validation["scheduled_dt"].strftime("%Y-%m-%d %H:%M:%S")
                            editor_name = edit_validation["cleaned_submitted_by"]
                            cleaned_edited_drug = edit_validation["cleaned_drug"]
                            final_edited_dose = edited_dose if edited_status == "Taken" else 0

                            previous_value = {
                                "patient_id": selected_event["patient_id"],
                                "prescriber_id": selected_event["prescriber_id"],
                                "drug": selected_event["drug"],
                                "dosage_mg": float(selected_event["dosage_mg"]),
                                "days_supply": int(selected_event["days_supply"]),
                                "refill_count": int(selected_event["refill_count"]),
                                "zip": int(selected_event["zip"]),
                                "status": selected_event["status"],
                                "scheduled_time": selected_event["scheduled_time"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(selected_event["scheduled_time"]) else "",
                                "submitted_by": selected_event.get("submitted_by", "unknown") or "unknown",
                            }

                            new_value = {
                                "patient_id": selected_event["patient_id"],
                                "prescriber_id": selected_event["prescriber_id"],
                                "drug": cleaned_edited_drug,
                                "dosage_mg": final_edited_dose,
                                "days_supply": int(edited_days_supply),
                                "refill_count": int(edited_refill_count),
                                "zip": int(edited_zip),
                                "status": edited_status,
                                "scheduled_time": edited_scheduled_dt,
                                "submitted_by": editor_name,
                            }

                            update_event(
                                event_id=int(selected_event["id"]),
                                patient_id=selected_event["patient_id"],
                                prescriber_id=selected_event["prescriber_id"],
                                drug=cleaned_edited_drug,
                                dosage_mg=final_edited_dose,
                                days_supply=int(edited_days_supply),
                                refill_count=int(edited_refill_count),
                                zip_code=int(edited_zip),
                                status=edited_status,
                                scheduled_time=edited_scheduled_dt,
                                submitted_by=editor_name,
                            )

                            log_audit(
                                event_id=int(selected_event["id"]),
                                patient_id=selected_event["patient_id"],
                                action="UPDATE",
                                submitted_by=editor_name,
                                previous_value=previous_value,
                                new_value=new_value,
                            )

                            st.success("Event updated successfully.")
                            st.rerun()

    with tab3:
        st.markdown('<div class="section-title">Alert Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Current rule-based and anomaly-based alerts for analyst review.</div>', unsafe_allow_html=True)

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### Rule-Based Alerts")
            if rules_alerts.empty:
                st.info("No rule-based alerts found.")
            else:
                st.write(rules_alerts.style.apply(color_row, axis=1))

        with a2:
            st.markdown("#### ML Anomaly Alerts")
            if ml_alerts.empty:
                st.info("No ML anomalies found.")
            else:
                st.dataframe(ml_alerts, use_container_width=True)

        st.markdown("#### Active Alert Feed")
        if rules_alerts.empty:
            st.success("No active alerts at this time.")
        else:
            for _, row in rules_alerts.iterrows():
                if row["severity"] == "High":
                    st.error(f"HIGH: {row['rule']} — Patient {row['patient_id']} ({row['drug']})")
                elif row["severity"] == "Medium":
                    st.warning(f"MEDIUM: {row['rule']} — Patient {row['patient_id']} ({row['drug']})")

    with tab4:
        st.markdown('<div class="section-title">Risk Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Risk trends, prescribing patterns, and behavior-based risk breakdown.</div>', unsafe_allow_html=True)
        st.markdown("#### Analytics Filters")

        f1, f2, f3 = st.columns(3)

        with f1:
            selected_risk_levels = st.multiselect(
                "Risk Level",
                options=["Low", "Moderate", "High", "Critical"],
                default=["Low", "Moderate", "High", "Critical"],
            )

        with f2:
            drug_options = sorted(df["drug"].dropna().unique().tolist())
            selected_drugs = st.multiselect("Drug", options=drug_options, default=drug_options)

        with f3:
            only_alert_patients = st.checkbox("Only patients with alerts")

        filtered_df = df.copy()

        if selected_risk_levels:
            filtered_df = filtered_df[filtered_df["risk_level"].isin(selected_risk_levels)]

        if selected_drugs:
            filtered_df = filtered_df[filtered_df["drug"].isin(selected_drugs)]

        if only_alert_patients:
            alert_patient_ids = set()
            if not rules_alerts.empty:
                alert_patient_ids.update(rules_alerts["patient_id"].dropna().astype(str))
            if not ml_alerts.empty:
                alert_patient_ids.update(ml_alerts["patient_id"].dropna().astype(str))
            filtered_df = filtered_df[filtered_df["patient_id"].astype(str).isin(alert_patient_ids)]

        if filtered_df.empty:
            st.warning("No records match the selected filters.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                fig1 = px.histogram(
                    filtered_df,
                    x="dosage_mg",
                    title="Dosage Distribution",
                    labels={"dosage_mg": "Dosage (mg)"},
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                fig2 = px.scatter(
                    filtered_df,
                    x="refill_count",
                    y="dosage_mg",
                    hover_data=["patient_id", "drug", "risk_level"],
                    title="Refill Count vs Dosage",
                    labels={"refill_count": "Refill Count", "dosage_mg": "Dosage (mg)"},
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)

            with c3:
                risk_counts = filtered_df["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                risk_counts["Risk Level"] = pd.Categorical(
                    risk_counts["Risk Level"],
                    categories=["Low", "Moderate", "High", "Critical"],
                    ordered=True,
                )
                risk_counts = risk_counts.sort_values("Risk Level")

                fig_risk = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Count",
                    title="Patients by Risk Level",
                    labels={"Count": "Patient Count"},
                )
                fig_risk.update_layout(height=400)
                st.plotly_chart(fig_risk, use_container_width=True)

            with c4:
                if not events_df.empty:
                    status_counts = events_df["status"].value_counts().reset_index()
                    status_counts.columns = ["Dose Status", "Count"]
                    fig3 = px.bar(
                        status_counts,
                        x="Dose Status",
                        y="Count",
                        title="Taken vs Missed Doses",
                        labels={"Count": "Number of Events"},
                    )
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("No patient event data available for adherence chart.")

        st.markdown("#### Random Forest Risk Probability")
        if "rf_risk_probability" in df.columns:
            fig_rf = px.histogram(
                df,
                x="rf_risk_probability",
                nbins=10,
                title="Predicted High-Risk Probability",
                labels={"rf_risk_probability": "Predicted Probability"},
            )
            fig_rf.update_layout(height=400)
            st.plotly_chart(fig_rf, use_container_width=True)

        if "rf_feature_importance" in df.attrs:
            st.markdown("#### Model Feature Importance")
            fi = df.attrs["rf_feature_importance"]
            fi_sorted = fi.sort_values(by="importance", ascending=True)
            fig_fi = px.bar(
                fi_sorted,
                x="importance",
                y="feature",
                orientation="h",
                title="Top Risk Drivers (Model)",
            )
            fig_fi.update_layout(height=400)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Model feature importance is unavailable. This may occur if there is insufficient data or only one risk class present.")

        st.markdown("#### Overall Risk Driver Breakdown")
        risk_driver_df = parse_risk_components_series(df)

        if risk_driver_df.empty:
            st.info("No elevated risk components available yet.")
        else:
            driver_summary = (
                risk_driver_df
                .groupby("factor", as_index=False)["points"]
                .sum()
                .sort_values("points", ascending=True)
                .tail(8)
            )

            fig_drivers = px.bar(
                driver_summary,
                x="points",
                y="factor",
                orientation="h",
                title="Top Overall Risk Drivers",
                labels={"points": "Total Risk Points", "factor": "Risk Driver"},
            )
            fig_drivers.update_layout(height=420)
            st.plotly_chart(fig_drivers, use_container_width=True)

        st.markdown("#### Risk Model Dataset")
        combined_columns = [
            "patient_id",
            "drug",
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
            "medium_alert_count",
            "high_alert_count",
            "ml_alert_count",
            "anomaly_severity",
            "risk_score",
            "risk_level",
            "risk_reasons",
            "risk_components",
            "rf_risk_probability",
            "rf_prediction",
        ]
        available_combined_columns = [col for col in combined_columns if col in df.columns]
        st.dataframe(df[available_combined_columns], use_container_width=True)

    with tab5:
        st.markdown('<div class="section-title">Exportable Patient Reports</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Generate a one-click patient report with alerts, adherence, and medication timeline.</div>',
            unsafe_allow_html=True,
        )

        if events_df.empty:
            st.info("No patient reports are available because no medication events have been submitted yet.")
        else:
            patient_options = sorted(events_df["patient_id"].replace("", pd.NA).dropna().unique().tolist())

            if not patient_options:
                st.info("No patients available for reports.")
            else:
                report_patient = st.selectbox("Select Patient for Report", patient_options, key="report_patient")

                report_events = events_df[events_df["patient_id"] == report_patient].copy()
                report_profile = df[df["patient_id"] == report_patient].copy()

                patient_rule_alerts = rules_alerts[rules_alerts["patient_id"] == report_patient] if not rules_alerts.empty else pd.DataFrame()
                patient_ml_alerts = ml_alerts[ml_alerts["patient_id"] == report_patient] if not ml_alerts.empty else pd.DataFrame()

                report_events["event_time"] = pd.to_datetime(report_events["event_time"], errors="coerce")
                report_events["scheduled_time"] = pd.to_datetime(report_events["scheduled_time"], errors="coerce")
                report_events = report_events.sort_values("event_time", ascending=False)

                taken_count = int((report_events["status"] == "Taken").sum()) if not report_events.empty else 0
                missed_count_report = int((report_events["status"] == "Missed").sum()) if not report_events.empty else 0
                total_events_report = len(report_events)
                adherence_score = round((taken_count / total_events_report) * 100, 1) if total_events_report > 0 else 0.0

                if not report_profile.empty:
                    summary_row = report_profile.sort_values("risk_score", ascending=False).iloc[0]
                    report_summary = pd.DataFrame([{
                        "patient_id": report_patient,
                        "drug": summary_row.get("drug", ""),
                        "risk_score": summary_row.get("risk_score", ""),
                        "risk_level": summary_row.get("risk_level", ""),
                        "status_badge": get_status_label(summary_row.get("risk_level", "Moderate")),
                        "adherence_score": adherence_score,
                        "taken_events": taken_count,
                        "missed_events": missed_count_report,
                        "total_events": total_events_report,
                        "rule_alert_count": len(patient_rule_alerts),
                        "high_rule_alerts": int((patient_rule_alerts["severity"] == "High").sum()) if not patient_rule_alerts.empty else 0,
                        "medium_rule_alerts": int((patient_rule_alerts["severity"] == "Medium").sum()) if not patient_rule_alerts.empty else 0,
                        "ml_alert_count": len(patient_ml_alerts),
                        "risk_reasons": summary_row.get("risk_reasons", ""),
                        "risk_components": summary_row.get("risk_components", ""),
                        "on_time_pct": summary_row.get("on_time_pct", ""),
                        "late_dose_count": summary_row.get("late_dose_count", ""),
                        "recent_missed_7d": summary_row.get("recent_missed_7d", ""),
                        "recent_missed_3d": summary_row.get("recent_missed_3d", ""),
                        "missed_streak": summary_row.get("missed_streak", ""),
                        "late_streak": summary_row.get("late_streak", ""),
                        "timing_variability_hours": summary_row.get("timing_variability_hours", ""),
                        "anomaly_severity": summary_row.get("anomaly_severity", ""),
                    }])
                else:
                    report_summary = pd.DataFrame([{
                        "patient_id": report_patient,
                        "adherence_score": adherence_score,
                        "taken_events": taken_count,
                        "missed_events": missed_count_report,
                        "total_events": total_events_report,
                        "rule_alert_count": len(patient_rule_alerts),
                        "ml_alert_count": len(patient_ml_alerts),
                    }])

                alert_summary = pd.DataFrame([{
                    "patient_id": report_patient,
                    "total_rule_alerts": len(patient_rule_alerts),
                    "high_rule_alerts": int((patient_rule_alerts["severity"] == "High").sum()) if not patient_rule_alerts.empty else 0,
                    "medium_rule_alerts": int((patient_rule_alerts["severity"] == "Medium").sum()) if not patient_rule_alerts.empty else 0,
                    "total_ml_alerts": len(patient_ml_alerts),
                }])

                timeline_cols = [c for c in [
                    "patient_id",
                    "drug",
                    "status",
                    "dosage_mg",
                    "scheduled_time",
                    "event_time",
                    "taken_on_time",
                    "submitted_by",
                ] if c in report_events.columns]

                timeline_df = report_events[timeline_cols].copy()
                if "scheduled_time" in timeline_df.columns:
                    timeline_df["scheduled_time"] = timeline_df["scheduled_time"].dt.strftime("%Y-%m-%d %H:%M")
                if "event_time" in timeline_df.columns:
                    timeline_df["event_time"] = timeline_df["event_time"].dt.strftime("%Y-%m-%d %H:%M")

                st.markdown("#### Report Preview")
                st.dataframe(report_summary, use_container_width=True)

                preview_col1, preview_col2 = st.columns(2)

                with preview_col1:
                    st.markdown("#### Alert Summary")
                    st.dataframe(alert_summary, use_container_width=True)

                with preview_col2:
                    st.markdown("#### Adherence Summary")
                    adherence_preview = pd.DataFrame([{
                        "patient_id": report_patient,
                        "adherence_score": adherence_score,
                        "taken_events": taken_count,
                        "missed_events": missed_count_report,
                        "total_events": total_events_report,
                    }])
                    st.dataframe(adherence_preview, use_container_width=True)

                st.markdown("#### Medication Event Timeline")
                st.dataframe(timeline_df, use_container_width=True)

                export_report_df = pd.concat([
                    pd.DataFrame([{"REPORT_SECTION": "PATIENT_SUMMARY"}]),
                    report_summary,
                    pd.DataFrame([{}]),
                    pd.DataFrame([{"REPORT_SECTION": "ALERT_SUMMARY"}]),
                    alert_summary,
                    pd.DataFrame([{}]),
                    pd.DataFrame([{"REPORT_SECTION": "ADHERENCE_SUMMARY"}]),
                    adherence_preview,
                    pd.DataFrame([{}]),
                    pd.DataFrame([{"REPORT_SECTION": "MEDICATION_EVENT_TIMELINE"}]),
                    timeline_df,
                ], ignore_index=True)

                st.download_button(
                    label="Download Patient Report CSV",
                    data=export_report_df.to_csv(index=False),
                    file_name=f"patient_report_{report_patient}.csv",
                    mime="text/csv",
                )

    with tab6:
        st.markdown('<div class="section-title">Audit Trail</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Review who submitted data, when it was submitted, and exactly what changed.</div>',
            unsafe_allow_html=True,
        )

        audit_df = load_audit_log()

        if not audit_df.empty and "submitted_by" in audit_df.columns:
            audit_df["submitted_by"] = audit_df["submitted_by"].replace("", pd.NA).fillna("legacy record")

        if audit_df.empty:
            st.info("No audit records available yet.")
        else:
            audit_display = audit_df.copy()
            audit_display["previous_parsed"] = audit_display["previous_value"].apply(safe_parse_json)
            audit_display["new_parsed"] = audit_display["new_value"].apply(safe_parse_json)
            audit_display["changed_fields_list"] = audit_display.apply(
                lambda row: get_changed_fields(row["previous_parsed"], row["new_parsed"]),
                axis=1,
            )
            audit_display["field_count_changed"] = audit_display["changed_fields_list"].apply(len)
            audit_display["changed_fields"] = audit_display["changed_fields_list"].apply(
                lambda x: ", ".join(x) if x else "none"
            )

            summary_cols = [
                "id",
                "event_id",
                "patient_id",
                "action",
                "submitted_by",
                "timestamp",
                "field_count_changed",
                "changed_fields",
            ]
            available_summary_cols = [c for c in summary_cols if c in audit_display.columns]

            st.markdown("#### Audit Summary")
            st.dataframe(audit_display[available_summary_cols], use_container_width=True)

            st.markdown("#### Audit Details")
            for _, row in audit_display.iterrows():
                action = row.get("action", "")
                patient_id = row.get("patient_id", "")
                submitted_by = row.get("submitted_by", "")
                timestamp = row.get("timestamp", "")
                expander_title = f"{action} | Patient {patient_id} | By {submitted_by} | {timestamp}"

                with st.expander(expander_title):
                    prev_dict = row["previous_parsed"]
                    new_dict = row["new_parsed"]
                    changed_keys = row["changed_fields_list"]

                    if action == "CREATE":
                        st.markdown("**Created Record**")
                        st.json(new_dict)
                    else:
                        left, right = st.columns(2)
                        with left:
                            st.markdown("**Previous Value**")
                            st.json(prev_dict)
                        with right:
                            st.markdown("**New Value**")
                            st.json(new_dict)

                    if changed_keys:
                        st.markdown("**Changed Fields Only**")
                        changed_rows = []
                        for key in changed_keys:
                            changed_rows.append({
                                "field": key,
                                "previous": prev_dict.get(key),
                                "new": new_dict.get(key),
                                "changed": "Updated",
                            })

                        changed_df = pd.DataFrame(changed_rows)
                        st.dataframe(changed_df.style.apply(highlight_changes, axis=1), use_container_width=True)
                    else:
                        st.caption("No field-level differences detected.")
