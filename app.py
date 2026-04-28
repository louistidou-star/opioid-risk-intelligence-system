from datetime import date, datetime, time
import json
import hashlib
import sqlite3
import os

import joblib

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


# ------------------------------------------------------------
# Public demo configuration
# ------------------------------------------------------------
# Access controls have been removed for the public research prototype.
# The app uses synthetic/demo data and all modes are available for demonstration.

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



def build_dose_timing_flags(events_df: pd.DataFrame) -> pd.DataFrame:
    """Create event-level flags for suspicious dose timing patterns.

    Flags are based on patient + drug history:
    - early_redose_flag: a taken dose less than 4 hours after the prior taken dose
    - frequent_6h_flag: 3 or more taken doses within a rolling 6-hour window
    - frequent_24h_flag: 4 or more taken doses within a rolling 24-hour window
    - nighttime_use_flag: taken dose between 10 PM and 5 AM
    - long_gap_flag: more than 36 hours since the prior taken dose
    """
    output_columns = [
        "id",
        "patient_id",
        "drug",
        "event_time",
        "status",
        "hours_since_last_taken",
        "taken_count_6h",
        "taken_count_24h",
        "early_redose_flag",
        "frequent_6h_flag",
        "frequent_24h_flag",
        "nighttime_use_flag",
        "long_gap_flag",
        "dose_timing_alert",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=output_columns)

    ev = events_df.copy()
    if "id" not in ev.columns:
        ev["id"] = range(1, len(ev) + 1)

    ev["patient_id"] = ev["patient_id"].astype(str).str.strip()
    ev["drug"] = ev["drug"].astype(str).str.strip()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev["status"] = ev["status"].astype(str)
    ev = ev.dropna(subset=["event_time"])

    if ev.empty:
        return pd.DataFrame(columns=output_columns)

    ev = ev.sort_values(["patient_id", "drug", "event_time"])
    ev["hours_since_last_taken"] = pd.NA
    ev["taken_count_6h"] = 0
    ev["taken_count_24h"] = 0
    ev["early_redose_flag"] = False
    ev["frequent_6h_flag"] = False
    ev["frequent_24h_flag"] = False
    ev["nighttime_use_flag"] = False
    ev["long_gap_flag"] = False

    for (_, _), group in ev.groupby(["patient_id", "drug"], dropna=True):
        taken = group[group["status"] == "Taken"].copy()
        if taken.empty:
            continue

        taken_times = taken["event_time"].tolist()
        taken_indices = taken.index.tolist()

        for position, idx in enumerate(taken_indices):
            current_time = taken_times[position]
            hour = current_time.hour

            if position > 0:
                previous_time = taken_times[position - 1]
                hours_since = (current_time - previous_time).total_seconds() / 3600.0
                ev.at[idx, "hours_since_last_taken"] = round(hours_since, 2)
                ev.at[idx, "early_redose_flag"] = hours_since < 4
                ev.at[idx, "long_gap_flag"] = hours_since > 36

            count_6h = sum(
                0 <= (current_time - prior_time).total_seconds() / 3600.0 <= 6
                for prior_time in taken_times[: position + 1]
            )
            count_24h = sum(
                0 <= (current_time - prior_time).total_seconds() / 3600.0 <= 24
                for prior_time in taken_times[: position + 1]
            )

            ev.at[idx, "taken_count_6h"] = count_6h
            ev.at[idx, "taken_count_24h"] = count_24h
            ev.at[idx, "frequent_6h_flag"] = count_6h >= 3
            ev.at[idx, "frequent_24h_flag"] = count_24h >= 4
            ev.at[idx, "nighttime_use_flag"] = hour >= 22 or hour < 5

    flag_cols = [
        "early_redose_flag",
        "frequent_6h_flag",
        "frequent_24h_flag",
        "nighttime_use_flag",
        "long_gap_flag",
    ]
    ev["dose_timing_alert"] = ev[flag_cols].any(axis=1)
    ev["hours_since_last_taken"] = pd.to_numeric(ev["hours_since_last_taken"], errors="coerce")

    return ev[output_columns]

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
        "avg_hours_between_taken",
        "early_redose_count",
        "frequent_dose_6h_count",
        "frequent_dose_24h_count",
        "nighttime_use_count",
        "long_gap_count",
        "dose_timing_alert_count",
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

    timing_flags = build_dose_timing_flags(ev)
    if not timing_flags.empty:
        timing_flags["hours_since_last_taken"] = pd.to_numeric(
            timing_flags["hours_since_last_taken"], errors="coerce"
        )
        timing_summary = timing_flags.groupby("patient_id", dropna=True).agg(
            avg_hours_between_taken=("hours_since_last_taken", "mean"),
            early_redose_count=("early_redose_flag", "sum"),
            frequent_dose_6h_count=("frequent_6h_flag", "sum"),
            frequent_dose_24h_count=("frequent_24h_flag", "sum"),
            nighttime_use_count=("nighttime_use_flag", "sum"),
            long_gap_count=("long_gap_flag", "sum"),
            dose_timing_alert_count=("dose_timing_alert", "sum"),
        ).reset_index()
    else:
        timing_summary = pd.DataFrame(columns=[
            "patient_id",
            "avg_hours_between_taken",
            "early_redose_count",
            "frequent_dose_6h_count",
            "frequent_dose_24h_count",
            "nighttime_use_count",
            "long_gap_count",
            "dose_timing_alert_count",
        ])

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
    summary = summary.merge(timing_summary, on="patient_id", how="left")
    summary["missed_streak"] = summary["missed_streak"].fillna(0).astype(int)
    summary["late_streak"] = summary["late_streak"].fillna(0).astype(int)

    timing_defaults = {
        "avg_hours_between_taken": 0,
        "early_redose_count": 0,
        "frequent_dose_6h_count": 0,
        "frequent_dose_24h_count": 0,
        "nighttime_use_count": 0,
        "long_gap_count": 0,
        "dose_timing_alert_count": 0,
    }
    for col, default in timing_defaults.items():
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(default)

    summary["avg_hours_between_taken"] = summary["avg_hours_between_taken"].round(2)
    for col in [c for c in timing_defaults if c != "avg_hours_between_taken"]:
        summary[col] = summary[col].astype(int)

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
        "avg_hours_between_taken": 0,
        "early_redose_count": 0,
        "frequent_dose_6h_count": 0,
        "frequent_dose_24h_count": 0,
        "nighttime_use_count": 0,
        "long_gap_count": 0,
        "dose_timing_alert_count": 0,
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

        if row["early_redose_count"] >= 1:
            add_points("dose_frequency", 20, "early redosing detected")
        if row["early_redose_count"] >= 2:
            add_points("dose_frequency", 10, "repeated early redosing")
        if row["frequent_dose_6h_count"] >= 1:
            add_points("dose_frequency", 20, "multiple doses within 6 hours")
        if row["frequent_dose_24h_count"] >= 1:
            add_points("dose_frequency", 15, "high dose frequency within 24 hours")
        if row["nighttime_use_count"] >= 2:
            add_points("dose_timing", 10, "repeated nighttime medication use")
        if row["long_gap_count"] >= 1:
            add_points("dose_timing", 10, "long gap between reported doses")
        if row["dose_timing_alert_count"] >= 3:
            add_points("dose_frequency", 10, "persistent suspicious dose timing pattern")

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
        if row["early_redose_count"] >= 1 and row["anomaly_severity"] >= 0.5:
            add_points("interaction", 10, "early redosing + anomaly combination")
        if row["frequent_dose_6h_count"] >= 1 and row["recent_missed_7d"] >= 1:
            add_points("interaction", 10, "irregular missed-dose and redosing pattern")

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


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

MODEL_FEATURE_COLUMNS = [
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


def load_trained_risk_model():
    """Load the saved model and feature list if they exist."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_COLUMNS_PATH):
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
            feature_columns = json.load(file)
        return model, feature_columns
    except Exception:
        return None, None


def apply_saved_risk_model(model_df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    """Use the trained model from /models instead of retraining during every app refresh."""
    output_df = model_df.copy()

    for col in feature_columns:
        if col not in output_df.columns:
            output_df[col] = 0

    output_df[feature_columns] = output_df[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    probabilities = model.predict_proba(output_df[feature_columns])[:, 1]
    output_df["rf_risk_probability"] = probabilities.round(3)
    output_df["rf_prediction"] = output_df["rf_risk_probability"].apply(
        lambda p: "High Risk Likely" if p >= 0.5 else "Lower Risk Likely"
    )
    output_df["model_source"] = "Saved trained model"

    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }).sort_values(by="importance", ascending=False)
        output_df.attrs["rf_feature_importance"] = feature_importance

    return output_df


def run_random_forest_risk_model(df: pd.DataFrame) -> pd.DataFrame:
    """Predict high-risk probability.

    Preferred path:
    1. Load a saved Random Forest model from models/risk_model.pkl.
    2. Use models/feature_columns.json for consistent feature ordering.

    Fallback path:
    If no saved model exists yet, train a temporary in-app model from the
    current rule-derived risk labels. This preserves demo functionality, but
    train_model.py should be used for the cleaner production-style workflow.
    """
    model_df = df.copy()

    saved_model, saved_feature_columns = load_trained_risk_model()
    if saved_model is not None and saved_feature_columns:
        return apply_saved_risk_model(model_df, saved_model, saved_feature_columns)

    available_features = [col for col in MODEL_FEATURE_COLUMNS if col in model_df.columns]
    if not available_features:
        model_df["rf_risk_probability"] = 0.0
        model_df["rf_prediction"] = "Unavailable"
        model_df["model_source"] = "No model features available"
        return model_df

    model_df[available_features] = model_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    model_df["rf_target"] = model_df["risk_level"].isin(["High", "Critical"]).astype(int)

    if model_df["rf_target"].nunique() < 2:
        model_df["rf_risk_probability"] = 0.0
        model_df["rf_prediction"] = "Unavailable"
        model_df["model_source"] = "Fallback model unavailable: only one class present"
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
    model_df["model_source"] = "Fallback in-app model"

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
# Alert review workflow helpers
# ------------------------------------------------------------
ALERT_REVIEW_DB = "opioid_risk.db"
ALERT_STATUSES = ["Open", "Reviewed", "Escalated", "Resolved", "False Positive"]
ALERT_STATUS_ORDER = {status: idx for idx, status in enumerate(ALERT_STATUSES)}


def get_alert_review_connection():
    return sqlite3.connect(ALERT_REVIEW_DB)


def init_alert_review_db():
    """Create a persistent alert review table for analyst workflow state."""
    with get_alert_review_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_reviews (
                alert_key TEXT PRIMARY KEY,
                alert_type TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                drug TEXT,
                severity TEXT,
                alert_name TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                assigned_to TEXT,
                reviewer TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def make_alert_key(*parts) -> str:
    raw = "|".join([str(part) for part in parts])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18]


def load_alert_reviews() -> pd.DataFrame:
    init_alert_review_db()
    with get_alert_review_connection() as conn:
        return pd.read_sql_query("SELECT * FROM alert_reviews", conn)


def upsert_alert_review(
    alert_key: str,
    alert_type: str,
    patient_id: str,
    drug: str,
    severity: str,
    alert_name: str,
    status: str,
    assigned_to: str,
    reviewer: str,
    notes: str,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_alert_review_connection() as conn:
        existing = conn.execute(
            "SELECT created_at FROM alert_reviews WHERE alert_key = ?",
            (alert_key,),
        ).fetchone()
        created_at = existing[0] if existing else now
        conn.execute(
            """
            INSERT INTO alert_reviews (
                alert_key, alert_type, patient_id, drug, severity, alert_name,
                status, assigned_to, reviewer, notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_key) DO UPDATE SET
                alert_type = excluded.alert_type,
                patient_id = excluded.patient_id,
                drug = excluded.drug,
                severity = excluded.severity,
                alert_name = excluded.alert_name,
                status = excluded.status,
                assigned_to = excluded.assigned_to,
                reviewer = excluded.reviewer,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                alert_key,
                alert_type,
                patient_id,
                drug,
                severity,
                alert_name,
                status,
                assigned_to,
                reviewer,
                notes,
                created_at,
                now,
            ),
        )
        conn.commit()


def build_alert_review_feed(
    rules_alerts: pd.DataFrame,
    ml_alerts: pd.DataFrame,
    timing_flags_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine rule alerts, ML anomalies, and dose-timing flags into one review queue."""
    alert_rows = []

    if not rules_alerts.empty:
        for _, row in rules_alerts.iterrows():
            patient_id = str(row.get("patient_id", "")).strip()
            drug = str(row.get("drug", "")).strip()
            rule = str(row.get("rule", "Rule-Based Alert")).strip()
            severity = str(row.get("severity", "Medium")).strip()
            alert_rows.append({
                "alert_key": make_alert_key("RULE", patient_id, drug, rule, severity),
                "alert_type": "Rule-Based",
                "patient_id": patient_id,
                "drug": drug,
                "severity": severity,
                "alert_name": rule,
                "alert_detail": f"{severity} rule alert: {rule}",
                "event_time": "",
            })

    if not ml_alerts.empty:
        for _, row in ml_alerts.iterrows():
            patient_id = str(row.get("patient_id", "")).strip()
            drug = str(row.get("drug", "")).strip()
            anomaly_score = row.get("anomaly_score", "")
            alert_rows.append({
                "alert_key": make_alert_key("ML", patient_id, drug, anomaly_score),
                "alert_type": "ML Anomaly",
                "patient_id": patient_id,
                "drug": drug,
                "severity": "Medium",
                "alert_name": "ML anomaly detected",
                "alert_detail": f"Anomalous pattern detected; anomaly score: {anomaly_score}",
                "event_time": "",
            })

    if not timing_flags_df.empty:
        timing_alerts = timing_flags_df[timing_flags_df["dose_timing_alert"] == True].copy()
        for _, row in timing_alerts.iterrows():
            patient_id = str(row.get("patient_id", "")).strip()
            drug = str(row.get("drug", "")).strip()
            event_id = row.get("id", "")
            flags = []
            if bool(row.get("early_redose_flag", False)):
                flags.append("early redosing")
            if bool(row.get("frequent_6h_flag", False)):
                flags.append("3+ doses within 6 hours")
            if bool(row.get("frequent_24h_flag", False)):
                flags.append("4+ doses within 24 hours")
            if bool(row.get("nighttime_use_flag", False)):
                flags.append("nighttime use")
            if bool(row.get("long_gap_flag", False)):
                flags.append("long gap")
            alert_name = ", ".join(flags) if flags else "Suspicious dose timing"
            severity = "High" if any(flag in alert_name for flag in ["early redosing", "3+ doses", "4+ doses"]) else "Medium"
            alert_rows.append({
                "alert_key": make_alert_key("TIMING", event_id, patient_id, drug, alert_name),
                "alert_type": "Dose Timing",
                "patient_id": patient_id,
                "drug": drug,
                "severity": severity,
                "alert_name": alert_name,
                "alert_detail": f"Timing alert: {alert_name}",
                "event_time": row.get("event_time", ""),
            })

    queue = pd.DataFrame(alert_rows)
    if queue.empty:
        return pd.DataFrame(columns=[
            "alert_key", "alert_type", "patient_id", "drug", "severity", "alert_name",
            "alert_detail", "event_time", "status", "assigned_to", "reviewer", "notes", "updated_at"
        ])

    if reviews_df.empty:
        queue["status"] = "Open"
        queue["assigned_to"] = ""
        queue["reviewer"] = ""
        queue["notes"] = ""
        queue["updated_at"] = ""
    else:
        review_cols = ["alert_key", "status", "assigned_to", "reviewer", "notes", "updated_at"]
        available_review_cols = [col for col in review_cols if col in reviews_df.columns]
        queue = queue.merge(reviews_df[available_review_cols], on="alert_key", how="left")
        queue["status"] = queue["status"].fillna("Open")
        for col in ["assigned_to", "reviewer", "notes", "updated_at"]:
            if col not in queue.columns:
                queue[col] = ""
            queue[col] = queue[col].fillna("")

    queue["status_rank"] = queue["status"].map(ALERT_STATUS_ORDER).fillna(0).astype(int)
    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    queue["severity_rank"] = queue["severity"].map(severity_rank).fillna(3).astype(int)
    queue = queue.sort_values(["status_rank", "severity_rank", "patient_id", "drug"])

    return queue

# ------------------------------------------------------------
# Data loading and detection pipeline
# ------------------------------------------------------------
init_db()
init_alert_review_db()

st.sidebar.title("System Controls")
st.sidebar.markdown("SIEM-based Risk Monitoring")

# Public prototype mode: no login, no role gate.
current_role = "Public Demo"
active_patient_id = ""

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

timing_flags_df = build_dose_timing_flags(events_df)
alert_reviews_df = load_alert_reviews()
alert_review_feed_df = build_alert_review_feed(rules_alerts, ml_alerts, timing_flags_df, alert_reviews_df)
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
if "model_source" in df.columns:
    st.sidebar.caption(f"ML Model: {df['model_source'].iloc[0]}")


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
# Patient View - Phone-Friendly Patient Page
# ------------------------------------------------------------
elif mode == "Patient View":

    st.markdown(
        """
        <style>
        .mobile-shell { max-width: 620px; margin: 0 auto; }
        .mobile-card {
            background: #ffffff;
            border: 1px solid #e8ecf2;
            border-radius: 22px;
            padding: 20px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            margin-bottom: 18px;
        }
        .mobile-title { font-size: 1.35rem; font-weight: 800; color: #111827; margin-bottom: 6px; }
        .mobile-subtitle { font-size: 0.92rem; color: #6b7280; margin-bottom: 12px; }
        .quick-status {
            font-size: 0.9rem;
            font-weight: 700;
            padding: 8px 12px;
            border-radius: 999px;
            display: inline-block;
            background: #eef2ff;
            color: #3730a3;
            border: 1px solid #c7d2fe;
        }
        div[data-testid="stForm"] button {
            min-height: 3.2rem;
            border-radius: 14px;
            font-weight: 800;
            font-size: 1.05rem;
        }
        @media (max-width: 700px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            .mobile-card { padding: 16px; border-radius: 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mobile-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mobile-card">
            <div class="mobile-title">Patient Medication Check-In</div>
            <div class="mobile-subtitle">
                Log your dose quickly. The system automatically records the submission time for monitoring and audit purposes.
            </div>
            <span class="quick-status">Phone-friendly patient mode</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_patient_id = active_patient_id if current_role == "Patient" else ""

    with st.form("mobile_patient_form", clear_on_submit=False):
        st.markdown("### Today's Dose")

        p_id = st.text_input(
            "Patient ID",
            value=default_patient_id,
            disabled=True if current_role == "Patient" else False,
            help="Patient role is locked to the authenticated Patient ID.",
        )

        submitted_by_default = active_patient_id if current_role == "Patient" else ""
        submitted_by = st.text_input("Submitted By", value=submitted_by_default)

        drug = st.text_input("Medication", placeholder="Example: Oxycodone")
        dose = st.number_input("Dose Taken (mg)", min_value=0.0, step=1.0, value=0.0)
        zip_code = st.number_input("ZIP Code", min_value=0, step=1)

        st.markdown("#### Scheduled Dose Time")
        scheduled_date = st.date_input("Scheduled Dose Date", value=date.today())
        scheduled_clock = st.time_input("Scheduled Dose Time", value=time(8, 0))

        st.markdown("#### How are you feeling?")
        pain_level = st.slider("Pain Level", 0, 10, 0, help="0 = no pain, 10 = worst pain")
        side_effects = st.text_area(
            "Side Effects / Notes",
            placeholder="Optional: nausea, dizziness, unusual symptoms, or context for this dose",
            height=90,
        )

        st.markdown("#### Quick Log")
        taken_col, missed_col = st.columns(2)
        with taken_col:
            taken_submit = st.form_submit_button("Taken")
        with missed_col:
            missed_submit = st.form_submit_button("Missed")

    submitted_status = None
    if taken_submit:
        submitted_status = "Taken"
    elif missed_submit:
        submitted_status = "Missed"

    if submitted_status:
        validation = validate_event_input(
            patient_id=p_id,
            submitted_by=submitted_by,
            drug=drug,
            dose=dose,
            status=submitted_status,
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
            final_dose = dose if submitted_status == "Taken" else 0

            event_payload = {
                "drug": cleaned_drug,
                "dosage_mg": final_dose,
                "days_supply": 1,
                "refill_count": 0,
                "zip_code": int(zip_code),
                "status": submitted_status,
                "scheduled_time": scheduled_dt,
                "pain_level": int(pain_level),
                "side_effects_notes": side_effects.strip(),
                "logged_from": "phone_friendly_patient_page",
            }

            event_id = insert_event(
                patient_id=cleaned_patient_id,
                prescriber_id="D_SIM",
                drug=cleaned_drug,
                dosage_mg=final_dose,
                days_supply=1,
                refill_count=0,
                zip_code=int(zip_code),
                status=submitted_status,
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

            st.success(f"Dose logged as {submitted_status}. Your submission time was recorded automatically.")
            st.rerun()

    st.markdown("---")
    st.markdown("### My Medication Summary")

    patient_id_input = default_patient_id if current_role == "Patient" else str(p_id).strip()

    if events_df.empty:
        st.info("No medication history has been recorded yet.")
    elif not patient_id_input:
        st.info("Enter a Patient ID above to view medication history.")
    else:
        patient_history = events_df[events_df["patient_id"] == patient_id_input].copy()

        if patient_history.empty:
            st.info("No records found for this patient yet.")
        else:
            patient_history["event_time"] = pd.to_datetime(patient_history["event_time"], errors="coerce")
            patient_history["scheduled_time"] = pd.to_datetime(patient_history["scheduled_time"], errors="coerce")
            patient_history = patient_history.sort_values("event_time", ascending=False)

            taken = patient_history[patient_history["status"] == "Taken"]
            missed = patient_history[patient_history["status"] == "Missed"]
            adherence = round((len(taken) / len(patient_history)) * 100, 1) if len(patient_history) else 0.0
            on_time = int(patient_history["taken_on_time"].fillna(0).sum()) if "taken_on_time" in patient_history.columns else 0

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

            s1, s2 = st.columns(2)
            s1.metric("Adherence", f"{adherence}%")
            s2.metric("Current Status", patient_risk_label)

            s3, s4 = st.columns(2)
            s3.metric("Missed Doses", len(missed))
            s4.metric("Taken On Time", on_time)

            st.caption(f"Last reported intake: {patient_last_intake}")

            st.markdown("#### Recent Medication History")
            display_cols = [c for c in [
                "drug",
                "status",
                "dosage_mg",
                "scheduled_time",
                "event_time",
                "taken_on_time",
                "submitted_by",
            ] if c in patient_history.columns]
            st.dataframe(patient_history[display_cols].head(10), use_container_width=True)

            st.markdown("#### Timeline")
            fig_patient_timeline = px.scatter(
                patient_history.sort_values("event_time"),
                x="event_time",
                y="drug",
                color="status",
                hover_data=[c for c in ["dosage_mg", "scheduled_time", "taken_on_time"] if c in patient_history.columns],
                title=f"Medication Timeline for {patient_id_input}",
                labels={"event_time": "Event Time", "drug": "Medication"},
            )
            fig_patient_timeline.update_layout(height=390, margin=dict(t=55, b=30, l=20, r=20))
            st.plotly_chart(fig_patient_timeline, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

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

        st.markdown("#### Suspicious Dose Timing Watchlist")
        if timing_flags_df.empty:
            st.info("No suspicious dose timing data available yet.")
        else:
            timing_watchlist = timing_flags_df[timing_flags_df["dose_timing_alert"] == True].sort_values(
                "event_time", ascending=False
            )
            if timing_watchlist.empty:
                st.success("No suspicious dose timing patterns detected.")
            else:
                st.dataframe(
                    timing_watchlist[
                        [
                            "patient_id",
                            "drug",
                            "event_time",
                            "hours_since_last_taken",
                            "taken_count_6h",
                            "taken_count_24h",
                            "early_redose_flag",
                            "frequent_6h_flag",
                            "frequent_24h_flag",
                            "nighttime_use_flag",
                            "long_gap_flag",
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
                    st.markdown("#### Dose Frequency & Timing Alerts")
                    patient_timing_flags = timing_flags_df[
                        timing_flags_df["patient_id"] == selected_patient
                    ].copy() if not timing_flags_df.empty else pd.DataFrame()

                    if patient_timing_flags.empty:
                        st.info("No dose timing records available for this patient yet.")
                    else:
                        timing_alerts_only = patient_timing_flags[patient_timing_flags["dose_timing_alert"] == True]
                        tf1, tf2, tf3, tf4 = st.columns(4)
                        tf1.metric("Early Redoses", int(patient_timing_flags["early_redose_flag"].sum()))
                        tf2.metric("3+ Doses / 6h", int(patient_timing_flags["frequent_6h_flag"].sum()))
                        tf3.metric("4+ Doses / 24h", int(patient_timing_flags["frequent_24h_flag"].sum()))
                        tf4.metric("Nighttime Uses", int(patient_timing_flags["nighttime_use_flag"].sum()))

                        if timing_alerts_only.empty:
                            st.success("No suspicious dose frequency patterns detected for this patient.")
                        else:
                            st.warning(f"{len(timing_alerts_only)} suspicious dose timing event(s) detected.")
                            st.dataframe(
                                timing_alerts_only[
                                    [
                                        "event_time",
                                        "drug",
                                        "hours_since_last_taken",
                                        "taken_count_6h",
                                        "taken_count_24h",
                                        "early_redose_flag",
                                        "frequent_6h_flag",
                                        "frequent_24h_flag",
                                        "nighttime_use_flag",
                                        "long_gap_flag",
                                    ]
                                ],
                                use_container_width=True,
                            )

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
        st.markdown('<div class="section-title">Alert Review Workflow</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Review, assign, escalate, resolve, or mark alerts as false positives.</div>',
            unsafe_allow_html=True,
        )

        if alert_review_feed_df.empty:
            st.success("No active alerts are available for review right now.")
        else:
            open_count = int((alert_review_feed_df["status"] == "Open").sum())
            reviewed_count = int((alert_review_feed_df["status"] == "Reviewed").sum())
            escalated_count = int((alert_review_feed_df["status"] == "Escalated").sum())
            closed_count = int((alert_review_feed_df["status"].isin(["Resolved", "False Positive"])).sum())

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Open", open_count)
            q2.metric("Reviewed", reviewed_count)
            q3.metric("Escalated", escalated_count)
            q4.metric("Closed", closed_count)

            st.markdown("#### Alert Queue Filters")
            f1, f2, f3 = st.columns(3)
            with f1:
                selected_statuses = st.multiselect(
                    "Status",
                    options=ALERT_STATUSES,
                    default=["Open", "Reviewed", "Escalated"],
                )
            with f2:
                selected_alert_types = st.multiselect(
                    "Alert Type",
                    options=sorted(alert_review_feed_df["alert_type"].dropna().unique().tolist()),
                    default=sorted(alert_review_feed_df["alert_type"].dropna().unique().tolist()),
                )
            with f3:
                selected_severities = st.multiselect(
                    "Severity",
                    options=sorted(alert_review_feed_df["severity"].dropna().unique().tolist()),
                    default=sorted(alert_review_feed_df["severity"].dropna().unique().tolist()),
                )

            filtered_alert_queue = alert_review_feed_df.copy()
            if selected_statuses:
                filtered_alert_queue = filtered_alert_queue[filtered_alert_queue["status"].isin(selected_statuses)]
            if selected_alert_types:
                filtered_alert_queue = filtered_alert_queue[filtered_alert_queue["alert_type"].isin(selected_alert_types)]
            if selected_severities:
                filtered_alert_queue = filtered_alert_queue[filtered_alert_queue["severity"].isin(selected_severities)]

            display_cols = [
                "alert_type",
                "severity",
                "status",
                "patient_id",
                "drug",
                "alert_name",
                "assigned_to",
                "reviewer",
                "updated_at",
                "notes",
            ]
            st.markdown("#### Unified Alert Queue")
            if filtered_alert_queue.empty:
                st.info("No alerts match the selected filters.")
            else:
                st.dataframe(filtered_alert_queue[display_cols], use_container_width=True)

            st.markdown("#### Review Selected Alert")
            review_options_df = filtered_alert_queue.copy() if not filtered_alert_queue.empty else alert_review_feed_df.copy()
            review_options_df["alert_label"] = review_options_df.apply(
                lambda row: f"{row['status']} | {row['severity']} | {row['alert_type']} | Patient {row['patient_id']} | {row['alert_name']}",
                axis=1,
            )

            selected_alert_label = st.selectbox(
                "Select Alert",
                review_options_df["alert_label"].tolist(),
                key="selected_alert_for_review",
            )
            selected_alert = review_options_df[review_options_df["alert_label"] == selected_alert_label].iloc[0]

            d1, d2 = st.columns([1.2, 2])
            with d1:
                st.markdown("**Alert Details**")
                st.write(f"**Type:** {selected_alert['alert_type']}")
                st.write(f"**Severity:** {selected_alert['severity']}")
                st.write(f"**Patient:** {selected_alert['patient_id']}")
                st.write(f"**Drug:** {selected_alert['drug']}")
                st.write(f"**Status:** {selected_alert['status']}")
            with d2:
                st.markdown("**Description**")
                st.write(selected_alert["alert_detail"])
                if selected_alert.get("notes", ""):
                    st.markdown("**Existing Notes**")
                    st.write(selected_alert.get("notes", ""))

            current_status = selected_alert.get("status", "Open")
            current_status_index = ALERT_STATUSES.index(current_status) if current_status in ALERT_STATUSES else 0

            with st.form("alert_review_form"):
                new_status = st.selectbox(
                    "Update Status",
                    ALERT_STATUSES,
                    index=current_status_index,
                )
                assigned_to = st.text_input("Assigned To", value=selected_alert.get("assigned_to", ""))
                reviewer = st.text_input("Reviewed By", value=selected_alert.get("reviewer", ""))
                notes = st.text_area("Review Notes", value=selected_alert.get("notes", ""), height=120)
                save_review = st.form_submit_button("Save Alert Review")

            if save_review:
                upsert_alert_review(
                    alert_key=selected_alert["alert_key"],
                    alert_type=selected_alert["alert_type"],
                    patient_id=selected_alert["patient_id"],
                    drug=selected_alert["drug"],
                    severity=selected_alert["severity"],
                    alert_name=selected_alert["alert_name"],
                    status=new_status,
                    assigned_to=assigned_to.strip(),
                    reviewer=reviewer.strip(),
                    notes=notes.strip(),
                )
                st.success("Alert review saved successfully.")
                st.rerun()

            st.markdown("#### Status Breakdown")
            status_counts = alert_review_feed_df["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status_workflow = px.bar(
                status_counts,
                x="Status",
                y="Count",
                title="Alert Review Status Breakdown",
                labels={"Count": "Number of Alerts"},
            )
            fig_status_workflow.update_layout(height=360)
            st.plotly_chart(fig_status_workflow, use_container_width=True)

            st.markdown("#### Original Detection Outputs")
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("##### Rule-Based Alerts")
                if rules_alerts.empty:
                    st.info("No rule-based alerts found.")
                else:
                    st.write(rules_alerts.style.apply(color_row, axis=1))

            with a2:
                st.markdown("##### ML Anomaly Alerts")
                if ml_alerts.empty:
                    st.info("No ML anomalies found.")
                else:
                    st.dataframe(ml_alerts, use_container_width=True)

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
                        "early_redose_count": summary_row.get("early_redose_count", ""),
                        "frequent_dose_6h_count": summary_row.get("frequent_dose_6h_count", ""),
                        "frequent_dose_24h_count": summary_row.get("frequent_dose_24h_count", ""),
                        "nighttime_use_count": summary_row.get("nighttime_use_count", ""),
                        "long_gap_count": summary_row.get("long_gap_count", ""),
                        "dose_timing_alert_count": summary_row.get("dose_timing_alert_count", ""),
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

                report_timing_flags = timing_flags_df[
                    timing_flags_df["patient_id"] == report_patient
                ].copy() if not timing_flags_df.empty else pd.DataFrame()

                timing_alert_summary = pd.DataFrame([{
                    "patient_id": report_patient,
                    "early_redose_events": int(report_timing_flags["early_redose_flag"].sum()) if not report_timing_flags.empty else 0,
                    "frequent_6h_events": int(report_timing_flags["frequent_6h_flag"].sum()) if not report_timing_flags.empty else 0,
                    "frequent_24h_events": int(report_timing_flags["frequent_24h_flag"].sum()) if not report_timing_flags.empty else 0,
                    "nighttime_use_events": int(report_timing_flags["nighttime_use_flag"].sum()) if not report_timing_flags.empty else 0,
                    "long_gap_events": int(report_timing_flags["long_gap_flag"].sum()) if not report_timing_flags.empty else 0,
                    "total_dose_timing_alerts": int(report_timing_flags["dose_timing_alert"].sum()) if not report_timing_flags.empty else 0,
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

                st.markdown("#### Dose Timing Alert Summary")
                st.dataframe(timing_alert_summary, use_container_width=True)

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
                    pd.DataFrame([{"REPORT_SECTION": "DOSE_TIMING_ALERT_SUMMARY"}]),
                    timing_alert_summary,
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

