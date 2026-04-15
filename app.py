import pandas as pd
import plotly.express as px
import streamlit as st

from detection import run_ml_detector, run_rules

st.set_page_config(page_title="Opioid Risk Detection", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("System Controls")
st.sidebar.markdown("SIEM-based Risk Monitoring")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
risk_threshold = st.sidebar.slider("ML Sensitivity", 0.05, 0.5, 0.15)

st.sidebar.subheader("System Status")
st.sidebar.success("SIEM Engine: ACTIVE")


def calculate_risk_score(row):
    score = 0

    if row["dosage_mg"] > 200:
        score += 40
    if row["refill_count"] > 3:
        score += 40
    if row["days_supply"] < 10:
        score += 20

    return min(score, 100)


def color_row(row):
    if row["severity"] == "High":
        return ["background-color: red; color: white"] * len(row)
    elif row["severity"] == "Medium":
        return ["background-color: orange"] * len(row)
    return [""] * len(row)


st.title("National Opioid Risk Intelligence System")
st.caption("Cybersecurity-driven anomaly detection and public health monitoring platform")

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/sample_events.csv")
except Exception as e:
    st.error(f"Error loading data: {e}")
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

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

# Patient input simulation
st.sidebar.subheader("Patient Input (Simulation)")

with st.sidebar.form("patient_form"):
    p_id = st.text_input("Patient ID")
    drug = st.text_input("Drug")
    dose = st.number_input("Dose Taken (mg)", min_value=0.0, step=1.0)
    status = st.selectbox("Status", ["Taken", "Missed"])
    submit = st.form_submit_button("Submit")

if submit:
    new_row = {
        "patient_id": p_id,
        "prescriber_id": "D_SIM",
        "drug": drug,
        "dosage_mg": dose if status == "Taken" else 0,
        "days_supply": 1,
        "refill_count": 0,
        "zip": 10001
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    st.sidebar.success("Entry submitted")

df["risk_score"] = df.apply(calculate_risk_score, axis=1)
st.sidebar.info(f"Records Loaded: {len(df)}")

try:
    rules_alerts = run_rules(df)
    ml_alerts = run_ml_detector(df, contamination=risk_threshold)
    st.sidebar.warning(f"Alerts Generated: {len(rules_alerts)}")
except Exception as e:
    st.error(f"Error running detection: {e}")
    st.stop()

st.subheader("Raw Data")
st.dataframe(df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Rule-Based Alerts")
    if rules_alerts.empty:
        st.info("No rule-based alerts found.")
    else:
        styled = rules_alerts.style.apply(color_row, axis=1)
        st.write(styled)

with col2:
    st.subheader("ML Anomaly Alerts")
    if ml_alerts.empty:
        st.info("No ML anomalies found.")
    else:
        st.dataframe(ml_alerts, use_container_width=True)

st.subheader("🚨 Active Alerts (SOC View)")

for _, row in rules_alerts.iterrows():
    if row["severity"] == "High":
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4a0000, #7a0000);
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 6px solid #ff4b4b;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        ">
            <h3 style="color: #ffffff; margin: 0 0 12px 0; font-size: 26px;">
                🔴 HIGH ALERT
            </h3>
            <p style="color: #ffffff; font-size: 18px; margin: 6px 0;">
                <b>Rule:</b> {row['rule']}
            </p>
            <p style="color: #ffffff; font-size: 18px; margin: 6px 0;">
                <b>Patient:</b> {row['patient_id']}
            </p>
            <p style="color: #ffffff; font-size: 18px; margin: 6px 0;">
                <b>Drug:</b> {row['drug']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif row["severity"] == "Medium":
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #5a3a00, #8a5a00);
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            border-left: 6px solid #ffa500;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        ">
            <h3 style="color: #ffffff; margin: 0 0 12px 0; font-size: 26px;">
                🟠 MEDIUM ALERT
            </h3>
            <p style="color: #ffffff; font-size: 18px; margin: 6px 0;">
                <b>Rule:</b> {row['rule']}
            </p>
            <p style="color: #ffffff; font-size: 18px; margin: 6px 0;">
                <b>Patient:</b> {row['patient_id']}
            </p>
            <p style="color: #ffffff; font-size: 18px; margin: 6px 0;">
                <b>Drug:</b> {row['drug']}
            </p>
        </div>
        """, unsafe_allow_html=True)

st.subheader("Risk Analytics Dashboard")

col3, col4 = st.columns(2)

with col3:
    fig1 = px.histogram(df, x="dosage_mg", title="Dosage Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col4:
    fig2 = px.scatter(
        df,
        x="refill_count",
        y="dosage_mg",
        hover_data=["patient_id", "drug", "prescriber_id"],
        title="Refill Count vs Dosage",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Geographic Risk Clusters")
geo = df["zip"].value_counts().reset_index()
geo.columns = ["zip", "count"]
st.dataframe(geo, use_container_width=True)

st.subheader("Top High-Risk Patients")
top_risk = df.sort_values(by="risk_score", ascending=False).head(5)
st.dataframe(top_risk, use_container_width=True)

st.subheader("Risk Score Distribution")
fig3 = px.histogram(df, x="risk_score", nbins=10, title="Risk Score Distribution")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Risk Summary")

high_risk = 0
medium_risk = 0

if not rules_alerts.empty and "severity" in rules_alerts.columns:
    high_risk = len(rules_alerts[rules_alerts["severity"] == "High"])
    medium_risk = len(rules_alerts[rules_alerts["severity"] == "Medium"])

col5, col6, col7, col8 = st.columns(4)
col5.metric("Total Records", len(df))
col6.metric("Total Alerts", len(rules_alerts))
col7.metric("High Risk", high_risk)
col8.metric("ML Anomalies", len(ml_alerts))

st.subheader("Download Results")

rule_csv = rules_alerts.to_csv(index=False).encode("utf-8")
ml_csv = ml_alerts.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Rule Alerts",
    data=rule_csv,
    file_name="rule_alerts.csv",
    mime="text/csv",
)

st.download_button(
    "Download ML Alerts",
    data=ml_csv,
    file_name="ml_alerts.csv",
    mime="text/csv",
)

st.subheader("Alert Definitions")
st.markdown("""
- **Excessive Refills**: More than 3 refills detected
- **High Dosage**: Dosage exceeds 200 mg threshold
- **Short Supply Abuse Pattern**: Low days supply with high refill count
- **ML Anomaly**: Statistical deviation from normal prescribing patterns
""")