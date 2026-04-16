# National Opioid Risk Intelligence System

A deployed healthcare risk intelligence prototype that applies SIEM-style monitoring, rule-based detection, and machine learning anomaly analysis to opioid usage data. The system supports patient event input, real-time alert generation, risk scoring, and analyst-facing visualization through a web-based dashboard.

## Why it matters

Opioid misuse, irregular dosing behavior, and medication adherence failures remain major public health concerns. This system demonstrates how cybersecurity-inspired monitoring principles can be adapted to healthcare risk analysis by turning medication-related events into structured signals that can be analyzed, scored, and escalated.

## Key features

- Real-time patient input simulation through a phone-friendly sidebar form
- Rule-based alert generation for suspicious or high-risk medication patterns
- Machine learning anomaly detection using Isolation Forest
- Risk scoring based on dosage, refill behavior, and supply patterns
- SOC-style alert panel for analyst review
- Interactive charts and downloadable alert outputs
- Deployed web application accessible through a live Streamlit interface
  
## Notes

This project is a prototype for demonstration and analytical evaluation purposes. It uses sample and simulated event data and is not intended for direct clinical deployment without appropriate security, privacy, compliance, and infrastructure controls.

## Tech stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly

## Project structure

```text
opioid-risk-intelligence-system/
│
├── app.py
├── detection.py
├── requirements.txt
├── .gitignore
└── data/
    └── sample_events.csv
