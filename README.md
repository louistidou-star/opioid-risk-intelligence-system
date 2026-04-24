# National Opioid Risk Intelligence Platform (NORIP)

## Overview

The National Opioid Risk Intelligence Platform (NORIP) is a cybersecurity-inspired healthcare analytics system designed to monitor medication adherence, detect behavioral risk patterns, and support analyst-driven intervention workflows.

The platform combines principles from SIEM (Security Information and Event Management), machine learning, and public health monitoring to identify high-risk opioid usage behaviors in near real-time.

---

## Key Features

### 1. Medication Event Tracking

* Timestamped medication intake events
* Scheduled vs. actual intake comparison
* Persistent storage using SQLite
* Full audit trail of all changes (create/update)

### 2. Adherence Monitoring

* Doses taken on time
* Missed dose tracking
* Adherence percentage calculation
* Late dose detection
* Missed-dose and late-dose streak analysis

### 3. Behavioral Risk Detection

* Early redosing detection
* Multiple doses within short time windows
* Nighttime medication usage patterns
* Long gaps between doses
* Timing variability analysis

### 4. Risk Scoring Engine

* Multi-factor rule-based scoring system
* Behavioral + adherence + alert-based risk contributions
* Recency-weighted scoring
* Risk levels: Low, Moderate, High, Critical
* Explainable risk components per patient

### 5. Machine Learning Integration

* Random Forest model for risk prediction
* High-risk probability estimation
* Feature importance visualization
* Hybrid rule-based + ML scoring approach

### 6. Temporal Risk Intelligence

* Risk trend tracking over time
* Daily aggregation of patient risk
* Rolling risk score analysis
* Risk spike detection (sudden increases in risk)

### 7. Alert Detection & Management

* Rule-based alerts (severity: Medium, High)
* Machine learning anomaly alerts
* Alert lifecycle workflow:

  * Open
  * Reviewed
  * Escalated
  * Resolved
  * False Positive
* Analyst notes and assignment tracking

### 8. Analyst Dashboard

* Patient-level monitoring interface
* Risk trend visualization
* Medication timeline view
* Alert history per patient
* Filtering by risk level, drug, and alert presence
* Editable patient event records

### 9. Reporting System

* Exportable patient reports (CSV)
* Includes:

  * Risk summary
  * Adherence metrics
  * Alert summaries
  * Full medication timeline

### 10. Audit Trail & Data Integrity

* Full logging of all data modifications
* Previous vs. new value tracking
* Changed-field highlighting
* User attribution (who made changes and when)

### 11. Role-Based Access Control

* Patient: submit and view personal data
* Analyst: monitor, analyze, and review alerts
* Admin: access audit logs and system controls
* Restricted access to sensitive features

---

## System Architecture

The platform is structured as a modular analytics system:

* **Frontend:** Streamlit-based interface (Patient + Analyst views)
* **Backend:** Python with SQLite database
* **Detection Engine:** Rule-based + ML anomaly detection
* **Data Layer:** Event logging, audit tracking, and persistence
* **Analytics Layer:** Risk scoring, trend analysis, and feature engineering

---

## Technologies Used

* Python 3.11
* Streamlit
* Pandas
* Scikit-learn (Random Forest)
* Plotly
* SQLite

---

## Use Case

This system demonstrates how cybersecurity methodologies (SIEM, anomaly detection, audit logging) can be adapted to healthcare monitoring to:

* Detect risky medication behavior
* Improve adherence visibility
* Support analyst-driven intervention workflows
* Provide explainable and auditable risk intelligence

---

## Limitations (Prototype)

* No real patient authentication system
* Machine learning model trained on synthetic/rule-derived labels
* No external clinical data integration
* No automated notification or intervention system

---

## Future Enhancements

* Intervention and notification system (analyst actions and follow-ups)
* Mobile-friendly patient interface
* Drug-specific clinical rules and thresholds
* Real-world model training with outcome data
* Secure authentication and patient identity mapping

---

## Author

Louis Tidou
Technology Management (Cybersecurity)
University of Bridgeport

---

## License

This project is for educational and research purposes.
