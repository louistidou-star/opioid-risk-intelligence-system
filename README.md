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
=======
The National Opioid Risk Intelligence Platform (NORIP) is a research-driven prototype designed to detect, analyze, and visualize high-risk opioid usage patterns using a cybersecurity-inspired approach.

The system applies methodologies traditionally used in Security Information and Event Management (SIEM) systems, including anomaly detection, behavioral analytics, and audit logging, to the domain of public health monitoring. The objective is to demonstrate how continuous behavioral data can be transformed into actionable risk intelligence for early detection of medication misuse.

---

## Problem Context

The opioid crisis remains a critical public health issue in the United States. Existing monitoring systems are largely based on static prescription data and retrospective reporting, which limits their ability to detect early warning signs of misuse.

In many cases, risk indicators such as irregular dosing behavior, missed medication patterns, or sudden changes in usage are not captured in real time. This delay reduces the effectiveness of preventive interventions and contributes to escalation before action can be taken.

---

## Proposed Approach

NORIP introduces a behavioral risk intelligence framework that continuously evaluates medication-related events and identifies emerging risk patterns.

Each medication event is treated as a data point and analyzed in context. The system combines adherence monitoring, behavioral anomaly detection, and machine learning to generate a dynamic risk profile for each patient. Temporal analysis is used to track changes over time and detect sudden increases in risk.

This approach shifts monitoring from static observation to continuous behavioral analysis.

---

## System Capabilities

The platform captures medication events with timestamps and compares scheduled intake times with actual reported usage. It evaluates adherence patterns, including missed doses, late doses, and streak behavior.

Behavioral analysis identifies patterns such as early redosing, repeated dosing within short intervals, nighttime medication usage, and long gaps between doses. These signals are aggregated into a multi-factor risk scoring model that reflects both current behavior and recent trends.

The system integrates a machine learning component that estimates the probability of high-risk behavior using a Random Forest model. This is combined with rule-based scoring to provide both predictive and explainable outputs.

Temporal intelligence is implemented through rolling risk calculations and detection of sudden risk increases, enabling early identification of escalation patterns.

An analyst-facing dashboard provides visibility into patient-level risk, event history, and alert activity. The system also maintains a full audit trail of all data changes, supporting traceability and accountability.

A simplified patient interface allows users to log medication intake using a mobile-friendly design, including dose status, pain level, and optional notes.

---

## Significance and Need

This platform addresses a key limitation in existing opioid monitoring systems by introducing real-time behavioral intelligence.

Rather than relying solely on prescription data, NORIP evaluates how medication is actually used over time. This enables earlier detection of risk patterns that may indicate misuse or non-adherence.

The system demonstrates how cybersecurity methodologies, originally developed for detecting threats in digital environments, can be adapted to identify risks in healthcare settings. This interdisciplinary approach has potential applications in public health surveillance, clinical monitoring, and risk management.

---

## Public Deployment Model

This application is deployed as a public research prototype to support demonstration and evaluation.

Access controls have been intentionally omitted in this version to allow reviewers and stakeholders to fully explore system functionality without restrictions. The goal of this deployment is to provide transparency into the system design and enable direct interaction with its analytical capabilities.

---

## Data Considerations

All data used within the platform is synthetic or simulated. The system does not process or store real patient information and is not connected to external healthcare systems.

---

## Disclaimer

This platform is intended solely for research and demonstration purposes. It is not a medical device and should not be used for clinical decision-making, diagnosis, or treatment.

---

## Future Development

Future iterations of the system may include secure authentication, medication-specific rule integration, real-world data validation, and automated intervention workflows. These enhancements would support deployment in controlled environments where data security and access management are required.

---

## Technologies

The platform is implemented using Python and Streamlit for the interface layer, with data processing handled through Pandas and machine learning components built using Scikit-learn. Visualization is supported through Plotly, and persistence is managed using SQLite.


---

## Author

Louis Tidou
Technology Management (Cybersecurity)
=======
Louis Tidou (louistidou@gmail.com)
M.S. Technology Management (Cybersecurity)
University of Bridgeport

---

## License

This project is for educational and research purposes.
=======
This project is intended for research, educational, and demonstration purposes.

