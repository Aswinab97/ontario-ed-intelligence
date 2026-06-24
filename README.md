<div align="center">

# 🏥 Ontario Healthcare Intelligence Platform

</div>

<div align="center">

> **Provincial Hospital Operations Center & Clinical Machine Learning Suite**
> Enterprise Operations Center · Throughput Forecasting · Geospatial Equity · Inpatient Capacity Risk Tracking · Controlled Substance Auditing

[![CI Pipeline](https://github.com/Aswinab97/ontario-ed-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Aswinab97/ontario-ed-intelligence/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-0.984_AUC-orange)
![Prophet](https://img.shields.io/badge/Prophet-Surge_Forecasting-blue)
![Azure](https://img.shields.io/badge/Azure-Container_Apps-0078D4?logo=microsoftazure)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

---

## 📌 Platform Intent & Overview

Ontario hospitals face two simultaneous crises: **front-door overcrowding** (patients can't get in) and **back-door gridlock** (patients can't get out). This platform connects both sides of that problem — combining operational reporting that hospital analysts use every day with predictive ML models that surface risk before it becomes a crisis.

Built to mirror the actual decision-support and performance-reporting workflows at Ontario Health, CIHI, regional OHTs, and prominent hospital networks.

---

## 🌐 Live Production Deployments

> Deployed on **Azure Container Apps** — Canada Central 🍁 utilizing Docker multistage builds and GitHub Actions CI/CD.

| Service / Interface | Operational Access URL |
|:---|:---|
| 🏥 **Executive Operations Dashboard** | https://ontario-ed-dashboard.icydune-b6841f56.canadacentral.azurecontainerapps.io |
| 🔌 **FastAPI Production Gateway** | https://ontario-ed-api.icydune-b6841f56.canadacentral.azurecontainerapps.io |
| 📖 **Interactive API OpenAPI Schema** | https://ontario-ed-api.icydune-b6841f56.canadacentral.azurecontainerapps.io/docs |

---

## 📸 Live Platform Screenshots

<div align="center">

### 📊 Executive Operations Center — Role-Based Dashboard
*Direct visibility into the health system's capacity, clinical triage delays, and data governance health metrics.*
<img width="1512" alt="Streamlit Dashboard" src="https://github.com/user-attachments/assets/b1d86d52-02a0-4993-8278-2f714dbd2993" />

<br><br>

### 📖 REST API Production Gateway (FastAPI OpenAPI/Swagger Documentation)
*High-throughput, asynchronous endpoints exposing containerized clinical model inferences for external EHR integration.*
<img width="1512" alt="Ontario ED Intelligence API Docs" src="https://github.com/user-attachments/assets/897b3f7b-bd2d-4e61-87a6-5d426e3481e3" />

<br><br>

### 🔬 Real-Time Inpatient Risk Inference Event
*Production JSON payload execution mapping multivariate clinical data to a real-time Alternate Level of Care (ALC) probability matrix.*
<img width="1512" height="982" alt="Live API Prediction — ALC Risk Score" src="https://github.com/user-attachments/assets/bca6828d-e2e9-4a53-a1c0-1b1e7eab65a1" />

</div>

---

## 🔬 Enterprise Core Architecture Layers

### 🏠 1. Executive Operations Center

The primary entry point for hospital leadership, aggregating multi-facility operational performance variables across a rolling 18-month window.

**Role-Based View Engine:** The dashboard dynamically restructures KPI block hierarchies and strategic brief diagnostics depending entirely on who is interacting with the data layer:

| Stakeholder Profile | Priority KPIs Surfaced | Strategic Directive Style |
|:---|:---|:---|
| 👑 **Strategic Executive (CEO/VP)** | Active ALC Beds · Bed Occupancy · Acute LOS · 30-Day Readmission | Regional LTC integration, escalation pathways |
| ⚙️ **Clinical Operations Manager** | Total ED Visits · Wait Time · Bed Occupancy · LWBS Rate | Shift scheduling, discharge coordination, throughput |
| 🩺 **Attending Physician / Clinical** | Wait Time · Admission Rate · Acute LOS · 30-Day Readmission | Order set protocols, early ALC flagging, case mix |

Month-over-month delta tracking on all KPIs with directional colour coding.

---

### 📊 2. Emergency Arrival Forecasting System

Proactive 30-day look-ahead projections for acute front-door volume across 6 GTA hospital hubs.

- **Core Model:** Facebook Prophet parameterized with multiplicative seasonal cycles and custom Ontario statutory holiday regressors
- **Scope:** Sunnybrook HSC · Unity Health · North York General · Scarborough Health Network · Humber River Health · Trillium Health Partners
- **Production Result:** 137 critical surge warning triggers generated over a rolling 30-day forecast horizon

<div align="center">
<img src="reports/ed_trends_by_hospital.png" width="750">
</div>

---

### 🎯 3. Triage Acuity Distribution (CTAS Analysis)

Breaks down emergency department performance benchmarks across all five Canadian Triage and Acuity Scale (CTAS) levels.

- **Clinical Bottleneck Discovery:** Surfaces volume distribution, mean wait times, and conversion admission rates per CTAS level
- **Key Finding:** CTAS 3/4 (Urgent/Less Urgent) cohorts experience the longest waits — inpatient bed blocks slowing throughput, not patient volume
- Facility-level filtering with month-over-month trend comparison

---

### 🔄 4. Unplanned Readmission Analytics Deep-Dive

CIHI-standard quality indicator tracking across 7-day and 30-day post-discharge return windows to detect care coordination gaps.

- **Facility Compliance Grid:** Stratifies hospitals by compliance states — *Meeting Target* · *Review Triggered* · *Action Required*
- **Alert:** Scarborough Health Network flagged at 14.5% 30-day readmission baseline → Action Required
- **Age Cohort Risk Map:** Readmission rate distribution spanning Under 18 through 85+ demographic bands

---

### 🫁 5. Clinical Diagnostic Profiling (ICD-10-CA)

Evaluates how abstracted clinical diagnostic chapters drive system resource consumption.

| ICD-10 Code | Condition | Avg LOS | 30-Day Readmission | ALC Attribution |
|:---|:---|:---|:---|:---|
| F03 | Dementia / Cognitive Decline | Highest | 13.0% | 54.2% |
| I63 | Acute Ischemic Stroke | High | 9.0% | 38.1% |
| I50 | Chronic Heart Failure | Moderate | 18.0% | 12.5% |
| J44 | COPD Exacerbation | Moderate | 15.0% | 9.0% |
| N39 | Urinary Tract Infection | Low | 4.0% | 1.2% |

**Key Finding:** Geriatric and cognitive decline profiles are the primary driver of ALC bed blocks — F03 Dementia patients average 2.4x system mean LOS.

---

### 🔍 6. Data Quality & Pipeline Governance Dashboard

Mirrors the real data stewardship and cleanup work health information management teams execute before monthly provincial submissions.

**Tracked indicators per facility:**
- ICD-10 coding lag (days to code post-discharge)
- Demographic field completeness percentage
- Null CTAS field count
- Timestamp sequence integrity validation
- Duplicate OHIP identifier rate
- Provincial submission readiness status

**Current Governance Alert:** Scarborough Health Network — 6.4-day ICD-10 coding lag + 2 timestamp anomalies detected → Pending audit review before monthly provincial submission lock. HIM notified for manual remediation.

---

### 🗄️ 7. Relational Star Schema Data Warehouse

The data infrastructure layer has transitioned from standard flat-file CSV pipelines into a robust, structured **Relational Star Schema Data Warehouse** engineered with **SQLite**. This architecture mirrors real-world Ontario administrative clinical environments (CIHI DAD/NACRS) where transactional hospital logs are aggregated into analytical data stores.

#### 🗺️ Data Warehouse Lineage Flow
```
[ Patient Arrives at ED ]
│
▼
[ CTAS Triage Evaluated ] ──► Captured in CIHI NACRS Registry
│
Is Inpatient Admission Required?
├──► NO  ──► Discharged ──► NACRS Record Closed
└──► YES ──► Transferred to Acute Bed
│
▼
Initiates CIHI DAD Registry
│
Track Inpatient LOS & ALC Status
│
▼
Closed Upon Abstract Discharge
```

---

#### 📐 Relational Schema Architecture
The warehouse implements an optimized Star Schema separating high-velocity operational facts from descriptive spatial metadata:
```
       ┌────────────────────────────────┐
       │     dim_hospital (Dimension)   │
       ├────────────────────────────────┤
       │  PK  │ hospital_id (TEXT)      │
       │      │ hospital_name (TEXT)    │
       │      │ region (TEXT)           │
       └───────────────┬────────────────┘
                       │ 1
                       │
                       │ ∞
┌─────────────────────────▼─────────────────────────┐
│             fact_operations (Fact Table)          │
├───────────────────────────────────────────────────┤
│  PK  │ fact_id (INTEGER - AUTOINCREMENT)          │
│  FK  │ hospital_id (TEXT)                         │
│      │ date (TEXT)                                │
│      │ total_ed_visits (INTEGER)                  │
│      │ mean_wait_time_hours (REAL)                │
│      │ admission_rate (REAL)                      │
│      │ bed_occupancy_rate (REAL)                  │
│      │ mean_los_days (REAL)                       │
│      │ active_alc_beds (INTEGER)                  │
│      │ readmission_rate_30d (REAL)                │
│      │ lwbs_rate (REAL)                           │
└───────────────────────────────────────────────────┘

```

#### 🔍 Interactive Live SQL Explorer UI
The application includes a production-grade **Healthcare SQL Warehouse Explorer** built into the Streamlit front-end. This module allows technical recruiters, data leads, and analysts to audit the underlying infrastructure natively:
* **Live Metadata Inspector:** Visualizes physical table schemas, column data types, and primary/foreign key relationships in real time.
* **Pre-Loaded Analytical Templates:** Features complex, pre-written diagnostic SQL queries optimized to pinpoint systemic hospital logjams (e.g., measuring "Exit-Block Bottlenecks" by matching high bed-occupancy ceilings against active ALC bed counts).
* **Dynamic Ad-Hoc Engine:** Provides a secure text canvas to write, compile, and execute raw SQL syntax directly against the SQLite engine, instantly outputting structured DataFrames alongside automated interactive visualizations.

```sql
-- Production Template Example: Exit-Block Throughput Gridlock Analysis
SELECT 
    h.hospital_name AS [Facility Name],
    f.date AS [Reporting Month],
    f.bed_occupancy_rate AS [Bed Occupancy (%)],
    f.active_alc_beds AS [Blocked ALC Beds],
    f.mean_wait_time_hours AS [ED Front-Door Wait Time (Hrs)]
FROM fact_operations f
JOIN dim_hospital h ON f.hospital_id = h.hospital_id
WHERE f.bed_occupancy_rate > 90.0 AND f.active_alc_beds > 40
ORDER BY f.mean_wait_time_hours DESC;
```

🗺️ 8. Geospatial Health Equity Mapping

Neighborhood-level structural access barrier analysis across regional GTA territories.
Data: Statistics Canada FSA 2021 digital boundary polygon shapefiles
Scope: 260 unique GTA Forward Sortation Areas

Result: Stark east-west equity gradient confirmed — Scarborough M1N/M1W at 13.3/100 vs North York at 95.0/100

---

🛏️ 9. Inpatient ALC Bed Block Risk Prediction

Identifies patients at risk of becoming Alternate Level of Care on day of admission — before the bed block occurs.
Model: XGBoost binary classifier + SHAP TreeExplainer (non-black-box risk surfacing)
Training Cohort: 8,000 patient admission records
Performance: ROC-AUC 0.984 | Average Precision 0.998
Result: 333 active bed block scenarios isolated across 6 GTA facilities
Top SHAP Risk Drivers:

| Rank	| | Feature |	| SHAP | | Value |
|:---|:---|:---|:---|:---|
1	Age	2.6561
2	Cognitive Impairment	1.4064
3	Has Caregiver	0.9888
4	Lives Alone	0.8703
5	Diagnosis Category	0.7848
Live interactive ALC Risk Calculator available directly in the dashboard.

---

💊 10. Controlled Substance Audit Engine
Prescribing pattern anomaly detection across regional provider cohorts.
Model: Isolation Forest unsupervised outlier detection framework
Scope: 2,000 active provider IDs across 6 hospital networks + community practice
Performance: Precision 0.812 | Recall 0.812
Anomaly Type	Count	Share	Remediation Pathway
Opioid Over-Prescriber	22	27.5%	CPSO Referral
Volume Outlier	20	25.0%	Billing Audit
High-Risk Drug Combinations	18	22.5%	Pharmacist Alert
Other Pattern Anomaly	20	25.0%	Manual Review

---

🏥 Ontario Healthcare Analyst Case Studies
Case Study 1 — Why Are ED Wait Times Increasing?
Problem: System-wide mean wait time increased 34% over 4 months.
Analysis:
Bed occupancy crossed the critical 92% threshold at 4 of 6 facilities
ALC patient volume increased 28% over the same period
CTAS 3/4 waits elevated disproportionately — inpatient blocks slowing ED throughput, not patient arrivals
Strategic Recommendation: Root cause is back-door gridlock, not front-door volume. Escalate LTC placement coordination. Automate bed escalation protocols when occupancy breaches 92%. Monitor ALC volume daily.
Case Study 2 — Which Patients Drive the Longest LOS?
Problem: Acute inpatient LOS trending upward, straining available bed supply.
Analysis:
F03 Dementia patients averaging 2.4x system mean LOS
54.2% of ALC-blocked beds attributed to cognitive decline and geriatric profiles
Charlson Comorbidity Index >4 strongly correlated with extended stay durations
Strategic Recommendation: Flag dementia and stroke admissions for social work review within 24 hours of admission. Initiate LTC placement referral at admission — not at discharge.
Case Study 3 — Which Facility Has the Highest Readmission Risk?
Problem: 30-day readmission rate variance spans 14.5% (Scarborough) vs 7.8% (North York General).
Analysis:
Scarborough patient cohort shows a higher proportion of 65+ age group
ICD-10 profile skewed toward CHF (I50) and COPD (J44) — both high-readmission diagnoses
Post-discharge follow-up gap likely contributing to early return events
Strategic Recommendation: Implement structured discharge checklists for CHF/COPD patients. Increase 7-day post-discharge phone follow-up frequency at Scarborough. Benchmark protocols against North York General.

---

🛠️ Technical Infrastructure
Layer	Technologies & Tools
Warehouse / Core Engine	SQLite · Relational Star Schema (Fact / Dimension Mapping)
Languages & Core	Python 3.10 · NumPy · Pandas · SQL (Structured Query Language)
ML & Forecasting	XGBoost · Facebook Prophet · scikit-learn (Isolation Forest)
Model Explainability	SHAP (SHapley Additive exPlanations)
Geospatial Processing	GeoPandas · Folium · Shapely · Fiona
Application & Server	Streamlit (Multi-page App + SQL Explorer Canvas) · FastAPI · Uvicorn
Cloud & DevOps	Docker · Azure Container Registry · Azure Container Apps · GitHub Actions CI/CD
Data Governance	Custom schema validation · PHIPA-compliant PHI de-identification framework

---

📁 Repository Structure
```
ontario-ed-intelligence/
├── .github/
│   └── workflows/ci.yml         <- Automated linting and image deployment pipelines
├── database/                    <- SQL Relational Warehouse Layer
│   ├── healthcare_warehouse.db  <- Compiled local SQLite relational database
│   └── create_database.py       <- ETL pipeline script parsing & seeding star schema
├── data/
│   ├── raw/                     <- Immutable raw baseline variables
│   └── processed/               <- Downstream calculated data assets
│       ├── surge_risk_summary.csv
│       └── rx_audit_list.csv
├── notebooks/                   <- Model exploration, training, and EDA notebooks
│   ├── 01_EDA_Ontario_ED.ipynb
│   ├── 02_ED_Surge_Forecaster.ipynb
│   ├── 03_ALC_Bed_Block_Analyzer.ipynb
│   └── 04_Rx_Anomaly_Detector.ipynb
├── reports/                     <- Stored visualizations and geospatial heatmaps
├── app.py                       <- Streamlit multi-module dashboard & SQL Explorer UI
├── main.py                      <- Asynchronous FastAPI backend server
├── Dockerfile.api               <- Multi-stage build recipe for API image
├── Dockerfile.dashboard         <- Multi-stage build recipe for dashboard image
├── requirements.txt
└── README.md
```

---

⚙️ Quick Start (Local Environment)

# Clone repository
git clone [https://github.com/Aswinab97/ontario-ed-intelligence.git](https://github.com/Aswinab97/ontario-ed-intelligence.git)
cd ontario-ed-intelligence

# Install dependencies
pip install -r requirements.txt

# Compile and Seed the Relational Star Schema Database
python database/create_database.py

# Launch Streamlit dashboard (Includes the SQL Warehouse Explorer page)
streamlit run app.py

# Launch FastAPI inference gateway (separate terminal)
uvicorn main:app --reload


---

⚠️ Data Governance, Privacy & Compliance
All patient traits, provider identifiers, and operational volumes used in this platform are synthetically modelled using non-identifiable provincial distributions. No Protected Health Information (PHI) or corporate records were ingested. This platform is fully compliant with the Personal Health Information Protection Act (PHIPA).
Real data integration points for a live production migration environment:
NACRS — National Ambulatory Care Reporting System (emergency encounter feeds)
DAD — Discharge Abstract Database (inpatient LOS and clinical abstract records)
ODB — Ontario Drug Benefit System (pharmacy dispensing and prescriber validation registers)

---


👤 Author
Aswin Anil Bindu — Healthcare Data & AI Analyst · Ontario, Canada
📄 License
Distributed under the open terms of the MIT License.
Contains information adapted from Statistics Canada Open Government Licence frameworks and Ontario Health open data repository metrics.
