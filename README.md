<div align="center">
    
# 🏥 Ontario Healthcare Intelligence Platform

</div>

<div align="center">

> **Provincial Hospital Operations Center & Clinical Machine Learning Suite**
> Performance Analytics · Throughput Forecasting · Geospatial Equity · Inpatient Risk Tracking · Controlled Substance Auditing

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

## 🌐 Live Deployment

> Deployed on **Azure Container Apps** — Canada Central 🍁

| Service | URL |
|---------|-----|
| 🏥 **Executive Operations Dashboard** | https://ontario-ed-dashboard.icydune-b6841f56.canadacentral.azurecontainerapps.io |
| 🔌 **FastAPI REST API** | https://ontario-ed-api.icydune-b6841f56.canadacentral.azurecontainerapps.io |
| 📖 **API Swagger Docs** | https://ontario-ed-api.icydune-b6841f56.canadacentral.azurecontainerapps.io/docs |

---

## 🎯 Project Overview

Ontario's health system requires integrated, scalable analytics to mitigate front-door overcrowding and back-door gridlock. Moving beyond isolated predictive experiments, this platform serves as a unified command infrastructure linking real geospatial boundary markers, downstream operational datasets, and advanced explainable machine learning models.

| Strategic Domain | Platform Capability | Core Architecture |
|------------------|---------------------|-------------------|
| **Executive Operations Center** | Multi-facility operational reporting, triage stratification, and automated root-cause diagnostics. | Dynamic MoM KPIs, CTAS & ICD-10 breakdown layers |
| **Throughput Forecasting** | Proactive, 30-day look-ahead projections for acute front-door volume arrivals. | Facebook Prophet time-series models with statutory holiday integration |
| **Geospatial Health Equity** | Neighborhood-level structural barriers and healthcare access disparity tracking. | GeoPandas FSA mapping over 260 Greater Toronto Area zones |
| **Inpatient Risk Tracking** | Automated detection of alternate level of care (ALC) exit blockages during initial admission. | XGBoost binary classifier + SHAP explainability values |
| **Controlled Substance Auditing** | Compliance tracking and prescribing pattern anomaly alerts across provider cohorts. | Isolation Forest unsupervised anomaly detection models |

---

## 📊 Analytics Architecture & System Design

### 🏠 Executive Operations Center & System Metrics
Serves as the primary system entry page, aggregating cross-facility performance across historical timelines and regional cohorts.
* **Operational Monitoring:** Breaks down macro system metrics across two distinct priority rows: *Demand/Throughput* (Total Visits, Wait Hours, LOS, Bed Occupancy) and *Quality/Exit Capacity* (Active ALC Beds, Admission Rates, LWBS Rates, Unplanned 30D Readmissions).
* **Clinical Triage Performance (CTAS):** Aggregates volumes and wait distributions by the Canadian Triage and Acuity Scale (CTAS 1–5), exposing clinical discordance bottlenecks where lower-acuity (CTAS 3/4) cohorts experience delayed wait times due to acute inpatient blocks.
* **Clinical Diagnostic Profiling (ICD-10):** Evaluates asset consumption metrics grouped by diagnostic category (e.g., F03 Dementia, I63 Ischemic Stroke), proving how post-acute placement limitations drive systemic gridlock.
* **Data Governance & Pipeline Integrity:** Embeds a live quality control audit matrix checking data completeness ratios, ICD-10 coding lag delays, timestamp index validity, and duplicate identifier metrics to verify submission compliance before provincial transfer.

---

### 📊 ED Surge Forecasting System
> Proactive capacity planning and diversion strategy mapping.
* **Model:** Facebook Prophet utilizing multiplicative seasonal parameters and custom provincial statutory holiday regressors.
* **Scope:** 6 core GTA hospital hubs (Sunnybrook HSC, Unity Health, North York General, Scarborough Health Network, Humber River Health, Trillium Health Partners).
* **Result:** 137 surge alerts flagged across a rolling 30-day horizon, evaluating structural volume shifts.

---

### 🗺️ Geospatial Health Equity Mapping
> Socioeconomic boundary analysis matching physical access bottlenecks.
* **Data Layer:** Statistics Canada Forward Sortation Area (FSA) 2021 digital boundary shapefiles.
* **Scope:** 260 unique GTA postal code boundary markers.
* **Result:** Confirmed stark geographic access variance, identifying structural high-need patterns in specific sub-regions (e.g., Scarborough M1W/M1N clusters).

---

### 🛏️ Inpatient ALC Bed Block Analysis
> Identifying acute-care exit bottlenecks before hallway medicine escalates.
* **Predictive Framework:** XGBoost ensemble binary classifier paired with SHAP TreeExplainer vectors.
* **Performance:** ROC-AUC: **0.984** | Average Precision: **0.998**.
* **Result:** Isolated 333 localized active bed block scenarios to focus clinical discharge teams.

---

### 💊 Controlled Substance Audit Engine
> Statistical guardrails and quality auditing over systemic prescribing cohorts.
* **Model:** Isolation Forest unsupervised outlier detection framework evaluating 2,000 active provider IDs.
* **Performance:** Precision: **0.812** | Recall: **0.812**.
* **Remediation Mapping:** Pinpoints outlier clinical footprints split between volume-driven variations, high-risk drug combinations, and outlier morphine milligram equivalents (MME).

---

## 🛠 Technical Infrastructure

| Layer | Technologies and Tools |
|---|---|
| **Languages & Core** | Python 3.10, NumPy, Pandas, Jupyter Workspace |
| **Statistical Modeling** | XGBoost, Facebook Prophet, scikit-learn (Isolation Forest) |
| **Model Explainability** | SHAP (SHapley Additive exPlanations) |
| **Geospatial Computation** | GeoPandas, Folium, Shapely, Fiona |
| **Interface & Data Layers** | Streamlit, FastAPI, Uvicorn Server, REST Architecture |
| **Containerization & Cloud** | Docker, Multistage Buildx, Azure Container Registry (ACR), Azure Container Apps |
| **Data Integrity** | Custom Automated Compliance Audit Framework (Schema & Null Validations) |

---

## 📁 Repository Structure

ontario-ed-intelligence/
├── .github/workflows/ci.yml   <- Automated CI linting & compilation
├── Dockerfile.api             <- Multistage Docker recipe for FastAPI service
├── Dockerfile.dashboard       <- Multistage Docker recipe for Streamlit client
├── fact_operations.csv        <- Executive Dashboard Data Layer (Operational Matrix)
├── data/
│   ├── raw/
│   └── processed/
│       ├── surge_risk_summary.csv
│       └── rx_audit_list.csv
├── notebooks/
│   ├── 01_EDA_Ontario_ED.ipynb
│   ├── 02_ED_Surge_Forecaster.ipynb
│   ├── 03_ALC_Bed_Block_Analyzer.ipynb
│   └── 04_Rx_Anomaly_Detector.ipynb
├── reports/
├── screenshots/
├── tests/
├── app.py                     <- Complete Professional Command Suite Entrypoint
├── main.py                    <- API Endpoint Handler
├── requirements.txt
└── README.md

---

## ⚠️ Data Governance & Disclaimer

All provider identifiers, patient traits, and operational daily volumes used outside geographical files are **synthetically modeled** utilizing non-identifiable provincial distributions. No Protected Health Information (PHI) or corporate records were ingested, complying fully with the *Personal Health Information Protection Act (PHIPA)*.

Public integration endpoints for migration:
* **NACRS (National Ambulatory Care Reporting System):** Direct real-time pipeline feed for emergency encounters.
* **DAD (Discharge Abstract Database):** Inpatient acute documentation integration for baseline LOS and clinical abstracts.
* **ODB (Ontario Drug Benefit System):** Pharmacy and clinician prescription validation registers.

---

## 👤 Author

**Aswin Anil Bindu** — Health Data Professional & Analytics Engineer
* 📍 Ontario, Canada
* 🔗 [GitHub](https://github.com/Aswinab97)
* 💼 [LinkedIn](https://www.linkedin.com/in/aswinab/)
