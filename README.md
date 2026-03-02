<div align="center">
    
# 🏥 Ontario ED Intelligence Platform

</div>

<div align="center">

> **AI-powered emergency department analytics for Ontario hospitals**
> Surge forecasting · Health equity mapping · ALC bed block detection · Prescription anomaly detection

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
| 📊 **Streamlit Dashboard** | https://ontario-ed-dashboard.icydune-b6841f56.canadacentral.azurecontainerapps.io |
| 🔌 **FastAPI REST API** | https://ontario-ed-api.icydune-b6841f56.canadacentral.azurecontainerapps.io |
| 📖 **API Swagger Docs** | https://ontario-ed-api.icydune-b6841f56.canadacentral.azurecontainerapps.io/docs |

---

## 🎯 Project Overview

Ontario's emergency departments face a systemic capacity crisis.
This platform gives hospital operations teams and Ontario Health planners
**actionable, explainable AI** - not black-box predictions.

| Problem | This Platform's Answer |
|---------|----------------------|
| Which hospitals will surge this week? | Prophet time-series forecast - 7 to 30 day horizon |
| Which neighbourhoods have the worst ED access? | GeoPandas FSA equity heatmap - 260 GTA zones |
| Which patients are blocking acute beds? | XGBoost + SHAP ALC classifier - AUC 0.984 |
| Which prescribers have abnormal opioid patterns? | Isolation Forest anomaly detector - 81.2% precision |

---

## 📸 Live Platform Screenshots

<div align="center">

### 📊 Streamlit Dashboard
<img width="1512" alt="Streamlit Dashboard" src="https://github.com/user-attachments/assets/b1d86d52-02a0-4993-8278-2f714dbd2993" />

<br><br>

### 📖 FastAPI Swagger Docs
<img width="1512" alt="Ontario ED Intelligence API" src="https://github.com/user-attachments/assets/897b3f7b-bd2d-4e61-87a6-5d426e3481e3" />

<br><br>

### 🔬 Live API Prediction — ALC Risk Score
<img width="1512" height="982" alt="Screenshot 2026-03-02 at 12 20 47 AM" src="https://github.com/user-attachments/assets/bca6828d-e2e9-4a53-a1c0-1b1e7eab65a1" />


</div>

## 📊 Modules

### Module 1 - ED Surge Forecaster
> Which hospitals will be over capacity in the next 7 days?

- **Model:** Facebook Prophet with Ontario statutory holiday regressors
- **Hospitals:** Sunnybrook HSC, Unity Health, North York General, Scarborough Health Network, Humber River Health, Trillium Health Partners
- **Result:** 137 surge days predicted across 6 hospitals in 30-day horizon

<div align="center">
<img src="reports/ed_trends_by_hospital.png" width="750">
</div>

---

### Module 2 - GTA Health Equity Heatmap
> Which neighbourhoods have the worst ED access and health outcomes?

- **Data:** Statistics Canada FSA Boundaries 2021 - real geographic shapefile
- **Coverage:** 260 Forward Sortation Areas across Greater Toronto Area
- **Result:** Scarborough (M1W, M1N) confirmed as highest-need zones

<div align="center">
<img src="reports/gta_fsa_base_map.png" width="750">
</div>

---

### Module 3 - ALC Bed Block Analyzer
> Which patients are blocking acute beds and need discharge planning today?

- **Model:** XGBoost binary classifier with SHAP explainability
- **ROC-AUC:** 0.984 | **Average Precision:** 0.998
- **Result:** 333 beds blocked across 6 hospitals

Top 5 ALC risk factors by SHAP:
1. Age - 2.6561
2. Cognitive Impairment - 1.4064
3. Has Caregiver - 0.9888
4. Lives Alone - 0.8703
5. Diagnosis - 0.7848

<div align="center">
<img src="reports/alc_model_performance.png" width="750">
</div>

---

### Module 4 - Prescription Anomaly Detector
> Which prescribers have unusual opioid or polypharmacy patterns?

- **Model:** Isolation Forest unsupervised anomaly detection
- **Coverage:** 2,000 prescribers across GTA hospitals and community settings
- **Precision:** 0.812 | **Recall:** 0.812

Anomaly breakdown:
- Opioid over-prescribers: 22 (27.5%)
- Volume outliers: 20 (25.0%)
- High-risk combinations: 18 (22.5%)
- Other anomalies: 20 (25.0%)

<div align="center">
<img src="reports/rx_opioid_risk_quadrant.png" width="750">
</div>

---

## 📈 Key Results Summary

| Module | Model | Key Metric | Result |
|--------|-------|------------|--------|
| ED Surge Forecaster | Facebook Prophet | Surge days (30-day) | **137 across 6 hospitals** |
| Health Equity Heatmap | GeoPandas + Folium | FSAs analysed | **260 GTA zones** |
| ALC Bed Block Analyzer | XGBoost + SHAP | ROC-AUC | **0.984** |
| ALC Bed Block Analyzer | XGBoost + SHAP | Beds blocked | **333 across 6 hospitals** |
| Rx Anomaly Detector | Isolation Forest | Precision / Recall | **0.812 / 0.812** |
| Rx Anomaly Detector | Isolation Forest | Flagged prescribers | **80 of 2,000** |

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python 3.10 |
| ML / Forecasting | XGBoost, Facebook Prophet, Isolation Forest, scikit-learn |
| Explainability | SHAP TreeExplainer |
| Geospatial | GeoPandas, Folium, Shapely |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| API | FastAPI + Uvicorn |
| Containerization | Docker, Docker Buildx (linux/amd64) |
| Registry | Azure Container Registry (ACR) |
| Deployment | Azure Container Apps — Canada Central |
| CI/CD | GitHub Actions |
| Data Sources | Statistics Canada FSA 2021, Ontario Health open data |

---

## ☁️ Azure Deployment Architecture

```
GitHub Actions CI
      │
      ▼
Docker Buildx (linux/amd64)
      │
      ▼
Azure Container Registry (ontarioedregistry.azurecr.io)
      │
      ├──► ontario-ed-api:v1        (FastAPI — port 8000)
      └──► ontario-ed-dashboard:v3  (Streamlit — port 8501)
                    │
                    ▼
      Azure Container Apps Environment
      ontario-ed-env — Canada Central
      ┌─────────────────────────────────────────┐
      │  ontario-ed-api                         │
      │  https://ontario-ed-api.icydune-...     │
      │  CPU: 0.5 | Memory: 1Gi                 │
      ├─────────────────────────────────────────┤
      │  ontario-ed-dashboard                   │
      │  https://ontario-ed-dashboard.icydune.. │
      │  CPU: 0.5 | Memory: 1Gi                 │
      └─────────────────────────────────────────┘
            Log Analytics Workspace
            workspace-ontarioedrgBklL
```

---

## 🚀 Deploy Your Own

### Prerequisites
- Azure CLI + Container Apps extension
- Docker Desktop with Buildx
- Azure subscription

### 1 — Clone and configure
```bash
git clone https://github.com/Aswinab97/ontario-ed-intelligence.git
cd ontario-ed-intelligence

export ACR_NAME=ontarioedregistry
export RESOURCE_GROUP=ontario-ed-rg
export ENV_NAME=ontario-ed-env
export LOCATION=canadacentral
export API_APP=ontario-ed-api
export DASH_APP=ontario-ed-dashboard
```

### 2 — Build and push images
```bash
az acr login --name $ACR_NAME

docker buildx build --platform linux/amd64 \
  -f Dockerfile.api \
  -t $ACR_NAME.azurecr.io/ontario-ed-api:v1 --push .

docker buildx build --platform linux/amd64 \
  -f Dockerfile.dashboard \
  -t $ACR_NAME.azurecr.io/ontario-ed-dashboard:v1 --push .
```

### 3 — Create environment and deploy
```bash
az containerapp env create \
  --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

ACR_PASSWORD=$(az acr credential show \
  --name $ACR_NAME --query "passwords[0].value" --output tsv)

az containerapp create \
  --name $API_APP \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $ACR_NAME.azurecr.io/ontario-ed-api:v1 \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 --ingress external \
  --cpu 0.5 --memory 1.0Gi

az containerapp create \
  --name $DASH_APP \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $ACR_NAME.azurecr.io/ontario-ed-dashboard:v1 \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8501 --ingress external \
  --cpu 0.5 --memory 1.0Gi
```

---

## 🚀 Quick Start (Local)

```bash
git clone https://github.com/Aswinab97/ontario-ed-intelligence.git
cd ontario-ed-intelligence
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## 📁 Repository Structure

```
ontario-ed-intelligence/
├── .github/workflows/ci.yml
├── Dockerfile.api
├── Dockerfile.dashboard
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
│   ├── dashboard.png
│   ├── api-docs.png
│   └── api-test.png
├── tests/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🏥 Ontario Health Context

| Priority | How This Platform Helps |
|----------|------------------------|
| **ED Overcrowding** | Surge forecasting enables proactive staffing and diversion decisions 7 days ahead |
| **Health Equity** | FSA-level mapping identifies underserved Scarborough communities for targeted investment |
| **ALC / LTC Pipeline** | Early ALC flag at admission enables same-day discharge planning and reduces hallway medicine |
| **Opioid Crisis** | Prescriber anomaly detection surfaces outliers for CPSO audit prioritization |

---

## ⚠️ Data Disclaimer

All patient, prescriber, and ED visit data is **synthetically generated**.
No real patient data or personal health information (PHI) is used.
Synthetic data is modelled on publicly available Ontario Health and Statistics Canada reports.

Real data integration points for production:
- NACRS for ED visit data
- Ontario Drug Benefit (ODB) database for prescribing patterns
- CIHI Discharge Abstract Database for ALC and LOS data
- Statistics Canada FSA boundaries (already integrated - real shapefile)

---

## 👤 Author

**Aswin** - Health Data Scientist
- 📍 Ontario, Canada
- 🔗 [GitHub](https://github.com/Aswinab97)

---

## 📄 License

MIT License

*Data sources: Statistics Canada Open Government Licence, Ontario Health open data*
