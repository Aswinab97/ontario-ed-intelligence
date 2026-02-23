# 🏥 Ontario ED Intelligence Platform

> **AI-powered emergency department analytics for Ontario hospitals**
> Built with real Statistics Canada data, Facebook Prophet, XGBoost, and SHAP explainability.

[![CI Pipeline](https://github.com/Aswinab97/ontario-ed-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Aswinab97/ontario-ed-intelligence/actions)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Project Overview

Ontario's emergency departments face a systemic crisis:
- **137 surge days** predicted across 6 GTA hospitals in the next 30 days
- **333 acute beds** currently blocked by ALC patients who should be in LTC
- **Scarborough FSAs** show the worst health equity scores in the GTA (13.3/100)
- **80 prescribers** flagged for anomalous opioid patterns across GTA hospitals

This platform provides hospital operations teams, Ontario Health planners, and clinical leadership with **actionable, explainable AI** — not black-box predictions.

---

## 📊 Modules

### Module 1 — ED Surge Forecaster
> *"Which hospitals will be over capacity in the next 7 days?"*

- **Model:** Facebook Prophet with Ontario statutory holiday effects
- **Hospitals:** Sunnybrook, Unity Health, North York General, Scarborough Health Network, Humber River Health, Trillium Health Partners
- **Result:** 137 surge days predicted across 6 hospitals in 30-day horizon

![ED Surge Dashboard](reports/gta_surge_dashboard.png)

---

### Module 2 — GTA Health Equity Heatmap
> *"Which neighbourhoods have the worst ED access and health outcomes?"*

- **Data:** Statistics Canada FSA Boundaries (2021) — real geographic data
- **Coverage:** 260 Forward Sortation Areas across Greater Toronto Area
- **Result:** Scarborough (M1W, M1N) confirmed as highest-need zones

![GTA Equity Heatmap](reports/gta_equity_heatmap.png)

---

### Module 3 — ALC Bed Block Analyzer
> *"Which patients are blocking acute beds and need discharge planning now?"*

- **Model:** XGBoost classifier with SHAP explainability
- **Performance:** ROC-AUC **0.984** | Average Precision **0.998**
- **Result:** 333 beds blocked across 6 hospitals (up to 87.5% ALC rate)
- **Top risk factors:** Age → Cognitive Impairment → Caregiver availability

![ALC SHAP](reports/alc_shap_explainability.png)

---

### Module 4 — Prescription Anomaly Detector
> *"Which prescribers have unusual opioid or polypharmacy patterns?"*

- **Model:** Isolation Forest (unsupervised anomaly detection)
- **Coverage:** 2,000 prescribers across GTA hospitals and community settings
- **Performance:** Precision **0.812** | Recall **0.812**
- **Breakdown:** 22 opioid over-prescribers | 20 volume outliers | 18 high-risk combos

![Rx Anomaly Detection](reports/rx_anomaly_detection.png)

---

## 📁 Repository Structure

    ontario-ed-intelligence/
    ├── .github/workflows/ci.yml
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    │   ├── 01_EDA_Ontario_ED.ipynb
    │   ├── 02_ED_Surge_Forecaster.ipynb
    │   ├── 03_ALC_Bed_Block_Analyzer.ipynb
    │   └── 04_Rx_Anomaly_Detector.ipynb
    ├── reports/
    ├── tests/
    ├── Dockerfile
    └── requirements.txt

---

## 🚀 Quick Start

    git clone https://github.com/Aswinab97/ontario-ed-intelligence.git
    cd ontario-ed-intelligence
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    jupyter notebook

---

## 📈 Key Results

| Module | Model | Metric | Result |
|--------|-------|--------|--------|
| ED Surge Forecaster | Prophet | Surge days (30-day) | **137 across 6 hospitals** |
| Equity Heatmap | GeoPandas | FSAs analysed | **260 GTA zones** |
| ALC Bed Block | XGBoost | ROC-AUC | **0.984** |
| ALC Bed Block | XGBoost | Beds blocked | **333 across 6 hospitals** |
| Rx Anomaly | Isolation Forest | Precision | **0.812** |
| Rx Anomaly | Isolation Forest | Flagged prescribers | **80 of 2,000** |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Languages | Python 3.10 |
| ML / AI | XGBoost, Prophet, Isolation Forest, scikit-learn |
| Explainability | SHAP |
| Geospatial | GeoPandas, Folium |
| Visualization | Matplotlib, Seaborn, Plotly |
| Data | Statistics Canada (FSA 2021), Ontario Health |
| Infrastructure | Docker, GitHub Actions CI/CD |

---

## 🏥 Ontario Health Context

This project addresses four of Ontario Health's top strategic priorities:

1. **ED Overcrowding** — Surge forecasting enables proactive staffing and diversion decisions
2. **Health Equity** — FSA-level mapping identifies underserved Scarborough communities
3. **ALC / LTC Pipeline** — Early ALC identification reduces hallway medicine and bed block
4. **Opioid Crisis** — Prescriber anomaly detection supports CPSO audit prioritization

---

## 📊 All Report Outputs

| File | Module | Description |
|------|--------|-------------|
| gta_fsa_base_map.png | Module 2 | GTA FSA base layer map |
| gta_equity_heatmap.png | Module 2 | Health equity choropleth |
| ed_trends_by_hospital.png | Module 1 | 3-year ED visit trends |
| ed_seasonality_patterns.png | Module 1 | Monthly and day-of-week patterns |
| sunnybrook_forecast.png | Module 1 | 30-day forecast with surge flags |
| gta_surge_dashboard.png | Module 1 | All 6 hospitals surge dashboard |
| alc_distribution.png | Module 3 | ALC rates by hospital and diagnosis |
| alc_model_performance.png | Module 3 | ROC curve, confusion matrix, score dist |
| alc_shap_explainability.png | Module 3 | SHAP feature importance and beeswarm |
| alc_beds_blocked_dashboard.png | Module 3 | Beds blocked by hospital |
| rx_prescribing_patterns.png | Module 4 | Prescribing patterns by specialty |
| rx_anomaly_detection.png | Module 4 | PCA and anomaly score distribution |
| rx_opioid_risk_quadrant.png | Module 4 | Opioid MME vs rate risk quadrant |

---

## 👤 Author

**Aswin** — Health Data Scientist
- 📍 Ontario, Canada
- 🔗 [GitHub](https://github.com/Aswinab97)

---

## 📄 License

MIT License

*Data sources: Statistics Canada Open Government Licence, Ontario Health open data*
