<div align="center">

# 🏥 Ontario ED Intelligence Platform

### AI-powered Emergency Department Analytics for GTA Hospitals

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()
[![Data](https://img.shields.io/badge/Data-Ontario%20Open%20Data-blue)](https://data.ontario.ca)

> *Predicting ED overcrowding, mapping health equity gaps, and detecting prescription anomalies across GTA Ontario hospitals — using real Ontario government open data.*

</div>

---

## 📌 The Problem

Ontario's emergency departments are in crisis:

- 🚨 **ED overcrowding** is the #1 operational challenge across every GTA hospital
- 🛏️ **ALC (Alternate Level of Care) patients** block acute beds — delaying care for new patients
- 📍 **Health inequity** — residents of Scarborough and North York wait significantly longer than those in wealthier neighbourhoods
- 💊 **Opioid prescribing anomalies** cluster in specific FSAs, invisible without data-driven analysis

This platform uses **real Ontario open data** to surface these problems — and help health systems act before they become crises.

---

## 🏗️ Project Modules

### Module 1 — 📊 ED Surge Forecaster
> *"Which GTA hospitals will be over capacity this week?"*

- Time-series forecasting of ED visit volumes per hospital
- Models: **Facebook Prophet** + **LSTM** hybrid
- Features: seasonality, flu index, statutory holidays, day-of-week patterns
- Output: 7-day rolling forecast with surge-risk flags per hospital

### Module 2 — 🗺️ GTA Health Equity Heatmap
> *"Are people in Scarborough waiting twice as long as those in North York?"*

- Links ED wait times → neighbourhood income → ON-Marg deprivation index
- Interactive **choropleth map** of GTA by FSA (Forward Sortation Area)
- Equity gap score per hospital catchment zone
- Tools: `GeoPandas`, `Folium`, `Plotly`

### Module 3 — 🛏️ ALC Bed Block Analyzer
> *"How many beds are blocked by patients who should be in LTC or home care?"*

- Identifies ALC patient profiles using discharge pattern analysis
- ML classifier predicting ALC risk at admission
- SHAP explainability — showing *why* a patient is flagged
- Directly supports Ontario Health's ALC Reduction Strategy

### Module 4 — 💊 Prescription Anomaly Detector (NLP)
> *"Which GTA neighbourhoods show abnormal opioid prescribing patterns?"*

- NLP pipeline parsing Ontario Drug Benefit (ODB) open data
- Drug name normalization → therapeutic category mapping
- Anomaly detection flagging FSAs with unusual prescribing spikes
- Equity overlay: correlates prescribing anomalies with income/marginalization

---

## 🏥 Target Hospitals & Organizations

This platform is built around the real operational challenges of GTA health systems:

| Organization | Relevant Module |
|---|---|
| Unity Health Toronto (St. Michael's, St. Joseph's, Providence) | Module 1, 3 |
| University Health Network (TGH, TWH, PMH, Toronto Rehab) | Module 1, 2 |
| Sunnybrook Health Sciences Centre | Module 1, 2 |
| The Hospital for Sick Children (SickKids) | Module 1 |
| Mount Sinai Hospital | Module 1, 3 |
| North York General Hospital | Module 2, 3 |
| Humber River Health | Module 2, 3 |
| Scarborough Health Network | Module 2, 4 |
| Michael Garron Hospital | Module 2 |
| Trillium Health Partners | Module 3 |
| Mackenzie Health | Module 2, 3 |
| Lakeridge Health | Module 3 |
| William Osler Health System | Module 2, 3 |
| **ICES** | Module 2, 4 |
| **Ontario Health** | All Modules |

---

## 📦 Data Sources (All Free & Public)

| Dataset | Source | Used In |
|---|---|---|
| NACRS ED Visit Data | CIHI / Ontario Health | Module 1, 3 |
| Ontario Health Profiles | Ontario Government | Module 2 |
| Wellbeing Toronto | City of Toronto Open Data | Module 2 |
| Ontario Marginalization Index (ON-Marg) | ICES (public subset) | Module 2, 4 |
| Ontario Drug Benefit (ODB) Data | Ontario Government | Module 4 |
| OHIP Regional Billing Patterns | Ontario Government | Module 4 |
| Statistics Canada PUMF | Statistics Canada | Module 2 |
| GTA FSA Boundary Shapefiles | Statistics Canada | Module 2 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Languages** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy, GeoPandas |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Deep Learning / Time Series** | PyTorch, Prophet |
| **NLP** | spaCy, HuggingFace Transformers, sklearn-crfsuite |
| **Explainability** | SHAP, LIME |
| **Visualization** | Plotly, Folium, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **API** | FastAPI + Pydantic |
| **Infrastructure** | Docker, GitHub Actions CI/CD |
| **Database** | PostgreSQL / SQLite |

---

## 📁 Repository Structure

```
ontario-ed-intelligence/
│
├── 📂 data/
│   ├── raw/                        # Original downloaded datasets
│   └── processed/                  # Cleaned, analysis-ready data
│
├── 📂 modules/
│   ├── 01_ed_forecasting/          # ED surge predictor (Prophet + LSTM)
│   ├── 02_equity_heatmap/          # GTA health equity choropleth
│   ├── 03_alc_analyzer/            # ALC bed block ML model
│   └── 04_rx_anomaly_detector/     # Prescription NLP pipeline
│
├── 📂 dashboard/
│   └── app.py                      # Streamlit live dashboard
│
├── 📂 api/
│   └── main.py                     # FastAPI REST endpoints
│
├── 📂 notebooks/
│   ├── 01_EDA_Ontario_ED.ipynb
│   ├── 02_Equity_Analysis.ipynb
│   ├── 03_ALC_Modeling.ipynb
│   └── 04_Rx_Anomaly_Detection.ipynb
│
├── 📂 reports/
│   └── gta_health_equity_report.pdf  # Policy brief (ICES-style)
│
├── 📂 tests/
│   └── test_modules.py
│
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── ⚙️  .github/workflows/ci.yml
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🗺️ Roadmap

- [x] Repository setup & documentation
- [ ] Module 2 — GTA Health Equity Heatmap
- [ ] Module 1 — ED Surge Forecaster
- [ ] Module 3 — ALC Bed Block Analyzer
- [ ] Module 4 — Prescription Anomaly Detector
- [ ] Streamlit Dashboard (live demo)
- [ ] FastAPI backend
- [ ] Docker + CI/CD
- [ ] GTA Health Equity Policy Report (PDF)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Aswinab97/ontario-ed-intelligence.git
cd ontario-ed-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

---

## 👤 Author

**Aswin A B**
Targeting data science & health informatics roles across GTA Ontario healthcare.

[![GitHub](https://img.shields.io/badge/GitHub-Aswinab97-black?logo=github)](https://github.com/Aswinab97)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<i>Built with the goal of making Ontario's healthcare system smarter, fairer, and faster.</i>
</div>
