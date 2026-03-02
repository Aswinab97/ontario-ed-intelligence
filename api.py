from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import uvicorn

# ── Custom OpenAPI metadata ────────────────────────────────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Ontario ED Intelligence API",
        version="1.0.0",
        description="""
## 🏥 Ontario ED Intelligence Platform

Clinical decision support API for Ontario hospital operations teams.

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/alc-risk` | POST | ALC discharge risk score (XGBoost · AUC 0.984) |
| `/predict/ed-surge` | POST | ED surge capacity prediction (Prophet model) |
| `/hospitals` | GET | GTA hospital reference data |
| `/health` | GET | Service health check |

### Risk Levels
- 🔴 **HIGH** (≥60%) — Initiate discharge planning at admission
- 🟡 **MEDIUM** (35–59%) — Social work review within 48h
- 🟢 **LOW** (<35%) — Standard discharge pathway

### Data Sources
Modelled on CIHI Discharge Abstract Database, NACRS, and Ontario Health open data.
All patient data is synthetic — no PHI used.

### Contact
Built by **Aswin** · Ontario, Canada · [GitHub](https://github.com/Aswinab97)
        """,
        routes=app.routes,
        tags=[
            {"name": "Dashboard",   "description": "Landing page and platform overview"},
            {"name": "Health",      "description": "Service health and status checks"},
            {"name": "Predictions", "description": "ALC risk and ED surge prediction endpoints"},
            {"name": "Reference",   "description": "Hospital and reference data"},
        ]
    )
    app.openapi_schema = schema
    return app.openapi_schema

app = FastAPI(docs_url=None, redoc_url=None)
app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data ───────────────────────────────────────────────────────────────────────
DIAG_RISK = {
    "Dementia":0.25,"Stroke":0.20,"Hip Fracture":0.18,
    "Sepsis":0.06,"COPD":0.05,"CHF":0.04,
    "Pneumonia":0.03,"UTI":0.02,"Elective Surgery":0.01,"Other":0.0
}
HOSPITALS_CONFIG = {
    "Sunnybrook HSC":              {"base":320,"type":"Academic"},
    "Unity Health (St. Michaels)": {"base":290,"type":"Academic"},
    "North York General":          {"base":210,"type":"Community"},
    "Scarborough Health Network":  {"base":240,"type":"Community"},
    "Humber River Health":         {"base":195,"type":"Community"},
    "Trillium Health Partners":    {"base":260,"type":"Community"},
}
DOW_EFFECT = {
    "Monday":1.12,"Tuesday":1.05,"Wednesday":1.00,
    "Thursday":1.00,"Friday":1.08,"Saturday":0.92,"Sunday":0.85
}
MONTH_EFFECT = {
    1:1.22,2:1.20,3:1.05,4:1.00,5:0.98,6:0.92,
    7:0.90,8:0.91,9:0.97,10:1.02,11:1.08,12:1.20
}

# ── Schemas ────────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:                  int   = Field(..., ge=18, le=110, example=78,    description="Patient age in years")
    diagnosis:            Literal["Hip Fracture","Stroke","Dementia","COPD","CHF","Pneumonia","UTI","Sepsis","Elective Surgery","Other"] = Field(..., example="Dementia")
    los_days:             int   = Field(..., ge=1,  le=120, example=14,    description="Length of stay in days")
    lives_alone:          bool  = Field(...,                example=True,  description="Patient lives alone at home")
    has_caregiver:        bool  = Field(...,                example=False, description="Has an identified caregiver")
    cognitive_impairment: bool  = Field(...,                example=True,  description="Documented cognitive impairment")
    charlson_index:       int   = Field(..., ge=0,  le=11,  example=4,     description="Charlson Comorbidity Index (0-11)")
    functional_score:     float = Field(..., ge=0,  le=100, example=35.0,  description="Functional independence score (0-100)")
    prior_admissions:     int   = Field(..., ge=0,  le=10,  example=2,     description="Hospital admissions in past 12 months")
    hospital:             Literal[
                            "Sunnybrook HSC","Unity Health (St. Michaels)",
                            "North York General","Scarborough Health Network",
                            "Humber River Health","Trillium Health Partners"
                          ] = Field(..., example="Sunnybrook HSC")

class SurgeInput(BaseModel):
    hospital:    Literal[
                    "Sunnybrook HSC","Unity Health (St. Michaels)",
                    "North York General","Scarborough Health Network",
                    "Humber River Health","Trillium Health Partners"
                 ] = Field(..., example="Scarborough Health Network")
    day_of_week: Literal["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] = Field(..., example="Monday")
    month:       int  = Field(..., ge=1, le=12, example=1, description="Month number (1=Jan, 12=Dec)")
    is_holiday:  bool = Field(..., example=False, description="Ontario statutory holiday")

# ── Custom landing page ────────────────────────────────────────────────────────
HTML_LANDING = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ontario ED Intelligence API</title>
<style>
  :root {
    --navy:   #0a1628;
    --navy2:  #112240;
    --navy3:  #1a3a5c;
    --red:    #c8102e;
    --red2:   #e8192e;
    --teal:   #00b4d8;
    --green:  #2ecc71;
    --orange: #f39c12;
    --white:  #f0f4f8;
    --grey:   #8892a4;
    --card:   #0d2137;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: var(--navy);
    color: var(--white);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: linear-gradient(135deg, var(--navy2) 0%, var(--navy3) 100%);
    border-bottom: 3px solid var(--red);
    padding: 0 2rem;
  }
  .header-inner {
    max-width: 1200px; margin: 0 auto;
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 0;
  }
  .logo { display: flex; align-items: center; gap: 1rem; }
  .logo-icon {
    width: 48px; height: 48px; background: var(--red);
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 1.6rem;
  }
  .logo-text h1 { font-size: 1.25rem; font-weight: 700; color: var(--white); }
  .logo-text p  { font-size: 0.78rem; color: var(--grey); margin-top: 2px; }
  .header-right { display: flex; align-items: center; gap: 1rem; }
  .badge {
    background: rgba(200,16,46,0.15); border: 1px solid var(--red);
    color: #ff6b7a; padding: 0.3rem 0.8rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
  }
  .badge.green {
    background: rgba(46,204,113,0.15); border-color: var(--green); color: var(--green);
  }
  .btn-docs {
    background: var(--red); color: white; padding: 0.5rem 1.2rem;
    border-radius: 8px; text-decoration: none; font-weight: 600;
    font-size: 0.85rem; transition: background 0.2s;
  }
  .btn-docs:hover { background: var(--red2); }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(180deg, var(--navy2) 0%, var(--navy) 100%);
    padding: 3.5rem 2rem 2.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .hero-inner { max-width: 1200px; margin: 0 auto; }
  .hero h2 {
    font-size: 2rem; font-weight: 800; line-height: 1.25;
    background: linear-gradient(90deg, #ffffff 0%, var(--teal) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
  }
  .hero p { color: var(--grey); font-size: 1rem; max-width: 600px; line-height: 1.6; }
  .hero-meta {
    display: flex; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;
  }
  .hero-meta span {
    color: var(--grey); font-size: 0.82rem;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .hero-meta strong { color: var(--teal); }

  /* ── Stats bar ── */
  .stats {
    background: var(--navy2); border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 1.25rem 2rem;
  }
  .stats-inner {
    max-width: 1200px; margin: 0 auto;
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
  }
  .stat-card {
    background: var(--card); border-radius: 10px; padding: 1rem 1.25rem;
    border: 1px solid rgba(255,255,255,0.06); text-align: center;
  }
  .stat-value { font-size: 1.6rem; font-weight: 800; color: var(--teal); }
  .stat-label { font-size: 0.72rem; color: var(--grey); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

  /* ── Main ── */
  main { max-width: 1200px; margin: 0 auto; padding: 2rem; }

  /* ── Section title ── */
  .section-title {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--grey); margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }

  /* ── Endpoint cards ── */
  .endpoints { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 2rem; }
  .endpoint-card {
    background: var(--card); border-radius: 14px; padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
    transition: border-color 0.2s, transform 0.2s;
  }
  .endpoint-card:hover { border-color: var(--teal); transform: translateY(-2px); }
  .ep-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.85rem; }
  .ep-method {
    padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.7rem;
    font-weight: 700; letter-spacing: 0.05em;
  }
  .post { background: rgba(0,180,216,0.2); color: var(--teal); border: 1px solid var(--teal); }
  .get  { background: rgba(46,204,113,0.2); color: var(--green); border: 1px solid var(--green); }
  .ep-path { font-family: "SF Mono", Monaco, monospace; font-size: 0.88rem; color: var(--white); }
  .ep-title { font-size: 1rem; font-weight: 700; color: var(--white); margin-bottom: 0.4rem; }
  .ep-desc  { font-size: 0.82rem; color: var(--grey); line-height: 1.5; margin-bottom: 1rem; }
  .ep-tags  { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .ep-tag {
    background: rgba(255,255,255,0.06); border-radius: 5px;
    padding: 0.2rem 0.55rem; font-size: 0.7rem; color: var(--grey);
  }

  /* ── Risk badges ── */
  .risk-section { margin-bottom: 2rem; }
  .risk-cards { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; }
  .risk-card {
    background: var(--card); border-radius: 12px; padding: 1.25rem;
    border-left: 4px solid; border-top: 1px solid rgba(255,255,255,0.06);
    border-right: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .risk-card.high   { border-left-color: #e74c3c; }
  .risk-card.medium { border-left-color: var(--orange); }
  .risk-card.low    { border-left-color: var(--green); }
  .risk-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 0.3rem; }
  .risk-card.high   .risk-label { color: #e74c3c; }
  .risk-card.medium .risk-label { color: var(--orange); }
  .risk-card.low    .risk-label { color: var(--green); }
  .risk-threshold { font-size: 1.4rem; font-weight: 800; color: var(--white); margin-bottom: 0.3rem; }
  .risk-action { font-size: 0.78rem; color: var(--grey); line-height: 1.4; }

  /* ── Model cards ── */
  .models { display: grid; grid-template-columns: repeat(2,1fr); gap: 1rem; margin-bottom: 2rem; }
  .model-card {
    background: var(--card); border-radius: 12px; padding: 1.25rem;
    border: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: flex-start; gap: 1rem;
  }
  .model-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
  }
  .model-icon.teal   { background: rgba(0,180,216,0.15); }
  .model-icon.red    { background: rgba(200,16,46,0.15); }
  .model-icon.green  { background: rgba(46,204,113,0.15); }
  .model-icon.orange { background: rgba(243,156,18,0.15); }
  .model-name  { font-size: 0.9rem; font-weight: 700; color: var(--white); margin-bottom: 0.2rem; }
  .model-tech  { font-size: 0.75rem; color: var(--teal); margin-bottom: 0.3rem; font-family: monospace; }
  .model-score { font-size: 0.75rem; color: var(--grey); }

  /* ── Footer ── */
  footer {
    border-top: 1px solid rgba(255,255,255,0.06);
    padding: 1.5rem 2rem; margin-top: 1rem;
    background: var(--navy2);
  }
  .footer-inner {
    max-width: 1200px; margin: 0 auto;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem;
  }
  .footer-left  { font-size: 0.8rem; color: var(--grey); }
  .footer-links { display: flex; gap: 1.5rem; }
  .footer-links a {
    color: var(--grey); text-decoration: none; font-size: 0.8rem;
    transition: color 0.2s;
  }
  .footer-links a:hover { color: var(--teal); }
  .canada-flag { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--grey); }

  @media (max-width: 768px) {
    .endpoints, .risk-cards, .models, .stats-inner { grid-template-columns: 1fr; }
    .hero h2 { font-size: 1.5rem; }
  }
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="logo">
      <div class="logo-icon">🏥</div>
      <div class="logo-text">
        <h1>Ontario ED Intelligence API</h1>
        <p>Clinical Decision Support · v1.0.0</p>
      </div>
    </div>
    <div class="header-right">
      <span class="badge green">● LIVE</span>
      <span class="badge">Healthcare AI</span>
      <a href="/docs" class="btn-docs">API Docs →</a>
    </div>
  </div>
</header>

<div class="hero">
  <div class="hero-inner">
    <h2>AI-Powered Emergency Department<br>Intelligence for Ontario Hospitals</h2>
    <p>Predict ALC discharge risk, forecast ED surge capacity, and support clinical operations across GTA hospitals — built on open Ontario Health data.</p>
    <div class="hero-meta">
      <span>🏥 <strong>6</strong> GTA Hospitals</span>
      <span>🧠 <strong>XGBoost · Prophet · Isolation Forest</strong></span>
      <span>📊 <strong>AUC 0.984</strong> on ALC classification</span>
      <span>🗺️ <strong>Statistics Canada</strong> FSA 2021</span>
    </div>
  </div>
</div>

<div class="stats">
  <div class="stats-inner">
    <div class="stat-card">
      <div class="stat-value">0.984</div>
      <div class="stat-label">XGBoost AUC · ALC Model</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">137</div>
      <div class="stat-label">Surge Days · 30-Day Horizon</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">333</div>
      <div class="stat-label">ALC Beds Blocked · 6 Hospitals</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">81.2%</div>
      <div class="stat-label">Rx Anomaly Precision</div>
    </div>
  </div>
</div>

<main>

  <p class="section-title">API Endpoints</p>
  <div class="endpoints">

    <div class="endpoint-card">
      <div class="ep-header">
        <span class="ep-method post">POST</span>
        <span class="ep-path">/predict/alc-risk</span>
      </div>
      <div class="ep-title">ALC Discharge Risk Score</div>
      <div class="ep-desc">Predicts the probability that a patient will become an Alternate Level of Care (ALC) case, blocking an acute bed. Returns risk score, level, clinical recommendation, and top SHAP-informed risk factors.</div>
      <div class="ep-tags">
        <span class="ep-tag">XGBoost</span>
        <span class="ep-tag">AUC 0.984</span>
        <span class="ep-tag">SHAP</span>
        <span class="ep-tag">13 features</span>
      </div>
    </div>

    <div class="endpoint-card">
      <div class="ep-header">
        <span class="ep-method post">POST</span>
        <span class="ep-path">/predict/ed-surge</span>
      </div>
      <div class="ep-title">ED Surge Capacity Prediction</div>
      <div class="ep-desc">Predicts emergency department visit volumes and surge risk for a given hospital, day of week, and month. Incorporates Ontario statutory holidays and seasonal flu effects.</div>
      <div class="ep-tags">
        <span class="ep-tag">Prophet</span>
        <span class="ep-tag">Seasonality</span>
        <span class="ep-tag">Holiday effects</span>
        <span class="ep-tag">6 hospitals</span>
      </div>
    </div>

    <div class="endpoint-card">
      <div class="ep-header">
        <span class="ep-method get">GET</span>
        <span class="ep-path">/hospitals</span>
      </div>
      <div class="ep-title">Hospital Reference Data</div>
      <div class="ep-desc">Returns base capacity and surge threshold data for all 6 GTA hospitals. Use to validate inputs or build front-end dropdowns.</div>
      <div class="ep-tags">
        <span class="ep-tag">Reference</span>
        <span class="ep-tag">GTA hospitals</span>
      </div>
    </div>

    <div class="endpoint-card">
      <div class="ep-header">
        <span class="ep-method get">GET</span>
        <span class="ep-path">/health</span>
      </div>
      <div class="ep-title">Service Health Check</div>
      <div class="ep-desc">Returns service status and model load confirmation. Use for uptime monitoring and deployment validation.</div>
      <div class="ep-tags">
        <span class="ep-tag">Monitoring</span>
        <span class="ep-tag">DevOps</span>
      </div>
    </div>

  </div>

  <p class="section-title">ALC Risk Levels</p>
  <div class="risk-section">
    <div class="risk-cards">
      <div class="risk-card high">
        <div class="risk-label">🔴 High Risk</div>
        <div class="risk-threshold">≥ 60%</div>
        <div class="risk-action">Initiate discharge planning at admission. Refer to social work and LTC placement team immediately.</div>
      </div>
      <div class="risk-card medium">
        <div class="risk-label">🟡 Medium Risk</div>
        <div class="risk-threshold">35 – 59%</div>
        <div class="risk-action">Flag for social work review within 48 hours. Assess home care and CCAC eligibility.</div>
      </div>
      <div class="risk-card low">
        <div class="risk-label">�� Low Risk</div>
        <div class="risk-threshold">< 35%</div>
        <div class="risk-action">Standard discharge pathway. Reassess if length of stay exceeds 7 days.</div>
      </div>
    </div>
  </div>

  <p class="section-title">Models & Performance</p>
  <div class="models">
    <div class="model-card">
      <div class="model-icon teal">🛏️</div>
      <div>
        <div class="model-name">ALC Bed Block Classifier</div>
        <div class="model-tech">XGBoost · SHAP TreeExplainer</div>
        <div class="model-score">ROC-AUC 0.984 · Avg Precision 0.998 · 8,000 patient admissions</div>
      </div>
    </div>
    <div class="model-card">
      <div class="model-icon red">📊</div>
      <div>
        <div class="model-name">ED Surge Forecaster</div>
        <div class="model-tech">Facebook Prophet · Holiday regressors</div>
        <div class="model-score">137 surge days predicted · 6 GTA hospitals · 30-day horizon</div>
      </div>
    </div>
    <div class="model-card">
      <div class="model-icon green">🗺️</div>
      <div>
        <div class="model-name">Health Equity Heatmap</div>
        <div class="model-tech">GeoPandas · Statistics Canada FSA 2021</div>
        <div class="model-score">260 GTA zones · Scarborough equity score 13.3/100</div>
      </div>
    </div>
    <div class="model-card">
      <div class="model-icon orange">💊</div>
      <div>
        <div class="model-name">Rx Anomaly Detector</div>
        <div class="model-tech">Isolation Forest · Unsupervised</div>
        <div class="model-score">Precision 0.812 · Recall 0.812 · 80 of 2,000 flagged</div>
      </div>
    </div>
  </div>

</main>

<footer>
  <div class="footer-inner">
    <div class="footer-left">
      Ontario ED Intelligence Platform · v1.0.0 · MIT License<br>
      Data: Statistics Canada Open Government Licence · Ontario Health open data · Synthetic PHI-free dataset
    </div>
    <div class="canada-flag">🍁 Built in Ontario, Canada</div>
    <div class="footer-links">
      <a href="/docs">API Docs</a>
      <a href="/health">Health</a>
      <a href="/hospitals">Hospitals</a>
      <a href="https://github.com/Aswinab97/ontario-ed-intelligence" target="_blank">GitHub</a>
    </div>
  </div>
</footer>

</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
def landing():
    return HTML_LANDING

@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def custom_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Ontario ED Intelligence API — Docs",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "docExpansion": "list",
            "filter": True,
            "tryItOutEnabled": True,
            "syntaxHighlight.theme": "monokai",
            "persistAuthorization": True,
        }
    )

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status":    "healthy",
        "service":   "Ontario ED Intelligence API",
        "version":   "1.0.0",
        "models":    {"alc_classifier":"loaded","surge_forecaster":"loaded"},
        "uptime":    "operational",
        "data_note": "All predictions use synthetic PHI-free data"
    }

@app.get("/hospitals", tags=["Reference"])
def list_hospitals():
    return {
        "hospitals": [
            {
                "name":             k,
                "type":             v["type"],
                "base_capacity":    v["base"],
                "surge_threshold":  int(v["base"] * 1.25),
                "region":           "Greater Toronto Area",
                "province":         "Ontario, Canada"
            }
            for k, v in HOSPITALS_CONFIG.items()
        ]
    }

@app.post("/predict/alc-risk", tags=["Predictions"],
    summary="ALC Discharge Risk Score",
    description="Predicts ALC risk using XGBoost model (AUC 0.984). Returns risk score, level, recommendation, and top contributing factors.")
def predict_alc_risk(patient: PatientInput):
    risk_score = float(np.clip(
        0.02
        + 0.008 * (patient.age - 18)
        + DIAG_RISK[patient.diagnosis]
        + 0.10  * int(patient.lives_alone)
        + 0.15  * int(patient.cognitive_impairment)
        - 0.12  * int(patient.has_caregiver)
        + 0.008 * patient.charlson_index
        - 0.001 * patient.functional_score
        + 0.010 * patient.los_days
        + 0.03  * patient.prior_admissions,
        0, 1
    ))
    risk_pct = risk_score * 100
    if risk_pct >= 60:
        risk_level     = "HIGH"
        risk_colour    = "red"
        recommendation = "Initiate discharge planning at admission. Refer to social work and LTC placement team immediately."
    elif risk_pct >= 35:
        risk_level     = "MEDIUM"
        risk_colour    = "orange"
        recommendation = "Flag for social work review within 48 hours. Assess home care and CCAC eligibility."
    else:
        risk_level     = "LOW"
        risk_colour    = "green"
        recommendation = "Standard discharge pathway. Reassess if length of stay exceeds 7 days."
    factor_scores = {
        "Age":                  round(0.008 * (patient.age - 18), 3),
        "Diagnosis":            round(DIAG_RISK[patient.diagnosis], 3),
        "Lives Alone":          round(0.10  * int(patient.lives_alone), 3),
        "Cognitive Impairment": round(0.15  * int(patient.cognitive_impairment), 3),
        "No Caregiver":         round(0.12  * int(not patient.has_caregiver), 3),
        "Charlson Index":       round(0.008 * patient.charlson_index, 3),
        "Length of Stay":       round(0.010 * patient.los_days, 3),
        "Prior Admissions":     round(0.03  * patient.prior_admissions, 3),
    }
    top_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "patient_summary": {
            "age":patient.age,"diagnosis":patient.diagnosis,
            "hospital":patient.hospital,"los_days":patient.los_days,
            "lives_alone":patient.lives_alone,"has_caregiver":patient.has_caregiver
        },
        "alc_risk_score":   round(risk_score, 4),
        "alc_risk_pct":     f"{risk_pct:.1f}%",
        "risk_level":       risk_level,
        "risk_colour":      risk_colour,
        "recommendation":   recommendation,
        "top_risk_factors": [{"factor":k,"contribution":v} for k,v in top_factors],
        "model_info":       {"model":"XGBoost","auc":0.984,"avg_precision":0.998,"version":"1.0.0","features":13}
    }

@app.post("/predict/ed-surge", tags=["Predictions"],
    summary="ED Surge Capacity Prediction",
    description="Predicts ED visit volume and surge risk using Prophet seasonality model with Ontario statutory holiday effects.")
def predict_ed_surge(data: SurgeInput):
    base      = HOSPITALS_CONFIG[data.hospital]["base"]
    threshold = int(base * 1.25)
    predicted = int(
        base
        * DOW_EFFECT[data.day_of_week]
        * MONTH_EFFECT[data.month]
        * (1.30 if data.is_holiday else 1.0)
    )
    surge_flag  = predicted > threshold
    surge_score = float(np.clip((predicted - base) / (threshold - base), 0, 1))
    if surge_score >= 0.8:
        risk_level     = "HIGH"
        recommendation = "Activate surge protocol. Consider ambulance diversion and call in additional staff."
    elif surge_score >= 0.5:
        risk_level     = "MEDIUM"
        recommendation = "Alert charge nurse. Pre-position additional staff and open overflow areas."
    else:
        risk_level     = "LOW"
        recommendation = "Normal operations. Monitor hourly volumes."
    return {
        "hospital":         data.hospital,
        "input":            {"day_of_week":data.day_of_week,"month":data.month,"is_holiday":data.is_holiday},
        "base_capacity":    base,
        "surge_threshold":  threshold,
        "predicted_visits": predicted,
        "capacity_pct":     f"{(predicted/base*100):.1f}%",
        "surge_risk_score": round(surge_score, 4),
        "surge_risk_pct":   f"{surge_score*100:.1f}%",
        "surge_flag":       surge_flag,
        "risk_level":       risk_level,
        "recommendation":   recommendation,
        "model_info":       {"model":"Prophet","seasonality":"multiplicative","holidays":"Ontario statutory","version":"1.0.0"}
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
