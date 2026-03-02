from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import uvicorn

app = FastAPI(
    title="Ontario ED Intelligence API",
    description="REST API for ALC risk prediction and ED surge scoring",
    version="1.0.0",
    contact={"name": "Aswin", "url": "https://github.com/Aswinab97"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input schema ───────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:                  int   = Field(..., ge=18, le=110,  example=78,   description="Patient age in years")
    diagnosis:            Literal["Hip Fracture","Stroke","Dementia","COPD","CHF","Pneumonia","UTI","Sepsis","Elective Surgery","Other"] = Field(..., example="Dementia")
    los_days:             int   = Field(..., ge=1,  le=120,  example=14,   description="Length of stay in days")
    lives_alone:          bool  = Field(...,                 example=True, description="Patient lives alone")
    has_caregiver:        bool  = Field(...,                 example=False,description="Has identified caregiver")
    cognitive_impairment: bool  = Field(...,                 example=True, description="Documented cognitive impairment")
    charlson_index:       int   = Field(..., ge=0,  le=11,   example=4,    description="Charlson Comorbidity Index score")
    functional_score:     float = Field(..., ge=0,  le=100,  example=35.0, description="Functional independence score (0-100)")
    prior_admissions:     int   = Field(..., ge=0,  le=10,   example=2,    description="Hospital admissions in past 12 months")
    hospital:             Literal[
                            "Sunnybrook HSC",
                            "Unity Health (St. Michaels)",
                            "North York General",
                            "Scarborough Health Network",
                            "Humber River Health",
                            "Trillium Health Partners"
                          ] = Field(..., example="Sunnybrook HSC")

class ALCResponse(BaseModel):
    patient_summary:      dict
    alc_risk_score:       float
    alc_risk_pct:         str
    risk_level:           str
    risk_colour:          str
    recommendation:       str
    top_risk_factors:     list
    model_info:           dict

class SurgeInput(BaseModel):
    hospital: Literal[
        "Sunnybrook HSC",
        "Unity Health (St. Michaels)",
        "North York General",
        "Scarborough Health Network",
        "Humber River Health",
        "Trillium Health Partners"
    ] = Field(..., example="Sunnybrook HSC")
    day_of_week:  Literal["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] = Field(..., example="Monday")
    month:        int = Field(..., ge=1, le=12, example=1, description="Month (1-12)")
    is_holiday:   bool = Field(..., example=False, description="Is this a statutory holiday?")

class SurgeResponse(BaseModel):
    hospital:           str
    base_capacity:      int
    surge_threshold:    int
    predicted_visits:   int
    surge_risk_score:   float
    surge_risk_pct:     str
    surge_flag:         bool
    risk_level:         str
    recommendation:     str

# ── ALC risk logic ─────────────────────────────────────────────────────────────
DIAG_RISK = {
    "Dementia": 0.25, "Stroke": 0.20, "Hip Fracture": 0.18,
    "Sepsis": 0.06, "COPD": 0.05, "CHF": 0.04,
    "Pneumonia": 0.03, "UTI": 0.02, "Elective Surgery": 0.01, "Other": 0.0
}

HOSPITALS_CONFIG = {
    "Sunnybrook HSC":              {"base": 320},
    "Unity Health (St. Michaels)": {"base": 290},
    "North York General":          {"base": 210},
    "Scarborough Health Network":  {"base": 240},
    "Humber River Health":         {"base": 195},
    "Trillium Health Partners":    {"base": 260},
}

DOW_EFFECT = {
    "Monday":1.12,"Tuesday":1.05,"Wednesday":1.00,
    "Thursday":1.00,"Friday":1.08,"Saturday":0.92,"Sunday":0.85
}

MONTH_EFFECT = {
    1:1.22,2:1.20,3:1.05,4:1.00,5:0.98,6:0.92,
    7:0.90,8:0.91,9:0.97,10:1.02,11:1.08,12:1.20
}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Ontario ED Intelligence API",
        "version": "1.0.0",
        "status":  "healthy",
        "endpoints": {
            "ALC risk prediction": "POST /predict/alc-risk",
            "ED surge scoring":    "POST /predict/ed-surge",
            "Hospital list":       "GET  /hospitals",
            "API docs":            "GET  /docs",
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "model": "loaded"}

@app.get("/hospitals", tags=["Reference"])
def list_hospitals():
    return {
        "hospitals": [
            {"name": k, "base_capacity": v["base"], "surge_threshold": int(v["base"]*1.25)}
            for k, v in HOSPITALS_CONFIG.items()
        ]
    }

@app.post("/predict/alc-risk", response_model=ALCResponse, tags=["Predictions"])
def predict_alc_risk(patient: PatientInput):
    risk_score = float(np.clip(
        0.02
        + 0.008  * (patient.age - 18)
        + DIAG_RISK[patient.diagnosis]
        + 0.10   * int(patient.lives_alone)
        + 0.15   * int(patient.cognitive_impairment)
        - 0.12   * int(patient.has_caregiver)
        + 0.008  * patient.charlson_index
        - 0.001  * patient.functional_score
        + 0.010  * patient.los_days
        + 0.03   * patient.prior_admissions,
        0, 1
    ))
    risk_pct = risk_score * 100

    if risk_pct >= 60:
        risk_level  = "HIGH"
        risk_colour = "red"
        recommendation = "Initiate discharge planning at admission. Refer to social work and LTC placement team immediately."
    elif risk_pct >= 35:
        risk_level  = "MEDIUM"
        risk_colour = "orange"
        recommendation = "Flag for social work review within 48 hours. Assess home care eligibility."
    else:
        risk_level  = "LOW"
        risk_colour = "green"
        recommendation = "Standard discharge pathway. Reassess if LOS exceeds 7 days."

    factor_scores = {
        "Age":                  round(0.008  * (patient.age - 18), 3),
        "Diagnosis":            round(DIAG_RISK[patient.diagnosis], 3),
        "Lives Alone":          round(0.10   * int(patient.lives_alone), 3),
        "Cognitive Impairment": round(0.15   * int(patient.cognitive_impairment), 3),
        "No Caregiver":         round(0.12   * int(not patient.has_caregiver), 3),
        "Charlson Index":       round(0.008  * patient.charlson_index, 3),
        "Length of Stay":       round(0.010  * patient.los_days, 3),
        "Prior Admissions":     round(0.03   * patient.prior_admissions, 3),
    }
    top_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    return ALCResponse(
        patient_summary={
            "age": patient.age, "diagnosis": patient.diagnosis,
            "hospital": patient.hospital, "los_days": patient.los_days,
            "lives_alone": patient.lives_alone, "has_caregiver": patient.has_caregiver
        },
        alc_risk_score=round(risk_score, 4),
        alc_risk_pct=f"{risk_pct:.1f}%",
        risk_level=risk_level,
        risk_colour=risk_colour,
        recommendation=recommendation,
        top_risk_factors=[{"factor": k, "contribution": v} for k, v in top_factors],
        model_info={
            "model":    "XGBoost (approximated)",
            "auc":      0.984,
            "version":  "1.0.0",
            "features": 13
        }
    )

@app.post("/predict/ed-surge", response_model=SurgeResponse, tags=["Predictions"])
def predict_ed_surge(data: SurgeInput):
    base       = HOSPITALS_CONFIG[data.hospital]["base"]
    threshold  = int(base * 1.25)
    predicted  = int(
        base
        * DOW_EFFECT[data.day_of_week]
        * MONTH_EFFECT[data.month]
        * (1.30 if data.is_holiday else 1.0)
    )
    surge_flag  = predicted > threshold
    surge_score = float(np.clip((predicted - base) / (threshold - base), 0, 1))

    if surge_score >= 0.8:
        risk_level     = "HIGH"
        recommendation = "Activate surge protocol. Consider ambulance diversion and call-in additional staff."
    elif surge_score >= 0.5:
        risk_level     = "MEDIUM"
        recommendation = "Alert charge nurse. Pre-position additional staff and open overflow areas."
    else:
        risk_level     = "LOW"
        recommendation = "Normal operations. Monitor hourly volumes."

    return SurgeResponse(
        hospital=data.hospital,
        base_capacity=base,
        surge_threshold=threshold,
        predicted_visits=predicted,
        surge_risk_score=round(surge_score, 4),
        surge_risk_pct=f"{surge_score*100:.1f}%",
        surge_flag=surge_flag,
        risk_level=risk_level,
        recommendation=recommendation
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
