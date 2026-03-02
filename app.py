import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(
    page_title="Ontario ED Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🏥 Ontario ED Intelligence")
st.sidebar.markdown("---")
module = st.sidebar.radio("Select Module", [
    "🏠 Overview",
    "📊 Module 1 — ED Surge Forecaster",
    "🗺️ Module 2 — Health Equity Heatmap",
    "🛏️ Module 3 — ALC Bed Block Analyzer",
    "💊 Module 4 — Rx Anomaly Detector",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("Prophet · XGBoost · SHAP · Isolation Forest")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Aswin** · Ontario, Canada")

HOSPITALS_CONFIG = {
    "Sunnybrook HSC":              {"base": 320, "noise": 35, "type": "Academic"},
    "Unity Health (St. Michaels)": {"base": 290, "noise": 30, "type": "Academic"},
    "North York General":          {"base": 210, "noise": 25, "type": "Community"},
    "Scarborough Health Network":  {"base": 240, "noise": 28, "type": "Community"},
    "Humber River Health":         {"base": 195, "noise": 22, "type": "Community"},
    "Trillium Health Partners":    {"base": 260, "noise": 30, "type": "Community"},
}

@st.cache_data
def generate_ed_data():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    all_data = []
    for hospital, cfg in HOSPITALS_CONFIG.items():
        base, noise = cfg["base"], cfg["noise"]
        n = len(dates)
        visits = np.full(n, float(base))
        dow = {0:1.12,1:1.05,2:1.00,3:1.00,4:1.08,5:0.92,6:0.85}
        for i, d in enumerate(dates):
            visits[i] *= dow[d.dayofweek]
            if d.month in [12,1,2]:
                visits[i] *= np.random.uniform(1.15,1.30)
            elif d.month in [6,7,8]:
                visits[i] *= np.random.uniform(0.88,0.95)
            visits[i] *= (1.03 ** (d.year - 2022))
        visits += np.random.normal(0, noise, n)
        visits = np.maximum(visits, 50).round().astype(int)
        for i, d in enumerate(dates):
            all_data.append({
                "date": d, "hospital": hospital,
                "type": cfg["type"], "ed_visits": visits[i],
                "base_capacity": int(base * 1.25)
            })
    df = pd.DataFrame(all_data)
    df["surge_flag"] = (df["ed_visits"] > df["base_capacity"]).astype(int)
    return df

@st.cache_data
def generate_rx_data():
    np.random.seed(42)
    N = 2000
    specialties = ["Emergency Medicine","Internal Medicine","Family Medicine",
                   "Orthopedics","Oncology","Psychiatry","General Surgery","Geriatrics"]
    specialty = np.random.choice(specialties, size=N,
        p=[0.15,0.20,0.25,0.10,0.08,0.08,0.08,0.06])
    hospitals = list(HOSPITALS_CONFIG.keys()) + ["Community Practice"]
    hospital  = np.random.choice(hospitals, size=N,
        p=[0.12,0.12,0.12,0.12,0.12,0.12,0.28])
    opioid_rate   = np.random.normal(12,4,N).clip(0,35)
    avg_mme       = np.random.normal(45,15,N).clip(0,120)
    benzo_combo   = np.random.normal(3,1.5,N).clip(0,10)
    polypharmacy  = np.random.normal(18,5,N).clip(5,40)
    pts_per_month = np.random.normal(120,35,N).clip(10,300)
    avg_rx        = np.random.normal(3.2,0.8,N).clip(1,8)
    n_anom = int(N*0.04)
    anom_idx = np.random.choice(N, n_anom, replace=False)
    ta = anom_idx[:n_anom//3]
    tb = anom_idx[n_anom//3:2*n_anom//3]
    tc = anom_idx[2*n_anom//3:]
    opioid_rate[ta]  = (opioid_rate[ta]  * np.random.uniform(2.5,4.0,len(ta))).clip(0,100)
    avg_mme[ta]      = (avg_mme[ta]      * np.random.uniform(2.0,3.5,len(ta))).clip(0,500)
    benzo_combo[tb]  = (benzo_combo[tb]  * np.random.uniform(4.0,7.0,len(tb))).clip(0,100)
    polypharmacy[tb] = (polypharmacy[tb] * np.random.uniform(1.8,2.5,len(tb))).clip(0,100)
    pts_per_month[tc]= (pts_per_month[tc]* np.random.uniform(2.5,4.0,len(tc))).clip(0,800)
    avg_rx[tc]       = (avg_rx[tc]       * np.random.uniform(1.8,2.5,len(tc))).clip(0,20)
    true_anom = np.zeros(N,dtype=int)
    true_anom[anom_idx] = 1
    return pd.DataFrame({
        "prescriber_id":        [f"CPSO-{100000+i}" for i in range(N)],
        "specialty":             specialty,
        "hospital":              hospital,
        "opioid_rate_pct":       opioid_rate.round(2),
        "avg_opioid_mme":        avg_mme.round(1),
        "benzo_opioid_combo_pct":benzo_combo.round(2),
        "polypharmacy_pct":      polypharmacy.round(2),
        "patients_per_month":    pts_per_month.round(0).astype(int),
        "avg_rx_per_patient":    avg_rx.round(2),
        "true_anomaly":          true_anom
    })

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if module == "🏠 Overview":
    st.title("🏥 Ontario ED Intelligence Platform")
    st.markdown("> AI-powered emergency department analytics for Ontario hospitals")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Hospitals Monitored",   "6",    "GTA")
    col2.metric("Surge Days (30-day)",   "137",  "All HIGH risk")
    col3.metric("Beds Blocked (ALC)",    "333",  "80.8% avg rate")
    col4.metric("Rx Anomalies Flagged",  "80",   "of 2,000 prescribers")
    st.markdown("---")
    st.subheader("Platform Modules")
    c1,c2 = st.columns(2)
    with c1:
        st.info("**📊 Module 1 — ED Surge Forecaster**\n\nFacebook Prophet · 30-day horizon · 6 hospitals\n\n137 surge days predicted")
        st.success("**🗺️ Module 2 — Health Equity Heatmap**\n\nGeoPandas · Statistics Canada FSA 2021\n\n260 GTA zones mapped")
    with c2:
        st.warning("**🛏️ Module 3 — ALC Bed Block Analyzer**\n\nXGBoost + SHAP · ROC-AUC 0.984\n\n333 beds blocked identified")
        st.error("**💊 Module 4 — Rx Anomaly Detector**\n\nIsolation Forest · Precision 0.812\n\n80 prescribers flagged")
    st.markdown("---")
    st.subheader("Report Gallery")
    images = {
        "GTA Equity Heatmap":      "gta_equity_heatmap.png",
        "ED Surge Dashboard":      "gta_surge_dashboard.png",
        "ALC SHAP Explainability": "alc_shap_explainability.png",
        "Rx Anomaly Detection":    "rx_anomaly_detection.png",
    }
    cols = st.columns(2)
    for idx,(title,fname) in enumerate(images.items()):
        fpath = os.path.join("reports", fname)
        if os.path.exists(fpath):
            cols[idx%2].image(fpath, caption=title, use_column_width=True)

# ── MODULE 1 ───────────────────────────────────────────────────────────────────
elif module == "📊 Module 1 — ED Surge Forecaster":
    from prophet import Prophet
    st.title("📊 ED Surge Forecaster")
    st.markdown("Facebook Prophet · Ontario statutory holidays · 30-day ahead")
    st.markdown("---")
    hospital_choice = st.selectbox("Select Hospital", list(HOSPITALS_CONFIG.keys()))
    forecast_days   = st.slider("Forecast horizon (days)", 7, 60, 30)
    with st.spinner(f"Training Prophet model for {hospital_choice}..."):
        df_ed = generate_ed_data()
        hdf = df_ed[df_ed["hospital"]==hospital_choice][["date","ed_visits"]].copy()
        hdf.columns = ["ds","y"]
        ont_hols = pd.DataFrame({
            "holiday": "ontario_statutory",
            "ds": pd.to_datetime([
                "2022-01-01","2022-04-15","2022-07-01","2022-09-05","2022-12-26",
                "2023-01-01","2023-04-07","2023-07-01","2023-09-04","2023-12-25",
                "2024-01-01","2024-03-29","2024-07-01","2024-09-02","2024-12-25",
                "2025-01-01","2025-04-18","2025-07-01","2025-09-01","2025-12-25",
            ]),
            "lower_window": -1, "upper_window": 1,
        })
        m = Prophet(holidays=ont_hols, yearly_seasonality=True,
                    weekly_seasonality=True, seasonality_mode="multiplicative",
                    interval_width=0.90, changepoint_prior_scale=0.05)
        m.fit(hdf)
        future   = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
    threshold     = int(HOSPITALS_CONFIG[hospital_choice]["base"] * 1.25)
    forecast_only = forecast[forecast["ds"] > hdf["ds"].max()]
    surge_days    = forecast_only[forecast_only["yhat"] > threshold]
    col1,col2,col3 = st.columns(3)
    col1.metric("Surge Threshold",     f"{threshold} visits/day")
    col2.metric("Predicted Surge Days",f"{len(surge_days)} / {forecast_days}")
    risk = "🔴 HIGH" if len(surge_days)>10 else "🟡 MEDIUM" if len(surge_days)>4 else "🟢 LOW"
    col3.metric("Risk Level", risk)
    fig,ax = plt.subplots(figsize=(14,5))
    hist_90 = hdf.tail(90)
    ax.scatter(hist_90["ds"], hist_90["y"], color="#1f77b4", s=12, alpha=0.5, label="Actual")
    ax.plot(forecast_only["ds"], forecast_only["yhat"], color="#ff7f0e", lw=2, label="Forecast")
    ax.fill_between(forecast_only["ds"], forecast_only["yhat_lower"],
                    forecast_only["yhat_upper"], alpha=0.2, color="#ff7f0e", label="90% CI")
    ax.axhline(y=threshold, color="red", linestyle="--", lw=1.5, label=f"Surge threshold ({threshold})")
    if len(surge_days):
        ax.scatter(surge_days["ds"], surge_days["yhat"], color="red", s=60, zorder=5, label="Surge days")
    ax.axvline(x=hdf["ds"].max(), color="grey", linestyle=":", lw=1.5)
    ax.set_title(f"{hospital_choice} — {forecast_days}-Day Forecast", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily ED Visits")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.xticks(rotation=30); plt.tight_layout()
    st.pyplot(fig)
    if len(surge_days):
        st.warning(f"⚠️ {len(surge_days)} surge days predicted. First: {surge_days['ds'].min().strftime('%B %d, %Y')}")

# ── MODULE 2 ───────────────────────────────────────────────────────────────────
elif module == "🗺️ Module 2 — Health Equity Heatmap":
    st.title("🗺️ GTA Health Equity Heatmap")
    st.markdown("Statistics Canada FSA Boundaries 2021 · 260 GTA zones")
    st.markdown("---")
    col1,col2,col3 = st.columns(3)
    col1.metric("FSAs Analysed",       "260")
    col2.metric("Lowest Equity Score", "13.3 / 100", "Scarborough M1N")
    col3.metric("Highest Equity Score","95.0 / 100", "North York")
    st.markdown("---")
    c1,c2 = st.columns(2)
    for col,fname,title in [
        (c1,"gta_fsa_base_map.png",   "GTA FSA Base Map"),
        (c2,"gta_equity_heatmap.png", "Health Equity Heatmap"),
    ]:
        fpath = os.path.join("reports",fname)
        if os.path.exists(fpath):
            col.image(fpath, caption=title, use_column_width=True)
    st.info("**Key finding:** Scarborough (M1N, M1W) equity score 13.3/100 vs North York 95.0/100. Clear east-west equity gradient across the GTA.")

# ── MODULE 3 ───────────────────────────────────────────────────────────────────
elif module == "🛏️ Module 3 — ALC Bed Block Analyzer":
    st.title("🛏️ ALC Bed Block Analyzer")
    st.markdown("XGBoost + SHAP · ROC-AUC 0.984 · 8,000 patient admissions")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("ROC-AUC",        "0.984")
    col2.metric("Avg Precision",  "0.998")
    col3.metric("Beds Blocked",   "333",  "across 6 hospitals")
    col4.metric("Top Risk Factor","Age",  "SHAP 2.66")
    st.markdown("---")
    tab1,tab2,tab3 = st.tabs(["📊 Distribution","🤖 Model Performance","🔍 SHAP"])
    with tab1:
        fpath = os.path.join("reports","alc_distribution.png")
        if os.path.exists(fpath): st.image(fpath, use_column_width=True)
        st.subheader("Beds Blocked by Hospital")
        beds_data = pd.DataFrame({
            "Hospital":      list(HOSPITALS_CONFIG.keys()),
            "Beds Blocked":  [63,63,57,52,51,47],
            "ALC Rate (%)":  [80.8,87.5,83.8,85.2,77.3,85.5],
        }).sort_values("Beds Blocked",ascending=False)
        st.dataframe(beds_data, use_column_width=True, hide_index=True)
    with tab2:
        fpath = os.path.join("reports","alc_model_performance.png")
        if os.path.exists(fpath): st.image(fpath, use_column_width=True)
    with tab3:
        fpath = os.path.join("reports","alc_shap_explainability.png")
        if os.path.exists(fpath): st.image(fpath, use_column_width=True)
        shap_data = pd.DataFrame({
            "Rank":    [1,2,3,4,5],
            "Feature": ["Age","Cognitive Impairment","Has Caregiver","Lives Alone","Diagnosis"],
            "SHAP":    [2.6561,1.4064,0.9888,0.8703,0.7848],
        })
        st.dataframe(shap_data, use_column_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("🧮 ALC Risk Calculator")
    c1,c2,c3 = st.columns(3)
    with c1:
        p_age  = st.slider("Patient Age", 18, 100, 78)
        p_diag = st.selectbox("Diagnosis",["Hip Fracture","Stroke","Dementia","COPD","CHF","Pneumonia","UTI","Sepsis","Elective Surgery","Other"])
        p_los  = st.slider("Length of Stay (days)", 1, 60, 12)
    with c2:
        p_alone = st.checkbox("Lives Alone",           value=True)
        p_carer = st.checkbox("Has Caregiver",         value=False)
        p_cogn  = st.checkbox("Cognitive Impairment",  value=True)
    with c3:
        p_charlson = st.slider("Charlson Index", 0, 11, 4)
        p_func     = st.slider("Functional Score (0-100)", 0, 100, 35)
        p_prior    = st.slider("Prior Admissions", 0, 10, 2)
    diag_risk = {"Dementia":0.25,"Stroke":0.20,"Hip Fracture":0.18,"COPD":0.05,
                 "CHF":0.04,"Pneumonia":0.03,"UTI":0.02,"Sepsis":0.06,"Elective Surgery":0.01,"Other":0.0}
    risk_score = float(np.clip(
        0.02 + 0.008*(p_age-18) + diag_risk[p_diag]
        + 0.10*p_alone + 0.15*p_cogn - 0.12*p_carer
        + 0.008*p_charlson - 0.001*p_func
        + 0.010*p_los + 0.03*p_prior, 0, 1))
    risk_pct = risk_score * 100
    st.markdown("---")
    if risk_pct >= 60:
        st.error(f"🔴 HIGH ALC RISK — {risk_pct:.1f}% · Initiate discharge planning at admission")
    elif risk_pct >= 35:
        st.warning(f"🟡 MEDIUM ALC RISK — {risk_pct:.1f}% · Flag for social work review within 48h")
    else:
        st.success(f"🟢 LOW ALC RISK — {risk_pct:.1f}% · Standard discharge pathway")
    st.progress(min(risk_score, 1.0))

# ── MODULE 4 ───────────────────────────────────────────────────────────────────
elif module == "💊 Module 4 — Rx Anomaly Detector":
    st.title("💊 Prescription Anomaly Detector")
    st.markdown("Isolation Forest · 2,000 prescribers · Precision 0.812 · Recall 0.812")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Prescribers Analysed","2,000")
    col2.metric("Anomalies Detected",  "80",  "4.0% rate")
    col3.metric("Precision",           "0.812")
    col4.metric("Recall",              "0.812")
    st.markdown("---")
    tab1,tab2,tab3 = st.tabs(["📊 Patterns","🔍 Anomaly Detection","📋 Audit List"])
    with tab1:
        fpath = os.path.join("reports","rx_prescribing_patterns.png")
        if os.path.exists(fpath): st.image(fpath, use_column_width=True)
    with tab2:
        c1,c2 = st.columns(2)
        for col,fname,title in [
            (c1,"rx_anomaly_detection.png",  "PCA + Anomaly Score"),
            (c2,"rx_opioid_risk_quadrant.png","Opioid Risk Quadrant"),
        ]:
            fpath = os.path.join("reports",fname)
            if os.path.exists(fpath): col.image(fpath, caption=title, use_column_width=True)
        anom_df = pd.DataFrame({
            "Anomaly Type":  ["Opioid Over-Prescriber","Volume Outlier","Other Anomaly","High-Risk Combinations"],
            "Count":         [22,20,20,18],
            "Percentage":    ["27.5%","25.0%","25.0%","22.5%"],
            "Action":        ["CPSO referral","Billing audit","Manual review","Pharmacist alert"],
        })
        st.dataframe(anom_df, use_column_width=True, hide_index=True)
    with tab3:
        processed_path = "data/processed/rx_audit_list.csv"
        if os.path.exists(processed_path):
            st.dataframe(pd.read_csv(processed_path).head(20), use_column_width=True, hide_index=True)
        else:
            st.info("Run Notebook 04 to generate the audit list CSV")
        st.markdown("---")
        st.subheader("Prescriber Lookup")
        df_rx = generate_rx_data()
        spec_filter = st.multiselect("Filter by Specialty",
            df_rx["specialty"].unique().tolist(),
            default=["Emergency Medicine","Family Medicine"])
        hosp_filter = st.multiselect("Filter by Hospital",
            df_rx["hospital"].unique().tolist(),
            default=list(HOSPITALS_CONFIG.keys())[:2])
        if spec_filter and hosp_filter:
            filtered = df_rx[
                df_rx["specialty"].isin(spec_filter) &
                df_rx["hospital"].isin(hosp_filter)
            ][["prescriber_id","specialty","hospital","opioid_rate_pct","avg_opioid_mme","patients_per_month"]]
            st.dataframe(filtered.head(50), use_column_width=True, hide_index=True)
