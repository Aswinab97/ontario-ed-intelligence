import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from prophet import Prophet

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── INITIAL WORKSPACE SETUP ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Ontario Healthcare Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core Hospital Configuration Blueprint
HOSPITALS_CONFIG = {
    "Sunnybrook HSC":              {"base": 320, "noise": 35, "type": "Academic"},
    "Unity Health (St. Michaels)": {"base": 290, "noise": 30, "type": "Academic"},
    "North York General":          {"base": 210, "noise": 25, "type": "Community"},
    "Scarborough Health Network":  {"base": 240, "noise": 28, "type": "Community"},
    "Humber River Health":         {"base": 195, "noise": 22, "type": "Community"},
    "Trillium Health Partners":    {"base": 260, "noise": 30, "type": "Community"},
}

# ── CENTRALIZED CACHED DATASETS ────────────────────────────────────────────────
@st.cache_data
def load_fact_operations():
    """Loads or safely falls back on the newly designed Executive Operations Data Layer"""
    if os.path.exists("fact_operations.csv"):
        df = pd.read_csv("fact_operations.csv")
        if "CTAS_1_Visits" in df.columns:
            df["Month"] = pd.to_datetime(df["Month"])
            return df
    
    months = pd.date_range("2025-01-01", "2026-06-01", freq="MS")
    fallback_data = []
    for hospital, cfg in HOSPITALS_CONFIG.items():
        for month in months:
            bed_occupancy = np.random.uniform(85, 98)
            alc_patients = np.random.uniform(15, 45) + (bed_occupancy - 85) * 0.5
            total_visits = int(cfg["base"] * np.random.uniform(28, 31))
            
            c1_vol = int(total_visits * 0.015)
            c2_vol = int(total_visits * 0.15)
            c3_vol = int(total_visits * 0.35)
            c4_vol = int(total_visits * 0.35)
            c5_vol = total_visits - (c1_vol + c2_vol + c3_vol + c4_vol)
            
            wait_time_global = 2.5 + (bed_occupancy - 80) * 0.12 + alc_patients * 0.04
            
            fallback_data.append({
                "Hospital": hospital, "Month": month,
                "ED_Visits": total_visits,
                "Wait_Time_Hours": round(max(1.5, wait_time_global), 2),
                "LOS_Days": round(4 + alc_patients * 0.03, 2),
                "Bed_Occupancy_Pct": round(bed_occupancy, 2),
                "ALC_Patients": int(alc_patients),
                "Admission_Rate_Pct": round(np.random.uniform(12, 20), 2),
                "LWBS_Rate_Pct": round(1.5 + wait_time_global * 0.7, 2),
                "Readmission_Rate_Pct": round(np.random.uniform(7, 15), 2),
                "Staffing_Index": int(np.random.choice([95, 100, 105])),
                "CTAS_1_Visits": c1_vol, "CTAS_2_Visits": c2_vol, "CTAS_3_Visits": c3_vol, "CTAS_4_Visits": c4_vol, "CTAS_5_Visits": c5_vol,
                "CTAS_1_Wait_Hrs": round(np.random.uniform(0.1, 0.3), 2), "CTAS_2_Wait_Hrs": round(1.2 + (bed_occupancy - 80)*0.03, 2),
                "CTAS_3_Wait_Hrs": round(wait_time_global * 1.15, 2), "CTAS_4_Wait_Hrs": round(wait_time_global * 1.30, 2), "CTAS_5_Wait_Hrs": round(wait_time_global * 0.85, 2),
                "CTAS_1_Admission_Pct": 85.0, "CTAS_2_Admission_Pct": 42.0, "CTAS_3_Admission_Pct": 18.0, "CTAS_4_Admission_Pct": 4.5, "CTAS_5_Admission_Pct": 0.5
            })
    df_fallback = pd.DataFrame(fallback_data)
    df_fallback.to_csv("fact_operations.csv", index=False)
    return df_fallback

@st.cache_data
def generate_ed_data():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
    all_data = []
    for hospital, cfg in HOSPITALS_CONFIG.items():
        base, noise = cfg["base"], cfg["noise"]
        n = len(dates)
        visits = np.full(n, float(base))
        dow = {0:1.12, 1:1.05, 2:1.00, 3:1.00, 4:1.08, 5:0.92, 6:0.85}
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
    specialty = np.random.choice(specialties, size=N, p=[0.15,0.20,0.25,0.10,0.08,0.08,0.08,0.06])
    hospitals = list(HOSPITALS_CONFIG.keys()) + ["Community Practice"]
    hospital  = np.random.choice(hospitals, size=N, p=[0.12,0.12,0.12,0.12,0.12,0.12,0.28])
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

# ── SIDEBAR NAVIGATION PANEL ──────────────────────────────────────────────────
st.sidebar.title("🏥 Health Intelligence Platform")
st.sidebar.markdown("---")
module = st.sidebar.radio("Executive Command Suite", [
    "🏠 Executive Operations Center",
    "📊 ED Surge Forecasting System",
    "🗺️ Geospatial Health Equity Mapping",
    "🛏️ Inpatient ALC Bed Block Analysis",
    "💊 Controlled Substance Audit Engine",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Analytics Architecture**")
st.sidebar.markdown("Prophet · XGBoost · SHAP · Isolation Forest")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Aswin** · Ontario, Canada")

# ── 🏠 EXECUTIVE OPERATIONS CENTER (HOMEPAGE) ──────────────────────────────────
if module == "🏠 Executive Operations Center":
    st.title("🏥 Ontario Health Executive Operations Center")
    st.caption("Strategic Performance Analysis, Throughput Velocity & Alternate Level of Care (ALC) Surveillance")
    st.markdown("---")
    
    df_ops = load_fact_operations()
    
    st.sidebar.header("Operations Scoping Filters")
    facility_list = ["All Ontario Facilities"] + list(df_ops["Hospital"].unique())
    selected_facility = st.sidebar.selectbox("Scope Health System", facility_list, key="exec_dashboard_facility_select")
    
    if selected_facility != "All Ontario Facilities":
        filtered_ops = df_ops[df_ops["Hospital"] == selected_facility]
    else:
        filtered_ops = df_ops

    latest_month = filtered_ops["Month"].max()
    prev_month = latest_month - pd.DateOffset(months=1)
    
    curr_df = filtered_ops[filtered_ops["Month"] == latest_month]
    prev_df = filtered_ops[filtered_ops["Month"] == prev_month]

    st.error("📋 **Analyst Operations Brief — System Exit Block & Capacity Gridlock Advisory**")
    ins_col1, ins_col2 = st.columns([2, 3])
    
    with ins_col1:
        st.markdown(f"""
        **System Performance Observations ({latest_month.strftime('%B %Y')}):**
        * **ED Front-Door Performance:** Median operational wait times have skewed upward sharply.
        * **LWBS Risk Scaling:** High wait times continue to drive patient flight (LWBS) rates near upper control bounds.
        * **Volume Baselines:** Front-door emergency arrival vectors are well within standard +/-5% seasonal standard deviations.
        """)
    with ins_col2:
        st.markdown("""
        **Operational Diagnostic & Action Pathway:**
        * **Root-Cause Matrix:** This issue is an *exit-block* from high alternate level of care (ALC) inpatient volumes. Acute bed capacity is locked at critical thresholds, slowing internal patient migration from the ED to medicine wards.
        * **Strategic Directives:** Deploy coordinated transitional care initiatives alongside regional long-term care partners; execute mandatory automated escalation protocols when bed occupancy thresholds breach 92%.
        """)
        
    st.markdown("---")

    st.subheader(f"Provincial Matrix Health Indicators — {latest_month.strftime('%B %Y')}")
    
    def calculate_kpi_metrics(column_name, aggregation_strategy="mean"):
        current_aggregated = curr_df[column_name].agg(aggregation_strategy)
        previous_aggregated = prev_df[column_name].agg(aggregation_strategy)
        percentage_delta = ((current_aggregated - previous_aggregated) / previous_aggregated) * 100 if previous_aggregated != 0 else 0
        return current_aggregated, percentage_delta

    visits_v, visits_d = calculate_kpi_metrics("ED_Visits", "sum")
    wait_v, wait_d = calculate_kpi_metrics("Wait_Time_Hours", "mean")
    los_v, los_d = calculate_kpi_metrics("LOS_Days", "mean")
    occupancy_v, occupancy_d = calculate_kpi_metrics("Bed_Occupancy_Pct", "mean")
    
    alc_v, alc_d = calculate_kpi_metrics("ALC_Patients", "sum")
    admission_v, admission_d = calculate_kpi_metrics("Admission_Rate_Pct", "mean")
    lwbs_v, lwbs_d = calculate_kpi_metrics("LWBS_Rate_Pct", "mean")
    readm_v, readm_d = calculate_kpi_metrics("Readmission_Rate_Pct", "mean")

    row1_1, row1_2, row1_3, row1_4 = st.columns(4)
    row1_1.metric("Total ED Visits", f"{int(visits_v):,}", f"{visits_d:+.1f}% MoM", delta_color="inverse")
    row1_2.metric("Mean Wait Time", f"{wait_v:.2f} hrs", f"{wait_d:+.1f}% MoM", delta_color="inverse")
    row1_3.metric("Acute Inpatient LOS", f"{los_v:.2f} days", f"{los_d:+.1f}% MoM", delta_color="inverse")
    row1_4.metric("Bed Occupancy Rate", f"{occupancy_v:.1f}%", f"{occupancy_d:+.1f}% MoM", delta_color="inverse")

    row2_1, row2_2, row2_3, row2_4 = st.columns(4)
    row2_1.metric("Active ALC Bed Count", f"{int(alc_v)} beds", f"{alc_d:+.1f}% MoM", delta_color="inverse")
    row2_2.metric("ED Admission Rate", f"{admission_v:.1f}%", f"{admission_d:+.1f}% MoM")
    row2_3.metric("LWBS Patient Rate", f"{lwbs_v:.1f}%", f"{lwbs_d:+.1f}% MoM", delta_color="inverse")
    row2_4.metric("Unplanned 30D Readmit", f"{readm_v:.1f}%", f"{readm_d:+.1f}% MoM", delta_color="inverse")

    st.markdown("---")

    st.subheader("Operational Analysis & Clinical Strata Drilldowns")
    tab_trends, tab_ctas, tab_icd, tab_dq = st.tabs([
        "📈 Historical Throughput Trends", 
        "🎯 Triage Acuity Distribution (CTAS)",
        "🫁 Clinical Diagnostic Profiling (ICD-10)",
        "🔍 Data Governance & Pipeline Integrity"
    ])
    
    with tab_trends:
        st.markdown("#### System Capacity Co-Movement Trends")
        trend_analysis_df = filtered_ops.groupby("Month")[["Wait_Time_Hours", "Bed_Occupancy_Pct", "LWBS_Rate_Pct"]].mean()
        st.line_chart(trend_analysis_df)
        
    with tab_ctas:
        st.markdown(f"#### Triage Performance Stratification ({latest_month.strftime('%B %Y')})")
        st.caption("Analyzing clinical velocity across the Canadian Triage and Acuity Scale (1=Emergent, 5=Non-Urgent)")
        
        ctas_labels = ["CTAS 1 (Resuscitation)", "CTAS 2 (Emergent)", "CTAS 3 (Urgent)", "CTAS 4 (Less Urgent)", "CTAS 5 (Non-Urgent)"]
        
        ctas_vols = [
            curr_df["CTAS_1_Visits"].sum(), curr_df["CTAS_2_Visits"].sum(),
            curr_df["CTAS_3_Visits"].sum(), curr_df["CTAS_4_Visits"].sum(),
            curr_df["CTAS_5_Visits"].sum()
        ]
        ctas_waits = [
            curr_df["CTAS_1_Wait_Hrs"].mean(), curr_df["CTAS_2_Wait_Hrs"].mean(),
            curr_df["CTAS_3_Wait_Hrs"].mean(), curr_df["CTAS_4_Wait_Hrs"].mean(),
            curr_df["CTAS_5_Wait_Hrs"].mean()
        ]
        ctas_adms = [
            curr_df["CTAS_1_Admission_Pct"].mean(), curr_df["CTAS_2_Admission_Pct"].mean(),
            curr_df["CTAS_3_Admission_Pct"].mean(), curr_df["CTAS_4_Admission_Pct"].mean(),
            curr_df["CTAS_5_Admission_Pct"].mean()
        ]
        
        ctas_summary_df = pd.DataFrame({
            "Triage Level": ctas_labels,
            "Monthly Visits Volume": [f"{v:,}" for v in ctas_vols],
            "Mean ED Wait Time (Hours)": [f"{w:.2f}h" for w in ctas_waits],
            "Conversion to Admission Rate": [f"{a:.1f}%" for a in ctas_adms]
        })
        st.dataframe(ctas_summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("##### Clinical Flow Discordance: Volumes vs. Waiting Times")
        chart_df = pd.DataFrame({
            "Acuity Level": ["CTAS 1", "CTAS 2", "CTAS 3", "CTAS 4", "CTAS 5"],
            "Wait Time (Hours)": ctas_waits
        }).set_index("Acuity Level")
        
        st.bar_chart(chart_df["Wait Time (Hours)"])
        st.caption("Notice the frontline gridlock: CTAS 3 and 4 cohorts swallow the longest waiting room times due to heavy system throughput blocks.")

    with tab_icd:
        st.markdown(f"#### Clinical Condition Performance Analysis ({latest_month.strftime('%B %Y')})")
        st.caption("Performance matrix grouped by abstract diagnostic cohorts across Ontario Health parameters")
        
        current_los = curr_df["LOS_Days"].mean()
        current_readm = curr_df["Readmission_Rate_Pct"].mean()
        
        diagnoses = ["F03 Dementia / Cognitive Decline", "I63 Acute Ischemic Stroke", "I50 Chronic Heart Failure (CHF)", "J44 COPD Exacerbation", "N39 Urinary Tract Infection (UTI)"]
        
        icd_los = [current_los * 2.4, current_los * 1.6, current_los * 1.1, current_los * 0.9, current_los * 0.5]
        icd_readm = [current_readm * 1.3, current_readm * 0.9, current_readm * 1.8, current_readm * 1.5, current_readm * 0.4]
        icd_alc_rates = ["54.2%", "38.1%", "12.5%", "9.0%", "1.2%"]
        
        icd_summary_df = pd.DataFrame({
            "ICD-10 Diagnostic Group": diagnoses,
            "Average Length of Stay": [f"{l:.1f} days" for l in icd_los],
            "30-Day Unplanned Readmission Rate": [f"{r:.1f}%" for r in icd_readm],
            "ALC Risk Attrib. Pct": icd_alc_rates
        })
        st.dataframe(icd_summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("##### Resource Allocation Mapping: Length of Stay (LOS) by Inpatient Profile")
        chart_icd_df = pd.DataFrame({
            "Condition": ["Dementia", "Stroke", "CHF", "COPD", "UTI"],
            "LOS (Days)": icd_los
        }).set_index("Condition")
        
        st.bar_chart(chart_icd_df["LOS (Days)"])
        st.caption("Analytical Takeaway: Geriatric and cognitive decline profiles exhibit massive exit length-of-stay extensions, acting as the primary system driver behind active bed blocks.")

    with tab_dq:
        st.markdown(f"#### Data Pipeline Integrity & Governance Audit ({latest_month.strftime('%B %Y')})")
        st.caption("Data completeness, schema validation, and documentation compliance scores for provincial reporting.")

        dq_c1, dq_c2, dq_c3, dq_c4 = st.columns(4)
        dq_c1.metric("Global Data Completeness", "99.4%", "+0.2% vs last mo")
        dq_c2.metric("Uncoded ICD-10 Records", "14 cases", "-8 cases MoM", delta_color="inverse")
        dq_c3.metric("Null Triage (CTAS) Fields", "3 cases", "0 change", delta_color="off")
        dq_c4.metric("Suspected Duplicate OHIP IDs", "0.04%", "-0.01% MoM", delta_color="inverse")

        st.markdown("---")
        st.markdown("##### Hospital Documentation Compliance Matrix")
        
        dq_data = {
            "Facility Identifier": ["Sunnybrook HSC", "Unity Health (St. Michaels)", "North York General", "Scarborough Health Network", "Humber River Health", "Trillium Health Partners"],
            "ICD-10 Coding Lag (Days)": [4.2, 5.1, 3.8, 6.4, 4.0, 5.5],
            "Demographic Completeness (%)": [99.8, 99.4, 99.9, 98.7, 99.5, 99.2],
            "Timestamp Sequence Integrity": ["Pass", "Pass", "Pass", "Warning (2 Cases)", "Pass", "Pass"],
            "Provincial Submission Status": ["Ready", "Ready", "Ready", "Pending Audit Review", "Ready", "Ready"]
        }
        dq_df = pd.DataFrame(dq_data)
        
        if selected_facility != "All Ontario Facilities":
            dq_df = dq_df[dq_df["Facility Identifier"] == selected_facility]
            
        st.dataframe(dq_df, use_container_width=True, hide_index=True)
        
        st.warning("""
        **📋 Data Governance Alert — Backlog Resolution Required:**
        * **Issue:** Scarborough Health Network is exhibiting an elevated ICD-10 coding lag (6.4 days) alongside a timestamp sequence anomaly affecting 2 non-admitted ED records. 
        * **Impact:** Delays local Ontario Health West regional funding reconciliation cycles.
        * **Remediation Plan:** Health Information Management (HIM) has been notified to execute manual data remediation protocols prior to the upcoming monthly provincial submission lock.
        """)

    st.markdown("---")
    st.subheader("Platform Platform Component Map")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.info("**📊 ED Surge Forecasting System**\n\nRuns advanced time-series inferences across historical baseline calendars to project next-month bottlenecks.")
        st.success("**🗺️ Geospatial Health Equity Mapping**\n\nMaps local forward sortation areas against health equity metrics to track access barriers outside clinical settings.")
    with m_col2:
        st.warning("**🛏️ Inpatient ALC Bed Block Analysis**\n\nSurfaces critical patient attributes matching clinical characteristics linked with alternate level of care delays.")
        st.error("**💊 Controlled Substance Audit Engine**\n\nApplies machine learning anomaly detection filters over provider cohorts to protect clinical care standards.")

# ── 📊 CLINICAL FORECASTING SYSTEM ─────────────────────────────────────────────
elif module == "📊 ED Surge Forecasting System":
    st.title("📊 ED Surge Forecasting System")
    st.markdown("Time-Series Forecast Model (Facebook Prophet) · Incorporating Ontario Statutory Holiday Calendars")
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

# ── 🗺️ GEOSPATIAL EQUITY LAYER ─────────────────────────────────────────────────
elif module == "🗺️ Geospatial Health Equity Mapping":
    st.title("🗺️ Geospatial Health Equity Mapping")
    st.markdown("Statistics Canada FSA Boundary Integration (2021) · 260 GTA Zones Analysed")
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

# ── 🛏️ RISK SURVEILLANCE ENGINE ───────────────────────────────────────────────
elif module == "🛏️ Inpatient ALC Bed Block Analysis":
    st.title("🛏️ Inpatient ALC Bed Block Analysis")
    st.markdown("XGBoost Predictive Framework + SHAP Explanations · 8,000 Patient Admission Cohorts")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("ROC-AUC Score",  "0.984")
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
        st.dataframe(beds_data, use_container_width=True, hide_index=True)
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
        st.dataframe(shap_data, use_container_width=True, hide_index=True)
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

# ── 💊 PRESCRIBING AUDIT ENGINE ────────────────────────────────────────────────
elif module == "💊 Controlled Substance Audit Engine":
    st.title("💊 Controlled Substance Audit Engine")
    st.markdown("Isolation Forest Outlier Detection Engine · 2,000 Practitioner Cohorts · Precision/Recall 0.812")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Prescribers Analysed","2,000")
    col2.metric("Anomalies Detected",  "80",  "4.0% rate")
    col3.metric("Model Precision",     "0.812")
    col4.metric("Model Recall",        "0.812")
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
        st.dataframe(anom_df, use_container_width=True, hide_index=True)
    with tab3:
        processed_path = "data/processed/rx_audit_list.csv"
        if os.path.exists(processed_path):
            st.dataframe(pd.read_csv(processed_path).head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Run Notebook 04 to generate the audit list CSV")
        st.markdown("---")
        st.subheader("Practitioner Registry Lookup")
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
            st.dataframe(filtered.head(50), use_container_width=True, hide_index=True)