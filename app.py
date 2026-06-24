import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
import requests
import folium
import sqlite3
from pathlib import Path
from streamlit_folium import st_folium
from prophet import Prophet

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ontario Healthcare Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── HOSPITAL CONFIG ────────────────────────────────────────────────────────────
HOSPITALS_CONFIG = {
    "Sunnybrook HSC":              {"base": 320, "noise": 35, "type": "Academic",  "lat": 43.7228, "lon": -79.3772},
    "Unity Health (St. Michaels)": {"base": 290, "noise": 30, "type": "Academic",  "lat": 43.6524, "lon": -79.3764},
    "North York General":          {"base": 210, "noise": 25, "type": "Community", "lat": 43.7672, "lon": -79.4103},
    "Scarborough Health Network":  {"base": 240, "noise": 28, "type": "Community", "lat": 43.7731, "lon": -79.2586},
    "Humber River Health":         {"base": 195, "noise": 22, "type": "Community", "lat": 43.7569, "lon": -79.5480},
    "Trillium Health Partners":    {"base": 260, "noise": 30, "type": "Community", "lat": 43.5936, "lon": -79.6397},
}

# ── DATA WAREHOUSE UTILITIES ───────────────────────────────────────────────────
def run_warehouse_query(query):
    db_path = Path(__file__).resolve().parent / "database" / "healthcare.db"
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)

# ── CACHED DATA LOADERS ────────────────────────────────────────────────────────
@st.cache_data
def load_fact_operations():
    if os.path.exists("fact_operations.csv"):
        df = pd.read_csv("fact_operations.csv")
        if "CTAS_1_Visits" in df.columns:
            df["Month"] = pd.to_datetime(df["Month"])
            return df
    months = pd.date_range("2025-01-01", "2026-06-01", freq="MS")
    rows = []
    for hospital, cfg in HOSPITALS_CONFIG.items():
        for month in months:
            bed_occ = np.random.uniform(85, 98)
            alc_pts = np.random.uniform(15, 45) + (bed_occ - 85) * 0.5
            total_v = int(cfg["base"] * np.random.uniform(28, 31))
            c1 = int(total_v * 0.015); c2 = int(total_v * 0.15)
            c3 = int(total_v * 0.35);  c4 = int(total_v * 0.35)
            c5 = total_v - (c1 + c2 + c3 + c4)
            wt = 2.5 + (bed_occ - 80) * 0.12 + alc_pts * 0.04
            rows.append({
                "Hospital": hospital, "Month": month,
                "ED_Visits": total_v,
                "Wait_Time_Hours": round(max(1.5, wt), 2),
                "LOS_Days": round(4 + alc_pts * 0.03, 2),
                "Bed_Occupancy_Pct": round(bed_occ, 2),
                "ALC_Patients": int(alc_pts),
                "Admission_Rate_Pct": round(np.random.uniform(12, 20), 2),
                "LWBS_Rate_Pct": round(1.5 + wt * 0.7, 2),
                "Readmission_Rate_Pct": round(np.random.uniform(7, 15), 2),
                "Staffing_Index": int(np.random.choice([95, 100, 105])),
                "CTAS_1_Visits": c1, "CTAS_2_Visits": c2,
                "CTAS_3_Visits": c3, "CTAS_4_Visits": c4, "CTAS_5_Visits": c5,
                "CTAS_1_Wait_Hrs": round(np.random.uniform(0.1, 0.3), 2),
                "CTAS_2_Wait_Hrs": round(1.2 + (bed_occ - 80) * 0.03, 2),
                "CTAS_3_Wait_Hrs": round(wt * 1.15, 2),
                "CTAS_4_Wait_Hrs": round(wt * 1.30, 2),
                "CTAS_5_Wait_Hrs": round(wt * 0.85, 2),
                "CTAS_1_Admission_Pct": 85.0, "CTAS_2_Admission_Pct": 42.0,
                "CTAS_3_Admission_Pct": 18.0, "CTAS_4_Admission_Pct": 4.5,
                "CTAS_5_Admission_Pct": 0.5,
            })
    df = pd.DataFrame(rows)
    df.to_csv("fact_operations.csv", index=False)
    return df


@st.cache_data
def generate_ed_data():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    rows = []
    dow = {0:1.12, 1:1.05, 2:1.00, 3:1.00, 4:1.08, 5:0.92, 6:0.85}
    for hospital, cfg in HOSPITALS_CONFIG.items():
        base, noise = cfg["base"], cfg["noise"]
        n = len(dates)
        visits = np.full(n, float(base))
        for i, d in enumerate(dates):
            visits[i] *= dow[d.dayofweek]
            if d.month in [12, 1, 2]:
                visits[i] *= np.random.uniform(1.15, 1.30)
            elif d.month in [6, 7, 8]:
                visits[i] *= np.random.uniform(0.88, 0.95)
            visits[i] *= (1.03 ** (d.year - 2022))
        visits += np.random.normal(0, noise, n)
        visits = np.maximum(visits, 50).round().astype(int)
        for i, d in enumerate(dates):
            rows.append({"date": d, "hospital": hospital, "type": cfg["type"],
                         "ed_visits": visits[i], "base_capacity": int(base * 1.25)})
    df = pd.DataFrame(rows)
    df["surge_flag"] = (df["ed_visits"] > df["base_capacity"]).astype(int)
    return df


@st.cache_data
def generate_rx_data():
    np.random.seed(42)
    N = 2000
    specialties = ["Emergency Medicine", "Internal Medicine", "Family Medicine",
                   "Orthopedics", "Oncology", "Psychiatry", "General Surgery", "Geriatrics"]
    specialty = np.random.choice(specialties, N, p=[0.15, 0.20, 0.25, 0.10, 0.08, 0.08, 0.08, 0.06])
    hospitals = list(HOSPITALS_CONFIG.keys()) + ["Community Practice"]
    hospital  = np.random.choice(hospitals, N, p=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.28])
    opioid_rate = np.random.normal(12, 4, N).clip(0, 35)
    avg_mme     = np.random.normal(45, 15, N).clip(0, 120)
    benzo_combo = np.random.normal(3, 1.5, N).clip(0, 10)
    polypharm   = np.random.normal(18, 5, N).clip(5, 40)
    pts_pm      = np.random.normal(120, 35, N).clip(10, 300)
    avg_rx      = np.random.normal(3.2, 0.8, N).clip(1, 8)
    n_anom = int(N * 0.04)
    anom_idx = np.random.choice(N, n_anom, replace=False)
    ta = anom_idx[:n_anom//3]; tb = anom_idx[n_anom//3:2*n_anom//3]; tc = anom_idx[2*n_anom//3:]
    opioid_rate[ta] = (opioid_rate[ta] * np.random.uniform(2.5, 4.0, len(ta))).clip(0, 100)
    avg_mme[ta]     = (avg_mme[ta]     * np.random.uniform(2.0, 3.5, len(ta))).clip(0, 500)
    benzo_combo[tb] = (benzo_combo[tb] * np.random.uniform(4.0, 7.0, len(tb))).clip(0, 100)
    polypharm[tb]   = (polypharm[tb]   * np.random.uniform(1.8, 2.5, len(tb))).clip(0, 100)
    pts_pm[tc]      = (pts_pm[tc]      * np.random.uniform(2.5, 4.0, len(tc))).clip(0, 800)
    avg_rx[tc]      = (avg_rx[tc]      * np.random.uniform(1.8, 2.5, len(tc))).clip(0, 20)
    true_anom = np.zeros(N, dtype=int); true_anom[anom_idx] = 1
    return pd.DataFrame({
        "prescriber_id": [f"CPSO-{100000+i}" for i in range(N)],
        "specialty": specialty, "hospital": hospital,
        "opioid_rate_pct": opioid_rate.round(2), "avg_opioid_mme": avg_mme.round(1),
        "benzo_opioid_combo_pct": benzo_combo.round(2), "polypharmacy_pct": polypharm.round(2),
        "patients_per_month": pts_pm.round(0).astype(int),
        "avg_rx_per_patient": avg_rx.round(2), "true_anomaly": true_anom,
    })


@st.cache_data(ttl=3600)
def load_ontario_public_wait_times():
    try:
        url = "https://data.ontario.ca/dataset/wait-times-in-emergency-departments/resource/ed-wait-times.json"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            raw = resp.json()
            if "result" in raw and "records" in raw["result"]:
                records = raw["result"]["records"]
                df = pd.DataFrame(records)
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                st.session_state["real_data_loaded"] = True
                return df, True
    except Exception:
        pass

    np.random.seed(99)
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023",
                "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024",
                "Q1 2025", "Q2 2025"]
    rows = []
    benchmarks = {
        "Sunnybrook HSC":              {"wait_90p": 5.8, "admit_90p": 12.4, "lwbs": 2.1},
        "Unity Health (St. Michaels)": {"wait_90p": 6.1, "admit_90p": 11.8, "lwbs": 2.4},
        "North York General":          {"wait_90p": 4.9, "admit_90p": 9.2,  "lwbs": 1.8},
        "Scarborough Health Network":  {"wait_90p": 7.2, "admit_90p": 14.5, "lwbs": 3.1},
        "Humber River Health":         {"wait_90p": 5.3, "admit_90p": 10.1, "lwbs": 2.0},
        "Trillium Health Partners":    {"wait_90p": 6.4, "admit_90p": 12.0, "lwbs": 2.6},
    }
    for hospital, b in benchmarks.items():
        for q in quarters:
            drift = 1 + (quarters.index(q) - 4) * 0.012
            rows.append({
                "hospital": hospital,
                "quarter": q,
                "90th_pctile_wait_hrs": round(b["wait_90p"] * drift * np.random.uniform(0.95, 1.05), 2),
                "90th_pctile_admit_hrs": round(b["admit_90p"] * drift * np.random.uniform(0.95, 1.05), 2),
                "lwbs_pct": round(b["lwbs"] * np.random.uniform(0.9, 1.1), 2),
                "source": "Synthetic — benchmarked against Ontario Health published ranges",
            })
    return pd.DataFrame(rows), False


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Health Intelligence Platform")
st.sidebar.markdown("---")
module = st.sidebar.radio("Executive Command Suite", [
    "🏠 Executive Operations Center",
    "🔍 Healthcare SQL Warehouse Explorer",
    "📊 ED Surge Forecasting System",
    "🗺️ Geospatial Health Equity Mapping",
    "🛏️ Inpatient ALC Bed Block Analysis",
    "💊 Controlled Substance Audit Engine",
    "📋 Ontario Healthcare Case Studies",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Analytics Architecture**")
st.sidebar.markdown("Prophet · XGBoost · SHAP · Isolation Forest")
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Data Notice:** All patient-level and operational data is synthetically "
    "modelled using non-identifiable provincial distributions. PHIPA compliant. "
    "Where noted, public Ontario Health summary statistics are used."
)
st.sidebar.markdown("Developed by **Aswin Anil Bindu** · Ontario, Canada")


# ══════════════════════════════════════════════════════════════════════════════
# 🏠  EXECUTIVE OPERATIONS CENTER
# ══════════════════════════════════════════════════════════════════════════════
if module == "🏠 Executive Operations Center":
    st.title("🏥 Ontario Health Executive Operations Center")
    st.caption("Strategic Performance Analysis · Throughput Velocity · Alternate Level of Care (ALC) Surveillance")
    st.markdown("---")

    df_ops = load_fact_operations()

    st.sidebar.header("Operations Scoping Filters")
    facility_list = ["All Ontario Facilities"] + list(df_ops["Hospital"].unique())
    selected_facility = st.sidebar.selectbox("Scope Health System", facility_list, key="exec_fac")
    role_view = st.sidebar.selectbox("Operational Profile View", [
        "👑 Strategic Executive (CEO/VP) View",
        "⚙️ Clinical Operations Manager View",
        "🩺 Attending Physician / Clinical View",
    ], key="role_view")

    filtered_ops = df_ops if selected_facility == "All Ontario Facilities" \
        else df_ops[df_ops["Hospital"] == selected_facility]

    latest_month = filtered_ops["Month"].max()
    prev_month   = latest_month - pd.DateOffset(months=1)
    curr_df = filtered_ops[filtered_ops["Month"] == latest_month]
    prev_df = filtered_ops[filtered_ops["Month"] == prev_month]

    st.error(f"📋 **Analyst Briefing Engine — Active Profile: {role_view.split('(')[0].strip()}**")
    ins1, ins2 = st.columns([2, 3])
    with ins1:
        st.markdown(f"""
        **System Observations ({latest_month.strftime('%B %Y')}):**
        * **ED Throughput:** Mean system-wide delays tracking near upper threshold bounds.
        * **LWBS Risk:** Long wait times continue to drive patient flight rates.
        * **Inflow Patterns:** Arrival volumes within acceptable seasonal bands (±5%).
        """)
    with ins2:
        if "Strategic" in role_view:
            st.markdown("""
            **Macro Diagnostic Assessment (CEO/VP):**
            * **System Bottleneck:** Primary challenge is *exit-block* from high ALC inpatient backlogs.
            * **Strategic Directives:** Engage regional LTC facilities for transitional spaces; activate
              bed-escalation protocols when occupancy exceeds 92%; monitor ALC daily.
            """)
        elif "Operations" in role_view:
            st.markdown("""
            **Throughput Optimization (Ops Manager):**
            * **Workflow Assessment:** CTAS 3/4 wait times elevated due to bed blocks on medicine wards.
            * **Ops Directives:** Align shift schedules with peak arrival windows; accelerate
              morning discharge coordination to free inpatient beds before noon.
            """)
        else:
            st.markdown("""
            **Frontline Flow Optimization (Clinical):**
            * **Clinical Performance:** CTAS 1/2 admission rates consistent; geriatric admissions
              showing extended LOS.
            * **Clinical Directives:** Apply standardized order sets early; flag ALC risk patients
              within 24 hours of admission. See ALC Risk Calculator for patient-level scoring.
            """)
    st.markdown("---")

    st.subheader(f"System Performance Grid — {latest_month.strftime('%B %Y')}")

    def kpi(col, agg="mean"):
        c = curr_df[col].agg(agg)
        p = prev_df[col].agg(agg)
        d = ((c - p) / p * 100) if p != 0 else 0
        return c, d

    visits_v, visits_d = kpi("ED_Visits", "sum")
    wait_v, wait_d     = kpi("Wait_Time_Hours")
    los_v, los_d       = kpi("LOS_Days")
    occ_v, occ_d       = kpi("Bed_Occupancy_Pct")
    alc_v, alc_d       = kpi("ALC_Patients", "sum")
    adm_v, adm_d       = kpi("Admission_Rate_Pct")
    lwbs_v, lwbs_d     = kpi("LWBS_Rate_Pct")
    readm_v, readm_d   = kpi("Readmission_Rate_Pct")

    c1, c2, c3, c4 = st.columns(4)
    if "Strategic" in role_view:
        c1.metric("Active ALC Bed Count",   f"{int(alc_v)} beds",    f"{alc_d:+.1f}% MoM",  delta_color="inverse")
        c2.metric("Bed Occupancy Rate",     f"{occ_v:.1f}%",         f"{occ_d:+.1f}% MoM",  delta_color="inverse")
        c3.metric("Acute Inpatient LOS",    f"{los_v:.2f} days",     f"{los_d:+.1f}% MoM",  delta_color="inverse")
        c4.metric("30D Unplanned Readmit",  f"{readm_v:.1f}%",       f"{readm_d:+.1f}% MoM",delta_color="inverse")
    elif "Operations" in role_view:
        c1.metric("Total ED Visits",        f"{int(visits_v):,}",    f"{visits_d:+.1f}% MoM",delta_color="inverse")
        c2.metric("Mean Wait Time",         f"{wait_v:.2f} hrs",     f"{wait_d:+.1f}% MoM",  delta_color="inverse")
        c3.metric("Bed Occupancy Rate",     f"{occ_v:.1f}%",         f"{occ_d:+.1f}% MoM",   delta_color="inverse")
        c4.metric("LWBS Patient Rate",      f"{lwbs_v:.1f}%",        f"{lwbs_d:+.1f}% MoM",  delta_color="inverse")
    else:
        c1.metric("Mean Wait Time",         f"{wait_v:.2f} hrs",     f"{wait_d:+.1f}% MoM",  delta_color="inverse")
        c2.metric("ED Admission Rate",      f"{adm_v:.1f}%",         f"{adm_d:+.1f}% MoM")
        c3.metric("Acute Inpatient LOS",    f"{los_v:.2f} days",     f"{los_d:+.1f}% MoM",   delta_color="inverse")
        c4.metric("30D Unplanned Readmit",  f"{readm_v:.1f}%",       f"{readm_d:+.1f}% MoM", delta_color="inverse")

    alc_share = (alc_v / (occ_v / 100 * 600)) * 100
    if alc_share > 30:
        st.error(
            f"🔴 **System-Wide ALC Alert:** {int(alc_v)} ALC patients represent "
            f"~{alc_share:.0f}% of occupied beds. Bed occupancy at {occ_v:.1f}%. "
            f"Mean wait time is {wait_v:.1f} hrs. **Exit-block is the primary driver of current ED delays.**"
        )
    elif alc_share > 20:
        st.warning(
            f"🟡 **ALC Watch:** {int(alc_v)} ALC patients (~{alc_share:.0f}% of occupied beds). "
            f"Monitor discharge velocity — occupancy at {occ_v:.1f}%."
        )
    else:
        st.success(f"🟢 ALC levels within acceptable range. Bed occupancy {occ_v:.1f}%.")

    st.markdown("---")

    st.subheader("Operational Analysis & Clinical Strata Drilldowns")
    tab_ctas, tab_readm, tab_icd, tab_dq, tab_arch = st.tabs([
        "🎯 Triage Acuity Distribution (CTAS)",
        "🔄 Unplanned Readmission Analytics",
        "🫁 Clinical Diagnostic Profiling (ICD-10)",
        "🔍 Data Governance & Pipeline Integrity",
        "🔐 Data Infrastructure Schema",
    ])

    with tab_ctas:
        st.markdown(f"#### Triage Performance Stratification ({latest_month.strftime('%B %Y')})")
        ctas_labels = ["CTAS 1 (Resuscitation)", "CTAS 2 (Emergent)",
                       "CTAS 3 (Urgent)", "CTAS 4 (Less Urgent)", "CTAS 5 (Non-Urgent)"]
        ctas_vols  = [curr_df[f"CTAS_{i}_Visits"].sum()       for i in range(1, 6)]
        ctas_waits = [curr_df[f"CTAS_{i}_Wait_Hrs"].mean()    for i in range(1, 6)]
        ctas_adms  = [curr_df[f"CTAS_{i}_Admission_Pct"].mean() for i in range(1, 6)]
        st.dataframe(pd.DataFrame({
            "Triage Level": ctas_labels,
            "Monthly Visit Volume": [f"{v:,}" for v in ctas_vols],
            "Mean ED Wait Time (Hrs)": [f"{w:.2f}h" for w in ctas_waits],
            "Conversion to Admission": [f"{a:.1f}%" for a in ctas_adms],
        }), use_container_width=True, hide_index=True)
        st.markdown("##### Clinical Flow Discordance: Wait Time by Acuity Level")
        st.bar_chart(pd.DataFrame({"Wait Time (Hours)": ctas_waits},
                                   index=["CTAS 1","CTAS 2","CTAS 3","CTAS 4","CTAS 5"]))

    with tab_readm:
        st.markdown(f"#### Unplanned Readmission Monitoring ({latest_month.strftime('%B %Y')})")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("30-Day Readmission Baseline", f"{readm_v:.1f}%",          f"{readm_d:+.1f}% MoM",  delta_color="inverse")
        rc2.metric("7-Day Return Velocity",        f"{(readm_v * 0.28):.1f}%", "Early post-discharge risk")
        rc3.metric("Post-Discharge Flight Rate",   f"{(lwbs_v * 0.4):.1f}%",   "Care coordination gaps")
        rc4.metric("Estimated Avoidable Cost",     f"${int(visits_v * 0.18):,}", "System excess expense")

        st.markdown("---")
        st.markdown("##### Facility-Level Readmission Performance")
        hosp_names   = list(HOSPITALS_CONFIG.keys())
        hosp_7d      = [1.8, 2.4, 1.5, 3.1, 1.9, 2.8]
        hosp_30d     = [8.4, 11.2, 7.8, 14.5, 9.2, 12.6]
        hosp_status  = ["✅ Meeting Target", "⚠️ Review Triggered", "✅ Meeting Target",
                        "🔴 Action Required", "✅ Meeting Target", "⚠️ Review Triggered"]
        rd_df = pd.DataFrame({
            "Ontario Facility": hosp_names,
            "7-Day Readmission": [f"{x:.1f}%" for x in hosp_7d],
            "30-Day Readmission": [f"{x:.1f}%" for x in hosp_30d],
            "Provincial Compliance": hosp_status,
        })
        if selected_facility != "All Ontario Facilities":
            rd_df = rd_df[rd_df["Ontario Facility"] == selected_facility]
        st.dataframe(rd_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### Readmission Risk by Patient Age Cohort")
        age_groups = ["Under 18", "18–44", "45–64", "65–74", "75–84", "85+"]
        age_rates  = [readm_v * 0.3, readm_v * 0.5, readm_v * 0.8,
                      readm_v * 1.1, readm_v * 1.5, readm_v * 2.1]
        st.bar_chart(pd.DataFrame({"Readmission Rate (%)": [round(x, 2) for x in age_rates]},
                                   index=age_groups))

    with tab_icd:
        st.markdown(f"#### Clinical Diagnostic Profiling ({latest_month.strftime('%B %Y')})")
        cur_los   = curr_df["LOS_Days"].mean()
        cur_readm = curr_df["Readmission_Rate_Pct"].mean()
        diagnoses   = ["F03 Dementia / Cognitive Decline", "I63 Acute Ischemic Stroke",
                       "I50 Chronic Heart Failure (CHF)", "J44 COPD Exacerbation",
                       "N39 Urinary Tract Infection (UTI)"]
        icd_los     = [cur_los * 2.4, cur_los * 1.6, cur_los * 1.1, cur_los * 0.9, cur_los * 0.5]
        icd_readm   = [cur_readm * 1.3, cur_readm * 0.9, cur_readm * 1.8, cur_readm * 1.5, cur_readm * 0.4]
        icd_alc     = ["54.2%", "38.1%", "12.5%", "9.0%", "1.2%"]
        st.dataframe(pd.DataFrame({
            "ICD-10 Diagnostic Group": diagnoses,
            "Avg Length of Stay": [f"{l:.1f} days" for l in icd_los],
            "30-Day Readmission Rate": [f"{r:.1f}%" for r in icd_readm],
            "ALC Attribution": icd_alc,
        }), use_container_width=True, hide_index=True)
        st.markdown("##### Length of Stay by Diagnostic Profile")
        st.bar_chart(pd.DataFrame({"LOS (Days)": icd_los},
                                   index=["Dementia", "Stroke", "CHF", "COPD", "UTI"]))

    with tab_dq:
        st.markdown(f"#### Data Pipeline Integrity & Governance Audit ({latest_month.strftime('%B %Y')})")
        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Global Data Completeness",    "99.4%", "+0.2% vs last month")
        dq2.metric("Uncoded ICD-10 Records",      "14 cases", "-8 MoM", delta_color="inverse")
        dq3.metric("Null CTAS Fields",            "3 cases",  "0 change", delta_color="off")
        dq4.metric("Duplicate OHIP Identifiers",  "0.04%",    "-0.01% MoM", delta_color="inverse")
        st.markdown("---")
        st.markdown("##### Hospital Documentation Compliance Matrix")
        dq_df = pd.DataFrame({
            "Facility": ["Sunnybrook HSC", "Unity Health (St. Michaels)", "North York General",
                         "Scarborough Health Network", "Humber River Health", "Trillium Health Partners"],
            "ICD-10 Coding Lag (Days)":   [4.2, 5.1, 3.8, 6.4, 4.0, 5.5],
            "Demographic Completeness":   ["99.8%", "99.4%", "99.9%", "98.7%", "99.5%", "99.2%"],
            "Timestamp Integrity":        ["Pass", "Pass", "Pass", "⚠️ Warning (2 cases)", "Pass", "Pass"],
            "Submission Status":          ["✅ Ready", "✅ Ready", "✅ Ready",
                                           "⏳ Pending Audit", "✅ Ready", "✅ Ready"],
        })
        if selected_facility != "All Ontario Facilities":
            dq_df = dq_df[dq_df["Facility"] == selected_facility]
        st.dataframe(dq_df, use_container_width=True, hide_index=True)

    with tab_arch:
        st.markdown("#### Data Infrastructure & SQL Warehouse Schema")
        st.markdown("### 🔀 1. Provincial Administrative Clinical Data Flow")
        st.code("""
[ Patient Arrives at ED ]
           │
           ▼
[ CTAS Triage Evaluated ] ──► Captured in CIHI NACRS Registry
           │
 Is Inpatient Admission Required?
       ├──► NO  ──► [ Discharged ] ──► NACRS Record Closed
       └──► YES ──► [ Transferred to Acute Bed ]
                           │
                           ▼
                  Initiates CIHI DAD Registry
                           │
                  Track Inpatient LOS & ALC Status
                           │
                           ▼
                  Closed Upon Abstract Discharge
        """, language="text")

        st.markdown("---")
        st.markdown("### 🔐 2. PHI De-Identification Matrix (PHIPA Compliant)")
        st.dataframe(pd.DataFrame({
            "Variable Class":            ["Patient Name", "Date of Birth", "OHIP Health Card",
                                          "Residential Location", "Age Metric"],
            "Raw EMR Source (PHI)":      ["John Doe", "1948-11-23", "9876-543-210-XM",
                                          "M5G 1X8 (Downtown Toronto)", "77.58"],
            "De-Identified Layer":       ["MASKED / NULL", "NULL (Suppressed)", "PID-80924 (SHA-256)",
                                          "FSA Region: M5G (Aggregated)", "Age Group: 75–84 (Bucket)"],
        }), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🗄️ 3. Relational Warehouse Layout")
        st.code("""
 [ dim_hospital ]                    [ dim_date ]
 ├── HospitalID (PK)                 ├── DateID (PK)
 ├── HospitalName (Unique)           ├── Date
 ├── Region                          ├── Month
 └── HospitalType                    └── Year
         │                                │
         └───────────────┬────────────────┘
                         ▼
               [ fact_operations ]
               ├── RecordID (PK)
               ├── Hospital (Staging Matching)
               ├── Date
               ├── ED_Visits
               └── Wait_Time_Hours / Bed_Occupancy_Pct / ALC_Patients
        """, language="text")

    st.markdown("---")
    st.subheader("Platform Component Map")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.info("**📊 ED Surge Forecasting**\nProphet time-series — 30-day volume projections with surge threshold alerts.")
        st.success("**🗺️ Geospatial Health Equity**\nInteractive Folium map — 6 GTA hospital sites + FSA equity score layer.")
    with pc2:
        st.warning("**🛏️ ALC Bed Block Analysis**\nXGBoost + SHAP — patient-level ALC risk scoring with live calculator.")
        st.error("**💊 Controlled Substance Audit**\nIsolation Forest — prescribing pattern anomaly detection across 2,000 providers.")


# ══════════════════════════════════════════════════════════════════════════════
# 🔍  HEALTHCARE SQL WAREHOUSE EXPLORER (NEW INTERACTIVE INTERFACE)
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🔍 Healthcare SQL Warehouse Explorer":
    st.title("🔍 Healthcare SQL Warehouse Explorer")
    st.markdown("""
    This interactive sandbox exposes the backend **Ontario Healthcare Data Warehouse** (`healthcare.db`). 
    You can construct analytical queries against performance benchmarks mirroring **NACRS** and **DAD** architectures.
    """)
    st.markdown("---")

    # Sidebar Schema Viewer
    st.sidebar.subheader("Live Warehouse Schema")
    table_choice = st.sidebar.selectbox("Select Table to Inspect Structure:", ["fact_operations", "dim_hospital"])
    
    try:
        schema_df = run_warehouse_query(f"PRAGMA table_info({table_choice});")
        if not schema_df.empty:
            st.sidebar.dataframe(schema_df[['name', 'type']], use_container_width=True, hide_index=True)
        else:
            st.sidebar.warning("Warehouse file not populated yet. Execute builder script.")
    except Exception:
        st.sidebar.error("Database initialization failed.")

    # KPI Dropdown Quick-Templates
    st.subheader("Executive Query Templates")
    query_presets = {
        "Select a KPI template to auto-load into the workspace...": "",
        "1. High-Strain Operational Bottlenecks (Wait Time > 4 hrs)": """SELECT 
    Hospital, 
    ROUND(AVG(Wait_Time_Hours), 2) as Avg_Wait_Hours, 
    ROUND(AVG(LOS_Days), 2) as Avg_Inpatient_LOS_Days 
FROM fact_operations 
GROUP BY Hospital 
HAVING Avg_Wait_Hours > 4.0
ORDER BY Avg_Wait_Hours DESC;""",
        
        "2. Alternate Level of Care (ALC) Impact on Occupancy": """SELECT 
    Date, 
    Hospital, 
    Bed_Occupancy_Pct, 
    ALC_Patients 
FROM fact_operations 
WHERE Bed_Occupancy_Pct > 85.0 
ORDER BY ALC_Patients DESC, Bed_Occupancy_Pct DESC;""",
        
        "3. Attrition Indicators (LWBS vs Wait Times)": """SELECT 
    Hospital, 
    ROUND(AVG(Wait_Time_Hours), 2) as Avg_Wait_Hours, 
    ROUND(AVG(LWBS_Rate_Pct), 2) as Avg_Left_Without_Being_Seen_Pct 
FROM fact_operations 
GROUP BY Hospital 
ORDER BY Avg_Left_Without_Being_Seen_Pct DESC;""",

        "4. System Capacity Cross-Join (Fact + Dimension Join)": """SELECT 
    h.HospitalType,
    h.Region,
    ROUND(AVG(f.Bed_Occupancy_Pct), 2) as Avg_Bed_Occupancy
FROM fact_operations f
JOIN dim_hospital h ON f.Hospital = h.HospitalName
GROUP BY h.HospitalType, h.Region;"""
    }

    selected_preset = st.selectbox("Load Standard Healthcare Template Query:", list(query_presets.keys()))
    
    if selected_preset != "Select a KPI template to auto-load into the workspace...":
        default_query = query_presets[selected_preset]
    else:
        default_query = """-- Custom SQLite Analytics Playground
SELECT Hospital, ROUND(AVG(Wait_Time_Hours), 2) AS AvgWait 
FROM fact_operations 
GROUP BY Hospital;"""

    # Query Input Terminal
    query_input = st.text_area("SQL Command Workspace (SQLite Dialect)", value=default_query, height=180)
    
    if st.button("Execute Analytics Query", type="primary"):
        if query_input.strip():
            try:
                with st.spinner("Processing execution plan..."):
                    results_df = run_warehouse_query(query_input)
                
                if not results_df.empty:
                    st.success(f"Execution successful: Returned {len(results_df)} records")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Chart Output Generation Engine
                    if len(results_df.columns) >= 2:
                        st.markdown("##### Instant Query Visualization")
                        x_axis = results_df.columns[0]
                        y_axis = results_df.columns[1]
                        
                        try:
                            results_df[y_axis] = pd.to_numeric(results_df[y_axis])
                            st.bar_chart(data=results_df, x=x_axis, y=y_axis)
                        except Exception:
                            st.info("Query output schema is categorical; chart visual generation bypassed.")
                else:
                    st.warning("Query executed cleanly but returned 0 rows or database is empty.")
                        
            except Exception as e:
                st.error(f"SQL Engine Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 📊  ED SURGE FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif module == "📊 ED Surge Forecasting System":
    st.title("📊 ED Surge Forecasting System")
    st.markdown("Facebook Prophet · Ontario Statutory Holiday Regressors · 30-Day Look-Ahead")
    st.markdown("---")

    wt_df, is_real = load_ontario_public_wait_times()
    if is_real:
        st.success("✅ **Live Ontario Health data loaded.** Displaying published ED wait time indicators.")
    else:
        st.info(
            "📊 **Ontario Health Quarterly Benchmarks** (synthetic, benchmarked against published ranges). "
            "Source: Ontario Health Wait Time reporting methodology."
        )

    st.markdown("##### 90th Percentile Wait Time Trend by Hospital")
    if "quarter" in wt_df.columns:
        pivot = wt_df.pivot_table(index="quarter", columns="hospital", values="90th_pctile_wait_hrs")
        st.line_chart(pivot)
        
    st.markdown("---")

    hospital_choice = st.selectbox("Select Hospital for Surge Forecast", list(HOSPITALS_CONFIG.keys()))
    forecast_days   = st.slider("Forecast Horizon (days)", 7, 60, 30)

    with st.spinner(f"Training Prophet model for {hospital_choice}..."):
        df_ed = generate_ed_data()
        hdf = df_ed[df_ed["hospital"] == hospital_choice][["date", "ed_visits"]].copy()
        hdf.columns = ["ds", "y"]
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Surge Threshold",       f"{threshold} visits/day")
    c2.metric("Predicted Surge Days",  f"{len(surge_days)} / {forecast_days}")
    risk = "🔴 HIGH" if len(surge_days) > 10 else "🟡 MEDIUM" if len(surge_days) > 4 else "🟢 LOW"
    c3.metric("Risk Level", risk)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(hdf.tail(90)["ds"], hdf.tail(90)["y"], color="#1f77b4", s=12, alpha=0.5, label="Actual")
    ax.plot(forecast_only["ds"], forecast_only["yhat"], color="#ff7f0e", lw=2, label="Forecast")
    ax.fill_between(forecast_only["ds"], forecast_only["yhat_lower"], forecast_only["yhat_upper"],
                    alpha=0.2, color="#ff7f0e", label="90% CI")
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
        st.warning(
            f"⚠️ {len(surge_days)} surge days predicted. "
            f"First: {surge_days['ds'].min().strftime('%B %d, %Y')}. "
        )


# ══════════════════════════════════════════════════════════════════════════════
# 🗺️  GEOSPATIAL HEALTH EQUITY
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🗺️ Geospatial Health Equity Mapping":
    st.title("🗺️ Geospatial Health Equity Mapping")
    st.markdown("Interactive GTA Hospital Network Map · FSA Equity Score Layer · 6 Active Facilities")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("FSAs Analysed",        "260")
    c2.metric("Lowest Equity Score",  "13.3 / 100", "Scarborough M1N")
    c3.metric("Highest Equity Score", "95.0 / 100", "North York")
    st.markdown("---")

    equity_data = {
        "Sunnybrook HSC":              {"equity": 72.4, "wait_90p": 5.8,  "alc_pct": 28.3, "beds": 376, "color": "blue"},
        "Unity Health (St. Michaels)": {"equity": 68.1, "wait_90p": 6.1,  "alc_pct": 31.2, "beds": 341, "color": "blue"},
        "North York General":          {"equity": 95.0, "wait_90p": 4.9,  "alc_pct": 21.4, "beds": 257, "color": "green"},
        "Scarborough Health Network":  {"equity": 13.3, "wait_90p": 7.2,  "alc_pct": 41.8, "beds": 293, "color": "red"},
        "Humber River Health":         {"equity": 58.7, "wait_90p": 5.3,  "alc_pct": 25.6, "beds": 239, "color": "orange"},
        "Trillium Health Partners":    {"equity": 61.2, "wait_90p": 6.4,  "alc_pct": 33.1, "beds": 310, "color": "orange"},
    }

    map_filter = st.selectbox("Filter by Metric Layer", [
        "Equity Score", "90th Pct Wait Time (hrs)", "ALC Rate (%)"
    ])

    m_map = folium.Map(location=[43.70, -79.42], zoom_start=11, tiles="CartoDB positron")

    for hospital, cfg in HOSPITALS_CONFIG.items():
        eq = equity_data[hospital]
        color = eq["color"]
        if map_filter == "Equity Score":
            val_str = f"Equity Score: {eq['equity']} / 100"
        elif map_filter == "90th Pct Wait Time (hrs)":
            val_str = f"90th Pct Wait: {eq['wait_90p']} hrs"
        else:
            val_str = f"ALC Rate: {eq['alc_pct']}%"

        popup_html = f"""
        <b>{hospital}</b><br>
        Type: {cfg['type']}<br>
        {val_str}<br>
        Equity Score: {eq['equity']} / 100<br>
        ALC Rate: {eq['alc_pct']}%<br>
        90th Pct Wait: {eq['wait_90p']} hrs<br>
        Licensed Beds: {eq['beds']}
        """
        folium.CircleMarker(
            location=[cfg["lat"], cfg["lon"]],
            radius=14 + (1 - eq["equity"] / 100) * 20,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{hospital} — Equity: {eq['equity']}/100",
        ).add_to(m_map)

    st_folium(m_map, width=1000, height=520)

    st.markdown("---")
    st.markdown("##### Facility Equity & Access Summary")
    eq_df = pd.DataFrame([{
        "Hospital": h,
        "Equity Score": f"{d['equity']} / 100",
        "90th Pct Wait (hrs)": d["wait_90p"],
        "ALC Rate (%)": d["alc_pct"],
        "Licensed Beds": d["beds"],
        "Access Status": "🔴 Critical" if d["equity"] < 30 else "🟡 Moderate" if d["equity"] < 65 else "🟢 Good",
    } for h, d in equity_data.items()])
    st.dataframe(eq_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 🛏️  INPATIENT ALC BED BLOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif module == "🛏️ Inpatient ALC Bed Block Analysis":
    st.title("🛏️ Inpatient ALC Bed Block Analysis")
    st.markdown("XGBoost Binary Classifier + SHAP TreeExplainer · 8,000 Patient Cohorts")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC Score",  "0.984")
    c2.metric("Avg Precision",  "0.998")
    c3.metric("Beds Blocked",   "333", "across 6 hospitals")
    c4.metric("Top Risk Factor","Age", "SHAP 2.66")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🤖 Model Performance", "🔍 SHAP Explainability"])

    with tab1:
        fpath = os.path.join("reports", "alc_distribution.png")
        if os.path.exists(fpath):
            st.image(fpath, use_column_width=True)
        st.subheader("Beds Blocked by Hospital")
        beds_df = pd.DataFrame({
            "Hospital":     list(HOSPITALS_CONFIG.keys()),
            "Beds Blocked": [63, 63, 57, 52, 51, 47],
            "ALC Rate (%)": [80.8, 87.5, 83.8, 85.2, 77.3, 85.5],
        }).sort_values("Beds Blocked", ascending=False)
        st.dataframe(beds_df, use_container_width=True, hide_index=True)

    with tab2:
        fpath = os.path.join("reports", "alc_model_performance.png")
        if os.path.exists(fpath):
            st.image(fpath, use_column_width=True)
        else:
            st.info("Run Notebook 03 to generate model performance charts.")
            st.dataframe(pd.DataFrame({
                "Metric": ["ROC-AUC", "Average Precision", "Sensitivity (Recall)", "Specificity"],
                "Score":  ["0.984",   "0.998",             "0.961",                "0.947"],
                "Benchmark": ["CIHI target >0.75", "> 0.95 excellent", "> 0.90 target", "> 0.90 target"],
            }), use_container_width=True, hide_index=True)

    with tab3:
        fpath = os.path.join("reports", "alc_shap_explainability.png")
        if os.path.exists(fpath):
            st.image(fpath, use_column_width=True)
        st.dataframe(pd.DataFrame({
            "Rank":    [1, 2, 3, 4, 5],
            "Feature": ["Age", "Cognitive Impairment", "Has Caregiver", "Lives Alone", "Diagnosis Category"],
            "SHAP Value": [2.6561, 1.4064, 0.9888, 0.8703, 0.7848],
            "Direction": ["↑ risk with age", "↑ risk if present", "↓ risk if present",
                          "↑ risk if alone", "↑ risk: dementia/stroke"],
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🧮 Live ALC Risk Calculator")

    ca, cb, cc = st.columns(3)
    with ca:
        p_age  = st.slider("Patient Age", 18, 100, 78)
        p_diag = st.selectbox("Primary Diagnosis", [
            "Hip Fracture", "Stroke", "Dementia", "COPD", "CHF",
            "Pneumonia", "UTI", "Sepsis", "Elective Surgery", "Other"])
        p_los  = st.slider("Current / Expected LOS (days)", 1, 60, 12)
    with cb:
        p_alone = st.checkbox("Lives Alone",          value=True)
        p_carer = st.checkbox("Has Caregiver",        value=False)
        p_cogn  = st.checkbox("Cognitive Impairment", value=True)
    with cc:
        p_charlson = st.slider("Charlson Comorbidity Index", 0, 11, 4)
        p_func     = st.slider("Functional Score (0–100)", 0, 100, 35)
        p_prior    = st.slider("Prior Admissions (12 months)", 0, 10, 2)

    diag_risk = {
        "Dementia": 0.25, "Stroke": 0.20, "Hip Fracture": 0.18, "COPD": 0.05,
        "CHF": 0.04, "Pneumonia": 0.03, "UTI": 0.02, "Sepsis": 0.06,
        "Elective Surgery": 0.01, "Other": 0.0,
    }
    risk_score = float(np.clip(
        0.02 + 0.008 * (p_age - 18) + diag_risk[p_diag]
        + 0.10 * p_alone + 0.15 * p_cogn - 0.12 * p_carer
        + 0.008 * p_charlson - 0.001 * p_func
        + 0.010 * p_los + 0.03 * p_prior, 0, 1))
    risk_pct = risk_score * 100

    st.markdown("---")
    if risk_pct >= 60:
        st.error(f"🔴 HIGH ALC RISK — {risk_pct:.1f}%")
        st.markdown("""
        **Recommended Actions:**
        * Initiate social work referral **today**
        * Begin LTC / convalescent care placement screening within 24 hours
        """)
    elif risk_pct >= 35:
        st.warning(f"🟡 MEDIUM ALC RISK — {risk_pct:.1f}%")
    else:
        st.success(f"🟢 LOW ALC RISK — {risk_pct:.1f}%")
    st.progress(min(risk_score, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 💊  CONTROLLED SUBSTANCE AUDIT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif module == "💊 Controlled Substance Audit Engine":
    st.title("💊 Controlled Substance Audit Engine")
    st.markdown("Isolation Forest Outlier Detection · 2,000 Prescriber Cohorts · Precision/Recall 0.812")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prescribers Analysed", "2,000")
    c2.metric("Anomalies Detected",   "80",    "4.0% rate")
    c3.metric("Model Precision",      "0.812")
    c4.metric("Model Recall",         "0.812")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Prescribing Patterns", "🔍 Anomaly Detection", "📋 Audit List"])

    with tab1:
        fpath = os.path.join("reports", "rx_prescribing_patterns.png")
        if os.path.exists(fpath):
            st.image(fpath, use_column_width=True)
        else:
            st.info("Run Notebook 04 to generate prescribing pattern charts.")
            df_rx = generate_rx_data()
            st.markdown("##### Opioid Rate Distribution by Specialty")
            spec_avg = df_rx.groupby("specialty")["opioid_rate_pct"].mean().sort_values(ascending=False)
            st.bar_chart(spec_avg)

    with tab2:
        cc1, cc2 = st.columns(2)
        for col, fname, title in [
            (cc1, "rx_anomaly_detection.png",   "PCA + Anomaly Score"),
            (cc2, "rx_opioid_risk_quadrant.png", "Opioid Risk Quadrant"),
        ]:
            fpath = os.path.join("reports", fname)
            if os.path.exists(fpath):
                col.image(fpath, caption=title, use_column_width=True)
        st.dataframe(pd.DataFrame({
            "Anomaly Type":     ["Opioid Over-Prescriber", "Volume Outlier", "High-Risk Combinations", "Other Pattern"],
            "Count":            [22, 20, 18, 20],
            "Share":            ["27.5%", "25.0%", "22.5%", "25.0%"],
            "Remediation Path": ["CPSO Referral", "Billing Audit", "Pharmacist Alert", "Manual Review"],
        }), use_container_width=True, hide_index=True)

    with tab3:
        processed = "data/processed/rx_audit_list.csv"
        if os.path.exists(processed):
            st.dataframe(pd.read_csv(processed).head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Run Notebook 04 to generate the audit list CSV.")
        st.markdown("---")
        st.subheader("Practitioner Registry Lookup")
        df_rx = generate_rx_data()
        spec_f = st.multiselect("Filter by Specialty", df_rx["specialty"].unique().tolist(),
                                default=["Emergency Medicine", "Family Medicine"])
        hosp_f = st.multiselect("Filter by Hospital", df_rx["hospital"].unique().tolist(),
                                default=list(HOSPITALS_CONFIG.keys())[:2])
        if spec_f and hosp_f:
            filtered = df_rx[
                df_rx["specialty"].isin(spec_f) & df_rx["hospital"].isin(hosp_f)
            ][["prescriber_id", "specialty", "hospital",
               "opioid_rate_pct", "avg_opioid_mme", "patients_per_month"]]
            st.dataframe(filtered.head(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 📋  ONTARIO HEALTHCARE ANALYST CASE STUDIES
# ══════════════════════════════════════════════════════════════════════════════
elif module == "📋 Ontario Healthcare Case Studies":
    st.title("📋 Ontario Healthcare Analyst Case Studies")
    st.caption("Three structured analyst scenarios — Problem → Analysis → Recommendation.")
    st.markdown("---")

    df_ops = load_fact_operations()
    latest = df_ops["Month"].max()
    curr   = df_ops[df_ops["Month"] == latest]
    prev   = df_ops[df_ops["Month"] == (latest - pd.DateOffset(months=4))]

    wait_now  = curr["Wait_Time_Hours"].mean()
    wait_then = prev["Wait_Time_Hours"].mean()
    wait_chg  = ((wait_now - wait_then) / wait_then * 100)
    alc_now   = curr["ALC_Patients"].sum()
    alc_then  = prev["ALC_Patients"].sum()
    alc_chg   = ((alc_now - alc_then) / alc_then * 100)
    occ_now   = curr["Bed_Occupancy_Pct"].mean()

    with st.expander("📌 Case Study 1 — Why Are ED Wait Times Increasing?", expanded=True):
        col_p, col_a, col_r = st.columns(3)
        with col_p:
            st.markdown("### 🔴 Problem")
            st.markdown(f"System-wide mean ED wait time has increased **{wait_chg:.0f}%** over the past 4 months.")
        with col_a:
            st.markdown("### 🔍 Analysis")
            st.markdown(f"ALC patient volume increased **{alc_chg:.0f}%** over the same period.")
        with col_r:
            st.markdown("### ✅ Recommendation")
            st.markdown("Root cause is **exit-block, not front-door volume.**")

        st.markdown("---")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Wait Time Change (4-month)", f"{wait_chg:+.0f}%", delta_color="inverse")
        kpi2.metric("ALC Volume Change (4-month)", f"{alc_chg:+.0f}%", delta_color="inverse")
        kpi3.metric("Current Bed Occupancy", f"{occ_now:.1f}%", delta_color="inverse")

    with st.expander("📌 Case Study 2 — Which Patients Drive the Longest LOS?"):
        cur_los = curr["LOS_Days"].mean()
        col_p, col_a, col_r = st.columns(3)
        with col_p:
            st.markdown("### 🔴 Problem")
            st.markdown(f"Acute inpatient LOS is trending upward (current mean: **{cur_los:.1f} days**).")
        with col_a:
            st.markdown("### 🔍 Analysis")
            st.markdown(f"F03 Dementia patients average **{cur_los * 2.4:.1f} days** — 2.4× the system mean.")
        with col_r:
            st.markdown("### ✅ Recommendation")
            st.markdown("Flag **dementia and stroke admissions** for social work review early.")

        st.markdown("---")
        diagnoses = ["Dementia", "Stroke", "CHF", "COPD", "UTI"]
        los_vals  = [cur_los * 2.4, cur_los * 1.6, cur_los * 1.1, cur_los * 0.9, cur_los * 0.5]
        st.bar_chart(pd.DataFrame({"Avg LOS (Days)": los_vals}, index=diagnoses))

    with st.expander("📌 Case Study 3 — Which Facility Has the Highest Readmission Risk?"):
        col_p, col_a, col_r = st.columns(3)
        with col_p:
            st.markdown("### 🔴 Problem")
            st.markdown("30-day unplanned readmission rates vary significantly across the network.")
        with col_a:
            st.markdown("### 🔍 Analysis")
            st.markdown("Scarborough equity score is **13.3/100** — lowest in the network.")
        with col_r:
            st.markdown("### ✅ Recommendation")
            st.markdown("Implement structured discharge checklists and post-discharge phone follow-ups.")

        st.markdown("---")
        hosp_names = list(HOSPITALS_CONFIG.keys())
        hosp_30d   = [8.4, 11.2, 7.8, 14.5, 9.2, 12.6]
        hosp_status = ["✅ Meeting Target", "⚠️ Review Triggered", "✅ Meeting Target",
                       "🔴 Action Required", "✅ Meeting Target", "⚠️ Review Triggered"]
        st.dataframe(pd.DataFrame({
            "Hospital": hosp_names,
            "30-Day Readmission Rate": [f"{x:.1f}%" for x in hosp_30d],
            "Status": hosp_status,
        }), use_container_width=True, hide_index=True)