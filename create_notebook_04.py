import nbformat

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell("""# 🏥 Ontario ED Intelligence — Notebook 04
## Module 4: Prescription Anomaly Detector

**Goal:** Detect unusual prescribing patterns in Ontario hospitals that may indicate:
- Opioid overprescribing (a major Ontario public health crisis)
- Drug-drug interaction risks
- Outlier prescribers vs. peer benchmarks
- Possible data entry errors or fraud

**Why this matters:**
- Ontario's opioid crisis costs the healthcare system $3.5B+ annually
- CPSO (College of Physicians and Surgeons of Ontario) actively monitors outlier prescribers
- IQVIA, Ontario Pharmacare, and ODB data are used for exactly this analysis

**Model:** Isolation Forest (unsupervised anomaly detection) + statistical outlier scoring"""))

cells.append(nbformat.v4.new_markdown_cell("## 0️⃣ Import Libraries"))

cells.append(nbformat.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

print('✅ Libraries loaded')
print(f'   pandas  : {pd.__version__}')
print(f'   sklearn : ', end='')
import sklearn; print(sklearn.__version__)"""))

cells.append(nbformat.v4.new_markdown_cell("""## 1️⃣ Generate Realistic Prescriber Dataset
> Based on real Ontario Drug Benefit (ODB) program patterns:
> - ~2,000 prescribers across GTA hospitals + community
> - Metrics tracked: opioid MME, polypharmacy rates, high-risk combos
> - ~3-5% expected anomaly rate (matches real-world audits)"""))

cells.append(nbformat.v4.new_code_cell("""N_PRESCRIBERS = 2000

specialties = ['Emergency Medicine', 'Internal Medicine', 'Family Medicine',
               'Orthopedics', 'Oncology', 'Psychiatry', 'General Surgery', 'Geriatrics']

specialty = np.random.choice(specialties, size=N_PRESCRIBERS,
    p=[0.15, 0.20, 0.25, 0.10, 0.08, 0.08, 0.08, 0.06])

hospitals = ['Sunnybrook HSC', 'Unity Health', 'North York General',
             'Scarborough Health Network', 'Humber River Health',
             'Trillium Health Partners', 'Community Practice']

hospital = np.random.choice(hospitals, size=N_PRESCRIBERS,
    p=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.28])

years_practice = np.random.randint(1, 40, N_PRESCRIBERS)

# Normal prescribing metrics
avg_rx_per_patient     = np.random.normal(3.2, 0.8, N_PRESCRIBERS).clip(1, 8)
opioid_rate_pct        = np.random.normal(12, 4, N_PRESCRIBERS).clip(0, 35)
avg_opioid_mme         = np.random.normal(45, 15, N_PRESCRIBERS).clip(0, 120)
high_dose_opioid_pct   = np.random.normal(5, 2, N_PRESCRIBERS).clip(0, 15)
benzo_opioid_combo_pct = np.random.normal(3, 1.5, N_PRESCRIBERS).clip(0, 10)
polypharmacy_pct       = np.random.normal(18, 5, N_PRESCRIBERS).clip(5, 40)
avg_days_supply        = np.random.normal(14, 4, N_PRESCRIBERS).clip(1, 30)
patients_per_month     = np.random.normal(120, 35, N_PRESCRIBERS).clip(10, 300)
generic_prescribe_pct  = np.random.normal(72, 10, N_PRESCRIBERS).clip(30, 99)
refill_rate_pct        = np.random.normal(35, 8, N_PRESCRIBERS).clip(10, 65)

# Inject anomalies (~4% of prescribers)
n_anomalies = int(N_PRESCRIBERS * 0.04)
anomaly_idx = np.random.choice(N_PRESCRIBERS, n_anomalies, replace=False)

# Type A: Opioid over-prescribers
type_a = anomaly_idx[:n_anomalies//3]
opioid_rate_pct[type_a]        *= np.random.uniform(2.5, 4.0, len(type_a))
avg_opioid_mme[type_a]         *= np.random.uniform(2.0, 3.5, len(type_a))
high_dose_opioid_pct[type_a]   *= np.random.uniform(3.0, 5.0, len(type_a))

# Type B: Dangerous combo prescribers
type_b = anomaly_idx[n_anomalies//3:2*n_anomalies//3]
benzo_opioid_combo_pct[type_b] *= np.random.uniform(4.0, 7.0, len(type_b))
polypharmacy_pct[type_b]       *= np.random.uniform(1.8, 2.5, len(type_b))

# Type C: Volume outliers
type_c = anomaly_idx[2*n_anomalies//3:]
patients_per_month[type_c]     *= np.random.uniform(2.5, 4.0, len(type_c))
avg_rx_per_patient[type_c]     *= np.random.uniform(1.8, 2.5, len(type_c))

# Clip all values to realistic ranges
opioid_rate_pct        = opioid_rate_pct.clip(0, 100)
avg_opioid_mme         = avg_opioid_mme.clip(0, 500)
high_dose_opioid_pct   = high_dose_opioid_pct.clip(0, 100)
benzo_opioid_combo_pct = benzo_opioid_combo_pct.clip(0, 100)
polypharmacy_pct       = polypharmacy_pct.clip(0, 100)
patients_per_month     = patients_per_month.clip(1, 800)
avg_rx_per_patient     = avg_rx_per_patient.clip(1, 20)

true_anomaly = np.zeros(N_PRESCRIBERS, dtype=int)
true_anomaly[anomaly_idx] = 1

df = pd.DataFrame({
    'prescriber_id':          [f'CPSO-{100000+i}' for i in range(N_PRESCRIBERS)],
    'specialty':               specialty,
    'hospital':                hospital,
    'years_practice':          years_practice,
    'avg_rx_per_patient':      avg_rx_per_patient.round(2),
    'opioid_rate_pct':         opioid_rate_pct.round(2),
    'avg_opioid_mme':          avg_opioid_mme.round(1),
    'high_dose_opioid_pct':    high_dose_opioid_pct.round(2),
    'benzo_opioid_combo_pct':  benzo_opioid_combo_pct.round(2),
    'polypharmacy_pct':        polypharmacy_pct.round(2),
    'avg_days_supply':         avg_days_supply.round(1),
    'patients_per_month':      patients_per_month.round(0).astype(int),
    'generic_prescribe_pct':   generic_prescribe_pct.round(2),
    'refill_rate_pct':         refill_rate_pct.round(2),
    'true_anomaly':            true_anomaly
})

print(f'✅ Prescriber dataset generated')
print(f'   Total prescribers    : {len(df):,}')
print(f'   Injected anomalies   : {true_anomaly.sum()} ({true_anomaly.mean()*100:.1f}%)')
print(f'   Specialties          : {df.specialty.nunique()}')
print(f'   Hospitals/settings   : {df.hospital.nunique()}')
df.head()"""))

cells.append(nbformat.v4.new_markdown_cell("## 2️⃣ Prescribing Pattern Overview by Specialty"))

cells.append(nbformat.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()

metrics = [
    ('opioid_rate_pct',       'Opioid Prescribing Rate (%)',      '#e41a1c'),
    ('avg_opioid_mme',        'Avg Opioid MME per Prescription',  '#ff7f00'),
    ('benzo_opioid_combo_pct','Benzo + Opioid Combo Rate (%)',    '#984ea3'),
    ('polypharmacy_pct',      'Polypharmacy Rate (%)',            '#377eb8'),
    ('patients_per_month',    'Patients per Month',               '#4daf4a'),
    ('generic_prescribe_pct', 'Generic Prescribing Rate (%)',     '#a65628'),
]

for idx, (col, label, colour) in enumerate(metrics):
    spec_avg = df.groupby('specialty')[col].mean().sort_values(ascending=True)
    bars = axes[idx].barh(spec_avg.index, spec_avg.values, color=colour, alpha=0.8)
    axes[idx].set_xlabel(label)
    axes[idx].set_title(label, fontsize=11, fontweight='bold')
    axes[idx].axvline(x=spec_avg.mean(), color='grey', linestyle='--', linewidth=1)
    for bar, val in zip(bars, spec_avg.values):
        axes[idx].text(val + spec_avg.max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=8)
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.suptitle('Ontario Prescribing Patterns by Specialty\\nGTA Hospitals + Community',
              fontsize=15, fontweight='bold')
plt.tight_layout()
os.makedirs('../reports', exist_ok=True)
plt.savefig('../reports/rx_prescribing_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Saved → reports/rx_prescribing_patterns.png')"""))

cells.append(nbformat.v4.new_markdown_cell("## 3️⃣ Train Isolation Forest Anomaly Detector"))

cells.append(nbformat.v4.new_code_cell("""feature_cols = [
    'avg_rx_per_patient', 'opioid_rate_pct', 'avg_opioid_mme',
    'high_dose_opioid_pct', 'benzo_opioid_combo_pct', 'polypharmacy_pct',
    'avg_days_supply', 'patients_per_month', 'generic_prescribe_pct',
    'refill_rate_pct'
]

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(
    n_estimators=300,
    contamination=0.04,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)

df['anomaly_pred']  = iso_forest.fit_predict(X_scaled)
df['anomaly_score'] = -iso_forest.score_samples(X_scaled)
df['is_anomaly']    = (df['anomaly_pred'] == -1).astype(int)

# Anomaly type classification
def classify_anomaly(row):
    if row['is_anomaly'] == 0:
        return 'Normal'
    if row['opioid_rate_pct'] > 35 or row['avg_opioid_mme'] > 120:
        return 'Opioid Over-Prescriber'
    if row['benzo_opioid_combo_pct'] > 15 or row['polypharmacy_pct'] > 40:
        return 'High-Risk Combinations'
    if row['patients_per_month'] > 250 or row['avg_rx_per_patient'] > 7:
        return 'Volume Outlier'
    return 'Other Anomaly'

df['anomaly_type'] = df.apply(classify_anomaly, axis=1)

detected   = df['is_anomaly'].sum()
true_anom  = df['true_anomaly'].sum()
true_pos   = ((df['is_anomaly']==1) & (df['true_anomaly']==1)).sum()
precision  = true_pos / detected if detected > 0 else 0
recall     = true_pos / true_anom if true_anom > 0 else 0

print(f'✅ Isolation Forest trained')
print(f'   Prescribers flagged  : {detected} ({detected/len(df)*100:.1f}%)')
print(f'   True anomalies       : {true_anom}')
print(f'   True positives       : {true_pos}')
print(f'   Precision            : {precision:.3f}')
print(f'   Recall               : {recall:.3f}')
print()
print('  Anomaly types detected:')
for atype, count in df[df['is_anomaly']==1]['anomaly_type'].value_counts().items():
    print(f'    {atype:<30} : {count}')"""))

cells.append(nbformat.v4.new_markdown_cell("## 4️⃣ Anomaly Visualisation — PCA + Score Distribution"))

cells.append(nbformat.v4.new_code_cell("""pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# PCA scatter — Normal vs Anomaly
colour_map = {
    'Normal':                  '#2ca02c',
    'Opioid Over-Prescriber':  '#d62728',
    'High-Risk Combinations':  '#ff7f0e',
    'Volume Outlier':          '#9467bd',
    'Other Anomaly':           '#8c564b'
}

for atype, colour in colour_map.items():
    mask = df['anomaly_type'] == atype
    size = 80 if atype != 'Normal' else 15
    alpha = 0.9 if atype != 'Normal' else 0.3
    axes[0].scatter(df.loc[mask,'pca1'], df.loc[mask,'pca2'],
                     c=colour, s=size, alpha=alpha, label=atype, zorder=3 if atype != 'Normal' else 1)

axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[0].set_title('PCA — Prescriber Anomaly Detection\\n(Coloured by Anomaly Type)',
                   fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.2)
axes[0].set_facecolor('#fafafa')

# Anomaly score distribution
axes[1].hist(df[df['is_anomaly']==0]['anomaly_score'], bins=50,
              alpha=0.6, color='#2ca02c', label='Normal', density=True)
axes[1].hist(df[df['is_anomaly']==1]['anomaly_score'], bins=50,
              alpha=0.8, color='#d62728', label='Anomaly', density=True)
threshold = df[df['is_anomaly']==1]['anomaly_score'].min()
axes[1].axvline(x=threshold, color='black', linestyle='--',
                 linewidth=2, label=f'Detection threshold')
axes[1].set_xlabel('Anomaly Score (higher = more anomalous)')
axes[1].set_ylabel('Density')
axes[1].set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Anomaly rate by hospital
hosp_anomaly = df.groupby('hospital')['is_anomaly'].mean().sort_values(ascending=True) * 100
colours_h = ['#d62728' if v > 5 else '#ff7f0e' if v > 3 else '#2ca02c' for v in hosp_anomaly.values]
bars = axes[2].barh(hosp_anomaly.index, hosp_anomaly.values, color=colours_h)
axes[2].axvline(x=4, color='grey', linestyle='--', linewidth=1.5, label='Expected rate (4%)')
axes[2].set_xlabel('Anomaly Rate (%)')
axes[2].set_title('Anomaly Rate by Hospital/Setting', fontsize=12, fontweight='bold')
axes[2].legend()
for bar, val in zip(bars, hosp_anomaly.values):
    axes[2].text(val + 0.05, bar.get_y() + bar.get_height()/2,
                  f'{val:.1f}%', va='center', fontsize=10)
axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle('Ontario Prescription Anomaly Detector\\nIsolation Forest Results',
              fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/rx_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Saved → reports/rx_anomaly_detection.png')"""))

cells.append(nbformat.v4.new_markdown_cell("## 5️⃣ Top Flagged Prescribers — Audit List"))

cells.append(nbformat.v4.new_code_cell("""top_anomalies = (df[df['is_anomaly']==1]
    .sort_values('anomaly_score', ascending=False)
    .head(20)[['prescriber_id','specialty','hospital','anomaly_type','anomaly_score',
               'opioid_rate_pct','avg_opioid_mme','benzo_opioid_combo_pct',
               'patients_per_month']]
    .reset_index(drop=True)
)

print('TOP 20 FLAGGED PRESCRIBERS — AUDIT PRIORITY LIST')
print('=' * 90)
print(top_anomalies.to_string(index=False))

top_anomalies.to_csv('../data/processed/rx_audit_list.csv', index=False)
print()
print('✅ Saved → data/processed/rx_audit_list.csv')"""))

cells.append(nbformat.v4.new_markdown_cell("## 6️⃣ Opioid MME vs Prescribing Rate — Risk Quadrant"))

cells.append(nbformat.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(14, 10))

normal = df[df['is_anomaly']==0]
anomalies = df[df['is_anomaly']==1]

ax.scatter(normal['opioid_rate_pct'], normal['avg_opioid_mme'],
            c='#aec7e8', s=20, alpha=0.4, label='Normal prescribers', zorder=1)

anom_colours = {
    'Opioid Over-Prescriber': '#d62728',
    'High-Risk Combinations': '#ff7f0e',
    'Volume Outlier':         '#9467bd',
    'Other Anomaly':          '#8c564b'
}

for atype, colour in anom_colours.items():
    mask = anomalies['anomaly_type'] == atype
    if mask.sum() > 0:
        ax.scatter(anomalies.loc[mask,'opioid_rate_pct'],
                    anomalies.loc[mask,'avg_opioid_mme'],
                    c=colour, s=120, alpha=0.9, label=atype,
                    edgecolors='black', linewidth=0.5, zorder=3)

# Risk quadrant lines
opioid_mean = df['opioid_rate_pct'].mean()
mme_mean    = df['avg_opioid_mme'].mean()
ax.axvline(x=opioid_mean, color='grey', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=mme_mean, color='grey', linestyle='--', linewidth=1, alpha=0.7)

ax.text(opioid_mean * 1.8, mme_mean * 1.8, '🔴 HIGH RISK\\nHigh Rate + High MME',
         fontsize=11, fontweight='bold', color='#d62728',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.8))
ax.text(2, mme_mean * 1.8, '🟡 MONITOR\\nLow Rate + High MME',
         fontsize=10, color='#ff7f0e',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0e0', alpha=0.8))
ax.text(opioid_mean * 1.8, 5, '🟡 MONITOR\\nHigh Rate + Low MME',
         fontsize=10, color='#ff7f0e',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0e0', alpha=0.8))
ax.text(2, 5, '🟢 NORMAL\\nLow Rate + Low MME',
         fontsize=10, color='#2ca02c',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0ffe0', alpha=0.8))

ax.set_xlabel('Opioid Prescribing Rate (%)', fontsize=12)
ax.set_ylabel('Average Opioid MME per Prescription', fontsize=12)
ax.set_title('Opioid Risk Quadrant Analysis\\nGTA Prescribers — Ontario ED Intelligence Platform',
              fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_facecolor('#fafafa')

plt.tight_layout()
plt.savefig('../reports/rx_opioid_risk_quadrant.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Saved → reports/rx_opioid_risk_quadrant.png')"""))

cells.append(nbformat.v4.new_markdown_cell("## 7️⃣ Final Summary"))

cells.append(nbformat.v4.new_code_cell("""print('=' * 65)
print('  PRESCRIPTION ANOMALY DETECTOR — FINAL SUMMARY')
print('=' * 65)
print(f'  Prescribers analysed     : {len(df):,}')
print(f'  Anomalies detected       : {df.is_anomaly.sum()} ({df.is_anomaly.mean()*100:.1f}%)')
print(f'  Precision                : {precision:.3f}')
print(f'  Recall                   : {recall:.3f}')
print()
print('  Anomaly Breakdown:')
for atype, count in df[df['is_anomaly']==1]['anomaly_type'].value_counts().items():
    pct = count / df.is_anomaly.sum() * 100
    print(f'    {atype:<30} : {count:>3} ({pct:.1f}%)')
print()
print('  Highest Risk Hospital/Setting:')
top_hosp = df.groupby("hospital")["is_anomaly"].mean().idxmax()
top_rate = df.groupby("hospital")["is_anomaly"].mean().max() * 100
print(f'    {top_hosp} — {top_rate:.1f}% anomaly rate')
print()
print('=' * 65)
print()
print('📁 ALL MODULE OUTPUTS:')
print('   Module 2 — GTA Equity Heatmap')
print('      reports/gta_fsa_base_map.png')
print('      reports/gta_equity_heatmap.png')
print('   Module 1 — ED Surge Forecaster')
print('      reports/ed_trends_by_hospital.png')
print('      reports/ed_seasonality_patterns.png')
print('      reports/sunnybrook_forecast.png')
print('      reports/gta_surge_dashboard.png')
print('   Module 3 — ALC Bed Block Analyzer')
print('      reports/alc_distribution.png')
print('      reports/alc_model_performance.png')
print('      reports/alc_shap_explainability.png')
print('      reports/alc_beds_blocked_dashboard.png')
print('   Module 4 — Rx Anomaly Detector')
print('      reports/rx_prescribing_patterns.png')
print('      reports/rx_anomaly_detection.png')
print('      reports/rx_opioid_risk_quadrant.png')
print()
print('🎉 ALL 4 MODULES COMPLETE!')
print('🚀 Next → Streamlit Dashboard + FastAPI')"""))

nb.cells = cells

with open('notebooks/04_Rx_Anomaly_Detector.ipynb', 'w') as f:
    nbformat.write(nb, f)

print('✅ Notebook 04 created successfully!')
