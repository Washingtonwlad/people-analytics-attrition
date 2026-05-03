import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# --- Configuración de página ---
st.set_page_config(
    page_title="People Analytics — Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Estilos globales ---
sns.set_theme(style='whitegrid', palette='muted', font_scale=0.85)
plt.rcParams['figure.dpi'] = 100

# --- Carga y procesamiento de datos ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('../data/raw/HR-Employee-Attrition.csv')
    df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
    df['AttritionBinary'] = (df['Attrition'] == 'Yes').astype(int)
    df['OverTimeBinary'] = (df['OverTime'] == 'Yes').astype(int)
    return df

# --- Clustering ---
@st.cache_resource
def run_clustering(df):
    clustering_vars = [
        'JobSatisfaction', 'EnvironmentSatisfaction',
        'RelationshipSatisfaction', 'JobInvolvement',
        'WorkLifeBalance', 'OverTimeBinary'
    ]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[clustering_vars])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    profile_names = {
        0: 'Strained but Present',
        1: 'Structurally Stable, Relationally Distant',
        2: 'Overloaded at Risk',
        3: 'Engaged and Balanced'
    }
    df['Profile'] = df['Cluster'].map(profile_names)
    return df, scaler, kmeans

# --- Modelo predictivo ---
@st.cache_resource
def train_model(df):
    df_model = pd.get_dummies(df.drop(columns=['Attrition', 'Profile']), drop_first=True)
    X = df_model.drop(columns=['AttritionBinary'])
    y = df_model['AttritionBinary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X, X_test

@st.cache_resource
def compute_shap_values(_explainer, X_full):
    return _explainer.shap_values(X_full)

# --- Ejecutar todo ---
df = load_and_prepare_data()
df, scaler, kmeans = run_clustering(df)
model, explainer, X, X_test = train_model(df)

# ============================================================
# HEADER
# ============================================================
st.title("📊 People Analytics: Behavioral Risk Profiling")
st.markdown(
    "**Dataset:** IBM HR Analytics · 1,470 employees · 35 variables  \n"
    "**Model:** XGBoost + SHAP · AUC-ROC: 0.763  \n"
    "**Segmentation:** K-Means behavioral clustering (k=4)"
)
st.divider()

# ============================================================
# SECCIÓN 1 — MÉTRICAS GLOBALES
# ============================================================
st.header("🏢 Workforce Overview")

total = len(df)
attrition_rate = df['AttritionBinary'].mean() * 100
high_risk = (df['Profile'] == 'Overloaded at Risk').sum()
stable = (df['Profile'] == 'Engaged and Balanced').sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Employees", f"{total:,}")
with col2:
    st.metric("Overall Attrition Rate", f"{attrition_rate:.1f}%",
              delta="-vs 20% industry avg", delta_color="normal")
with col3:
    st.metric("Overloaded at Risk", f"{high_risk:,}",
              delta=f"{high_risk/total*100:.1f}% of workforce",
              delta_color="inverse")
with col4:
    st.metric("Engaged & Balanced", f"{stable:,}",
              delta=f"{stable/total*100:.1f}% of workforce",
              delta_color="normal")

st.divider()

# ============================================================
# SECCIÓN 1 — GRÁFICOS DE OVERVIEW
# ============================================================
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Attrition Rate by Behavioral Profile")

    profile_order = [
        'Engaged and Balanced',
        'Structurally Stable, Relationally Distant',
        'Strained but Present',
        'Overloaded at Risk'
    ]
    attrition_by_profile = (
        df.groupby('Profile')['AttritionBinary']
        .mean() * 100
    ).reindex(profile_order)

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(attrition_by_profile.index,
                   attrition_by_profile.values,
                   color=colors, edgecolor='white', height=0.5)
    ax.axvline(x=16.1, color='gray', linestyle='--',
               linewidth=1.5, label='Global baseline (16.1%)')
    ax.set_xlabel('Attrition rate (%)')
    ax.set_xlim(0, 40)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, attrition_by_profile.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontweight='bold', fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Employees per Profile")

    count_by_profile = df['Profile'].value_counts().reindex(profile_order)
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    ax2.barh(count_by_profile.index, count_by_profile.values,
             color=colors, edgecolor='white', height=0.5)
    ax2.set_xlabel('Number of employees')
    for i, val in enumerate(count_by_profile.values):
        ax2.text(val + 3, i, str(val), va='center', fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

# ============================================================
# SECCIÓN 2 — EXPLORADOR DE PERFILES
# ============================================================
st.header("🔍 Behavioral Profile Explorer")

profile_colors = {
    'Engaged and Balanced': '#2ecc71',
    'Structurally Stable, Relationally Distant': '#3498db',
    'Strained but Present': '#f39c12',
    'Overloaded at Risk': '#e74c3c'
}

profile_descriptions = {
    'Engaged and Balanced': (
        "The most stable profile. Employees in this group report high relationship "
        "satisfaction and work-life balance, with zero overtime. They represent the "
        "organizational ideal and the lowest attrition risk (8.5%). Retention strategy: "
        "maintain current conditions and use as cultural benchmark."
    ),
    'Structurally Stable, Relationally Distant': (
        "Low attrition (10.6%) despite the lowest relationship satisfaction in the dataset. "
        "Zero overtime and good work-life balance compensate for interpersonal distance. "
        "Vulnerable if structural conditions deteriorate. Intervention: relationship-building "
        "initiatives without disrupting workload balance."
    ),
    'Strained but Present': (
        "Average attrition rate (16.2%) with the lowest perceived work-life balance (1.71/4) "
        "despite minimal overtime. The source of strain is not captured in formal workload "
        "metrics — suggesting informal pressure or unmeasured demands. "
        "Intervention: qualitative investigation of strain sources."
    ),
    'Overloaded at Risk': (
        "Highest attrition rate (29.7%) — nearly double the global baseline. "
        "Defined almost entirely by 100% overtime prevalence. Satisfaction and engagement "
        "scores are similar to other profiles; this group does not leave because they "
        "dislike their work — they leave because of structural overload. "
        "Intervention: workload reduction and overtime policy review."
    )
}

selected_profile = st.selectbox(
    "Select a behavioral profile:",
    options=list(profile_colors.keys())
)

profile_df = df[df['Profile'] == selected_profile]
profile_color = profile_colors[selected_profile]

col_info, col_radar = st.columns([2, 3])

with col_info:
    st.markdown(f"### {selected_profile}")
    st.markdown(f"**Attrition rate:** {profile_df['AttritionBinary'].mean()*100:.1f}%")
    st.markdown(f"**Employees:** {len(profile_df):,} ({len(profile_df)/len(df)*100:.1f}% of workforce)")
    st.markdown("---")
    st.markdown(profile_descriptions[selected_profile])

    st.markdown("**Key metrics:**")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Avg Monthly Income",
                  f"${profile_df['MonthlyIncome'].mean():,.0f}")
        st.metric("Avg Age",
                  f"{profile_df['Age'].mean():.1f} yrs")
    with m2:
        st.metric("Overtime %",
                  f"{profile_df['OverTimeBinary'].mean()*100:.0f}%")
        st.metric("Avg Tenure",
                  f"{profile_df['YearsAtCompany'].mean():.1f} yrs")

with col_radar:
    clustering_vars = [
        'JobSatisfaction', 'EnvironmentSatisfaction',
        'RelationshipSatisfaction', 'JobInvolvement',
        'WorkLifeBalance', 'OverTimeBinary'
    ]
    labels = [
        'Job\nSatisfaction', 'Environment\nSatisfaction',
        'Relationship\nSatisfaction', 'Job\nInvolvement',
        'Work-Life\nBalance', 'OverTime'
    ]

    values = profile_df[clustering_vars].mean().values.tolist()
    max_vals = [4, 4, 4, 4, 4, 1]
    values_norm = [v / m for v, m in zip(values, max_vals)]
    values_norm += values_norm[:1]

    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax3.plot(angles, values_norm, 'o-', linewidth=2, color=profile_color)
    ax3.fill(angles, values_norm, alpha=0.25, color=profile_color)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7)
    ax3.set_title(f'Behavioral Profile — {selected_profile}',
                  fontweight='bold', pad=20, fontsize=9)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()

st.divider()

# ============================================================
# SECCIÓN 3 — INDIVIDUAL RISK EXPLORER
# ============================================================
st.header("👤 Individual Risk Explorer")
st.markdown("Select an employee to see their predicted attrition probability and SHAP explanation.")

# Preparar probabilidades para todo el dataset
df_model_full = pd.get_dummies(
    df.drop(columns=['Attrition', 'Profile']), drop_first=True
)
X_full = df_model_full.drop(columns=['AttritionBinary'])
X_full = X_full.reindex(columns=X.columns, fill_value=0)
all_probs = model.predict_proba(X_full)[:, 1]
df['AttritionProb'] = all_probs

# Selector de empleado
employee_idx = st.slider(
    "Select employee index:",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

employee = df.iloc[employee_idx]
emp_prob = employee['AttritionProb']
emp_profile = employee['Profile']
emp_actual = employee['Attrition']
emp_color = profile_colors[emp_profile]

# Layout: métricas izquierda, waterfall derecha
col_emp, col_waterfall = st.columns([1, 2])

with col_emp:
    st.markdown(f"### Employee #{employee_idx}")

    # Barra de riesgo visual
    risk_label = (
        "🔴 High Risk" if emp_prob >= 0.5
        else "🟡 Moderate Risk" if emp_prob >= 0.25
        else "🟢 Low Risk"
    )
    st.markdown(f"**Attrition Probability:** {emp_prob*100:.1f}%")
    st.progress(float(emp_prob))
    st.markdown(f"**Risk Level:** {risk_label}")
    st.markdown(f"**Actual Outcome:** {'Left ❌' if emp_actual == 'Yes' else 'Stayed ✅'}")
    st.markdown(f"**Behavioral Profile:** :{emp_profile}")
    st.divider()

    # Perfil del empleado
    st.markdown("**Employee characteristics:**")
    chars = {
        'Age': int(employee['Age']),
        'Department': employee['Department'],
        'Job Role': employee['JobRole'],
        'Monthly Income': f"${employee['MonthlyIncome']:,}",
        'Overtime': employee['OverTime'],
        'Years at Company': int(employee['YearsAtCompany']),
        'Work-Life Balance': f"{int(employee['WorkLifeBalance'])}/4",
        'Job Satisfaction': f"{int(employee['JobSatisfaction'])}/4",
    }
    for key, val in chars.items():
        st.markdown(f"- **{key}:** {val}")

with col_waterfall:
    st.markdown("### SHAP Explanation")
    st.markdown("Variables pushing toward attrition **(red)** or retention **(blue)**")

    shap_values_full = compute_shap_values(explainer, X_full)
    shap_exp = shap.Explanation(
        values=shap_values_full[employee_idx],
        base_values=explainer.expected_value,
        data=X_full.iloc[employee_idx].values,
        feature_names=X_full.columns.tolist()
    )

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    plt.sca(ax4)
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    ax4.set_title(
        f'Employee #{employee_idx} — Risk: {emp_prob*100:.1f}% | '
        f'Actual: {emp_actual}',
        fontweight='bold', fontsize=9, pad=10
    )
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.divider()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    ---
    **People Analytics: Behavioral Risk Profiling & Attrition Prediction**  
    Washington Casamen Nolasco · Psychologist & Behavioral Data Scientist  
    Dataset: IBM HR Analytics (synthetic, public domain) · Model: XGBoost + SHAP  
    [GitHub](https://github.com/Washingtonwlad/people-analytics-attrition)
    """
)