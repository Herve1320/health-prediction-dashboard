import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from clinical_pipeline import PIPELINE_STAGES, assess_patient, load_pipeline_models

# ============================================================
# APPLICATION CONSTANTS
# ============================================================

APP_TITLE = "Hypertension Risk Screening Dashboard"
APP_SUBTITLE = "Interconnected Clinical Risk Support Pipeline"
DASHBOARD_VERSION = "1.1.0"
MODEL_VERSION = "v2.0"
LAST_UPDATED = "June 2026"

OVERALL_SCORE_MAX = 19.1

INTENDED_USE = (
    "This dashboard provides model-based risk screening support for blood-pressure-related risk patterns. "
    "It is intended for demonstration, research, and analytical review workflows."
)

SAFETY_NOTICE = (
    "This system is not intended to diagnose, treat, prevent, or replace professional medical judgment. "
    "Outputs should be reviewed by a qualified professional before any clinical decision."
)

PRIVACY_NOTICE = (
    "Do not enter identifiable real patient information in the public demo. Use synthetic, sample, or approved "
    "de-identified data only."
)

# ============================================================
# CONFIG
# ============================================================

try:
    from db_config import DASHBOARD_PASSWORD, APP_MODE
except Exception:
    DASHBOARD_PASSWORD = ""
    APP_MODE = "auto"

# ============================================================
# DATABASE / CLOUD FALLBACK
# ============================================================

try:
    from db_service import (
        fetch_assessment_steps,
        fetch_model_registry,
        fetch_patients,
        fetch_research_history,
        fetch_research_summary,
        save_research_assessment,
        test_connection,
    )
except Exception:
    def test_connection():
        return False, "Database service unavailable. Running in CSV/cloud mode."

    def fetch_patients():
        csv_path = "generated_data/ml_dataset.csv"
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def fetch_model_registry():
        return pd.DataFrame()

    def fetch_research_history(limit=100):
        return pd.DataFrame()

    def fetch_research_summary():
        return {}

    def fetch_assessment_steps(assessment_id):
        return pd.DataFrame()

    def save_research_assessment(*args, **kwargs):
        return None


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fb;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
        padding: 26px;
        border-radius: 18px;
        border: 1px solid #dde6f2;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    .risk-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border-left: 8px solid #007bff;
        margin-bottom: 16px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }

    .info-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #dde6f2;
        margin-bottom: 14px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.04);
    }

    .metric-box {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dde6f2;
    }

    .small-muted {
        color: #6c757d;
        font-size: 14px;
    }

    .section-label {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    .pill {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        background:#eef4ff;
        border:1px solid #d6e6ff;
        color:#1f5fbf;
        font-size:13px;
        margin-right:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LABELS, WEIGHTS, AND EXPLANATIONS
# ============================================================

LABELS = {
    "BP_Stage": {
        0: ("Normal", "#28a745"),
        1: ("Elevated", "#ffc107"),
        2: ("Hypertension Stage 1", "#fd7e14"),
        3: ("Hypertension Stage 2", "#dc3545"),
    },
    "Health_Risk_Tier": {
        0: ("Low Tier", "#28a745"),
        1: ("Moderate Tier", "#fd7e14"),
        2: ("Critical Tier", "#dc3545"),
    },
}

BINARY_STAGES = {
    "Chronic_Hypertension_Development",
    "Hypertensive_Event",
    "Hypertensive_Crisis_Risk",
    "BP_Medication_Recommendation",
    "Cardiovascular_Event_Risk",
    "Stroke_Risk",
    "Heart_Attack_Risk",
    "Emergency_Visit_Risk",
}

# Must match compute_overall_risk() inside clinical_pipeline.py
OVERALL_SCORE_WEIGHTS = {
    "BP_Stage": "BP stage × 0.8",
    "Health_Risk_Tier": "Health tier × 0.6",
    "Hypertensive_Crisis_Risk": 3.0,
    "Emergency_Visit_Risk": 3.0,
    "Stroke_Risk": 2.5,
    "Heart_Attack_Risk": 2.5,
    "Chronic_Hypertension_Development": 1.5,
    "Cardiovascular_Event_Risk": 1.5,
    "Hypertensive_Event": 1.0,
    "BP_Medication_Recommendation": 0.5,
}

HIGH_SEVERITY_STAGES = {
    "Hypertensive_Crisis_Risk": 3.0,
    "Emergency_Visit_Risk": 3.0,
    "Stroke_Risk": 2.5,
    "Heart_Attack_Risk": 2.5,
    "Chronic_Hypertension_Development": 1.5,
    "Cardiovascular_Event_Risk": 1.5,
    "Hypertensive_Event": 1.0,
    "BP_Medication_Recommendation": 0.5,
}

STAGE_FRIENDLY_NAMES = {
    "BP_Stage": "Blood Pressure Stage",
    "Health_Risk_Tier": "Health Risk Tier",
    "Future_Systolic": "Projected Systolic BP",
    "Future_Diastolic": "Projected Diastolic BP",
    "Chronic_Hypertension_Development": "Chronic Hypertension Pattern",
    "Hypertensive_Event": "Hypertensive Pattern Flag",
    "Hypertensive_Crisis_Risk": "Hypertensive Crisis Risk Signal",
    "BP_Medication_Recommendation": "Medication Review Signal",
    "Cardiovascular_Event_Risk": "Cardiovascular Strain Signal",
    "Stroke_Risk": "Stroke Risk Signal",
    "Heart_Attack_Risk": "Cardiac Event Risk Signal",
    "Emergency_Visit_Risk": "Emergency-Care Risk Signal",
    "Risk_Score": "Internal Hemodynamic Risk Index",
    "Probability_Hypertension": "Hypertension Association Probability",
}

STAGE_EXPLANATIONS = {
    "BP_Stage": (
        "Classifies the blood pressure pattern from average systolic and diastolic values. "
        "This is the first stage and influences many downstream outputs."
    ),
    "Health_Risk_Tier": (
        "Summarizes general risk using the blood pressure stage and blood pressure volatility. "
        "Higher volatility suggests readings are less stable over time."
    ),
    "Future_Systolic": (
        "Projects a future systolic blood pressure value based on the current vitals and early pipeline state. "
        "It is a model estimate, not a measured blood pressure value."
    ),
    "Future_Diastolic": (
        "Projects a future diastolic blood pressure value using the current vitals and upstream pipeline outputs."
    ),
    "Chronic_Hypertension_Development": (
        "Flags whether the profile resembles a longer-term hypertension pattern based on stage, volatility, and risk tier."
    ),
    "Hypertensive_Event": (
        "Flags whether the current profile resembles an active hypertensive pattern."
    ),
    "Hypertensive_Crisis_Risk": (
        "Flags a severe blood-pressure-related risk signal using blood pressure stage and projected systolic pressure."
    ),
    "BP_Medication_Recommendation": (
        "Flags a need for medication review. This does not prescribe medication and must be interpreted by a qualified professional."
    ),
    "Cardiovascular_Event_Risk": (
        "Flags cardiovascular strain based on blood pressure stage, pulse pressure, and volatility."
    ),
    "Stroke_Risk": (
        "Combines chronic hypertension pattern, crisis signal, and cardiovascular strain signal to estimate stroke-related risk."
    ),
    "Heart_Attack_Risk": (
        "Combines cardiovascular strain, blood pressure stage, and crisis signal to estimate cardiac event risk."
    ),
    "Emergency_Visit_Risk": (
        "Estimates whether the profile is associated with higher emergency-care risk."
    ),
    "Risk_Score": (
        "An internal hemodynamic index used by the pipeline. It is not the final overall score and should not be interpreted as a clinical scale."
    ),
    "Probability_Hypertension": (
        "A probability-like model output between 0 and 1 estimating association with hypertension pattern."
    ),
}

RISK_SCALE = pd.DataFrame(
    [
        {"Level": "Low", "Score Range": "0.0 – 1.9", "Meaning": "Few or no major risk signals activated."},
        {"Level": "Moderate", "Score Range": "2.0 – 3.9", "Meaning": "Early warning signals are present."},
        {"Level": "High", "Score Range": "4.0 – 6.9", "Meaning": "Multiple important risk signals are active."},
        {"Level": "Critical", "Score Range": "7.0+", "Meaning": "Several high-severity risk signals are active together."},
    ]
)


# ============================================================
# AUTHENTICATION
# ============================================================

def check_access():
    if not DASHBOARD_PASSWORD:
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title(f"{APP_TITLE} — Sign in")
    st.caption("This dashboard is password-protected.")

    pwd = st.text_input("Password", type="password")

    if st.button("Enter"):
        if pwd == DASHBOARD_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")

    return False


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_resource
def load_assets():
    return load_pipeline_models()


@st.cache_data(ttl=60)
def load_patients_from_db():
    return fetch_patients()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_cloud_mode():
    return str(APP_MODE).lower() in ("cloud", "csv", "streamlit")


def format_stage_output(stage_name, value):
    if stage_name in LABELS:
        label, color = LABELS[stage_name].get(int(value), ("Unknown", "#6c757d"))
        return label, color

    if stage_name in BINARY_STAGES or stage_name == "Overall_Risk_Flag":
        try:
            value_int = int(value)
        except Exception:
            value_int = 1 if value is True else 0

        if value_int == 1:
            return "Risk Signal Active", "#dc3545"
        return "Not Flagged", "#28a745"

    if stage_name == "Probability_Hypertension":
        try:
            return f"{float(value):.1%}", "#007bff"
        except Exception:
            return str(value), "#007bff"

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.1f}", "#007bff"

    return str(value), "#007bff"


def build_patient_row(age, gender, systolic, diastolic, volatility, pulse, readings, region_id=1):
    gen_val = 0 if gender == "Male" else 1

    return pd.Series({
        "Age": age,
        "Gender": gen_val,
        "RegionID": region_id,
        "Avg_Systolic": systolic,
        "Avg_Diastolic": diastolic,
        "BP_Volatility": volatility,
        "Pulse_Pressure": pulse,
        "Reading_Count": readings,
    }), gender


def get_overall_interpretation(level):
    if level == "Low":
        return "Low risk: few or no major risk signals were activated."
    if level == "Moderate":
        return "Moderate risk: early warning signals are present, but severe indicators are limited."
    if level == "High":
        return "High risk: multiple clinically important risk signals were activated."
    return "Critical risk: several high-severity risk signals were activated together."


def get_triggered_risk_factors(preds):
    triggered = []

    for name, weight in HIGH_SEVERITY_STAGES.items():
        value = preds.get(name, 0)

        try:
            is_triggered = int(value) == 1
        except Exception:
            is_triggered = value is True

        if is_triggered:
            triggered.append({
                "stage": name,
                "label": STAGE_FRIENDLY_NAMES.get(name, name.replace("_", " ")),
                "weight": weight,
                "explanation": STAGE_EXPLANATIONS.get(name, ""),
            })

    return triggered


def explain_score_contribution(preds):
    rows = []

    bp_stage = float(preds.get("BP_Stage", 0))
    health_tier = float(preds.get("Health_Risk_Tier", 0))

    rows.append({
        "Component": "Blood Pressure Stage",
        "Output": bp_stage,
        "Weight Rule": "stage × 0.8",
        "Contribution": round(bp_stage * 0.8, 2),
    })

    rows.append({
        "Component": "Health Risk Tier",
        "Output": health_tier,
        "Weight Rule": "tier × 0.6",
        "Contribution": round(health_tier * 0.6, 2),
    })

    for item in get_triggered_risk_factors(preds):
        rows.append({
            "Component": item["label"],
            "Output": "Active",
            "Weight Rule": f"+{item['weight']}",
            "Contribution": item["weight"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Contribution", ascending=False, ignore_index=True)

    return df


def summarize_patient_inputs(patient):
    return {
        "Age": int(patient.get("Age")),
        "Gender": "Male" if int(patient.get("Gender", 0)) == 0 else "Female",
        "Avg Systolic": float(patient.get("Avg_Systolic")),
        "Avg Diastolic": float(patient.get("Avg_Diastolic")),
        "BP Volatility": float(patient.get("BP_Volatility")),
        "Pulse Pressure": float(patient.get("Pulse_Pressure")),
        "Reading Count": int(patient.get("Reading_Count")),
    }


def flag_input_drivers(patient):
    drivers = []

    sys_val = float(patient.get("Avg_Systolic"))
    dia_val = float(patient.get("Avg_Diastolic"))
    vol_val = float(patient.get("BP_Volatility"))
    pulse_val = float(patient.get("Pulse_Pressure"))
    age_val = int(patient.get("Age"))

    if sys_val >= 140:
        drivers.append(("Average systolic BP", sys_val, "High systolic pattern"))
    elif sys_val >= 130:
        drivers.append(("Average systolic BP", sys_val, "Stage 1 range signal"))
    elif sys_val >= 120:
        drivers.append(("Average systolic BP", sys_val, "Elevated range signal"))

    if dia_val >= 90:
        drivers.append(("Average diastolic BP", dia_val, "High diastolic pattern"))
    elif dia_val >= 80:
        drivers.append(("Average diastolic BP", dia_val, "Elevated diastolic signal"))

    if vol_val >= 15:
        drivers.append(("BP volatility", vol_val, "High variation in readings"))
    elif vol_val >= 10:
        drivers.append(("BP volatility", vol_val, "Moderate variation in readings"))

    if pulse_val >= 60:
        drivers.append(("Pulse pressure", pulse_val, "Elevated hemodynamic strain signal"))
    elif pulse_val >= 50:
        drivers.append(("Pulse pressure", pulse_val, "Moderate hemodynamic strain signal"))

    if age_val >= 65:
        drivers.append(("Age", age_val, "Older age group in risk profile"))

    return pd.DataFrame(drivers, columns=["Input Factor", "Value", "Why it matters"])


def render_status_pills():
    data_mode = "Cloud demo dataset" if is_cloud_mode() else "Local SQL / CSV fallback"
    st.markdown(
        f"""
        <span class="pill">Dashboard {DASHBOARD_VERSION}</span>
        <span class="pill">Model {MODEL_VERSION}</span>
        <span class="pill">{data_mode}</span>
        <span class="pill">Last updated: {LAST_UPDATED}</span>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RESULT RENDERING
# ============================================================

def render_assessment_result(result, patient_row=None):
    overall = result["overall"]
    preds = result["predictions"]

    level = overall["Overall_Risk_Level"]
    score = overall["Overall_Risk_Score"]

    level_colors = {
        "Low": "#28a745",
        "Moderate": "#ffc107",
        "High": "#fd7e14",
        "Critical": "#dc3545",
    }

    interpretation = get_overall_interpretation(level)
    triggered = get_triggered_risk_factors(preds)
    contribution_df = explain_score_contribution(preds)

    st.markdown(
        f"""
        <div class="risk-card" style="border-left-color:{level_colors.get(level, '#007bff')};">
            <div class="section-label">Composite screening output</div>
            <h2 style="margin:0;color:{level_colors.get(level, '#007bff')};">
                Overall Risk Level: {level}
            </h2>
            <p style="font-size:17px;margin:8px 0 0 0;">
                Composite Pipeline Risk Score: <b>{score}</b> / {OVERALL_SCORE_MAX}
            </p>
            <p style="font-size:15px;margin:8px 0 0 0;">
                <b>Interpretation:</b> {interpretation}
            </p>
            <p style="font-size:13px;margin:8px 0 0 0;color:#6c757d;">
                This score is an internal weighted pipeline score. It is not a blood pressure measurement,
                a diagnosis, or a treatment recommendation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Overall flag", "Review Needed" if overall["Overall_Risk_Flag"] else "Clear")
    oc2.metric("Composite score", f"{score} / {OVERALL_SCORE_MAX}")
    oc3.metric("Active risk signals", len(triggered))
    oc4.metric("Stages evaluated", len(result["steps"]))

    with st.expander("How to interpret this score", expanded=True):
        st.markdown(
            """
            The **Composite Pipeline Risk Score** is created from the outputs of the connected pipeline.
            It is a weighted score based on blood pressure stage, health tier, and downstream risk signals.
            """
        )
        st.dataframe(RISK_SCALE, use_container_width=True, hide_index=True)

        if triggered:
            st.markdown("**Main active risk signals:**")
            for item in triggered:
                st.markdown(f"- **{item['label']}** — contribution weight: `{item['weight']}`")
        else:
            st.markdown("No major downstream binary risk signals were active.")

    with st.expander("Why did the system produce this result?", expanded=True):
        if patient_row is not None:
            st.markdown("**Patient factors used by the pipeline:**")
            st.dataframe(
                pd.DataFrame([summarize_patient_inputs(patient_row)]),
                use_container_width=True,
                hide_index=True,
            )

            driver_df = flag_input_drivers(patient_row)
            if not driver_df.empty:
                st.markdown("**Key input drivers detected:**")
                st.dataframe(driver_df, use_container_width=True, hide_index=True)
            else:
                st.markdown("No major elevated input drivers were detected from the entered values.")

        st.markdown("**Score contribution breakdown:**")
        st.dataframe(contribution_df, use_container_width=True, hide_index=True)

        st.caption(
            "The breakdown shows how each active stage contributed to the final composite score. "
            "The largest contributions usually come from crisis risk, emergency-care risk, stroke risk, and cardiac event risk signals."
        )

    st.subheader("Stage-by-stage explanation")

    for i, step in enumerate(result["steps"], 1):
        stage_name = step["stage"]
        friendly = STAGE_FRIENDLY_NAMES.get(stage_name, stage_name.replace("_", " "))
        label, color = format_stage_output(stage_name, step["value"])

        with st.expander(f"Step {i}: {friendly} → {label}", expanded=(i <= 4)):
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown(f"**Purpose:** {step['description']}")
                st.markdown(
                    f"**Plain-language explanation:** "
                    f"{STAGE_EXPLANATIONS.get(stage_name, 'No explanation available.')}"
                )

                if step["depends_on"]:
                    dep_vals = []
                    for d in step["depends_on"]:
                        dep_label = STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " "))
                        dep_value = step["upstream_snapshot"].get(d)
                        dep_vals.append(f"**{dep_label}** = `{dep_value}`")

                    st.markdown("**Influenced by earlier pipeline outputs:**")
                    st.markdown(", ".join(dep_vals))
                else:
                    st.markdown("**Based on:** patient vitals only.")

            with c2:
                st.markdown(
                    f"""
                    <div style="background:#ffffff;border:1px solid #dde6f2;border-radius:12px;padding:14px;text-align:center;">
                        <p style="margin:0;color:#6c757d;">Stage Result</p>
                        <h4 style="margin:6px 0;color:{color};">{label}</h4>
                        <p style="margin:0;color:#6c757d;font-size:13px;">Raw output: {step['value']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if stage_name == "Risk_Score":
                st.info(
                    "This is the Internal Hemodynamic Risk Index. It is not the same as the final "
                    "Composite Pipeline Risk Score at the top of the assessment."
                )

    st.subheader("Clinical review notes")

    notes = []

    if preds.get("BP_Medication_Recommendation") == 1:
        notes.append(
            "Medication review signal is active. This suggests the profile may require professional medication review, not automatic medication selection."
        )

    if preds.get("Hypertensive_Crisis_Risk") == 1:
        notes.append(
            "Hypertensive crisis signal is active. The projected blood-pressure pattern should be reviewed carefully."
        )

    if preds.get("Emergency_Visit_Risk") == 1:
        notes.append(
            "Emergency-care risk signal is active. The model associates this profile with a higher likelihood of urgent-care needs."
        )

    if preds.get("Stroke_Risk") == 1:
        notes.append(
            "Stroke risk signal is active. Chronic hypertension and cardiovascular strain signals should be reviewed together."
        )

    if preds.get("Heart_Attack_Risk") == 1:
        notes.append(
            "Cardiac event risk signal is active. Cardiovascular strain and acute pressure indicators contributed to this result."
        )

    if not notes:
        notes.append("No major downstream acute-risk signals were active in this assessment.")

    for note in notes:
        st.info(note)

    st.warning(SAFETY_NOTICE)


# ============================================================
# MAIN APPLICATION
# ============================================================

if not check_access():
    st.stop()

config, models, metadata = load_assets()
db_ok, db_message = test_connection()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🫀 Risk Screening")
st.sidebar.caption(APP_SUBTITLE)
st.sidebar.markdown("---")

if is_cloud_mode():
    st.sidebar.info("Deployment: Cloud")
    st.sidebar.caption("Data mode: Demo / CSV dataset")
else:
    st.sidebar.info("Deployment: Local")
    if db_ok:
        st.sidebar.success("Data source: SQL Server")
    else:
        st.sidebar.warning("Data source: CSV fallback")
        st.sidebar.caption(db_message)

if models:
    st.sidebar.success(f"Models loaded: {len(models)} stages")
else:
    st.sidebar.error("No trained models found.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Dashboard version: {DASHBOARD_VERSION}")
st.sidebar.caption(f"Model version: {MODEL_VERSION}")
st.sidebar.caption(f"Last updated: {LAST_UPDATED}")
st.sidebar.markdown("---")
st.sidebar.warning("Use synthetic or approved de-identified data only.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Assessment",
    "Risk Explanation",
    "Pipeline Methodology",
    "Model Performance",
    "Research Log",
    "About & Limitations",
])


# ============================================================
# TAB 1 — ASSESSMENT
# ============================================================

with tab1:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="section-label">Clinical risk support</div>
            <h1 style="margin:0;">{APP_TITLE}</h1>
            <p style="font-size:17px;margin:10px 0 0 0;">{INTENDED_USE}</p>
            <p style="font-size:14px;margin:8px 0 0 0;color:#6c757d;">{SAFETY_NOTICE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_status_pills()

    st.markdown("### Patient input")

    st.info(
        "The dashboard uses age, gender, average blood pressure values, blood pressure volatility, "
        "pulse pressure, and reading count to run the connected risk pipeline."
    )

    input_mode = st.radio(
        "Input source",
        ["Manual entry", "Load from dataset"],
        horizontal=True,
        disabled=False,
    )

    selected_patient_id = None
    region_id = 1

    if input_mode == "Load from dataset":
        try:
            patients_df = load_patients_from_db()

            if patients_df.empty:
                st.warning(
                    "No patient dataset found. Include `generated_data/ml_dataset.csv` for cloud mode "
                    "or connect SQL Server locally."
                )
                input_mode = "Manual entry"
            else:
                options = patients_df["PatientID"].tolist()
                selected_patient_id = st.selectbox("Select sample patient", options)
                row = patients_df[patients_df["PatientID"] == selected_patient_id].iloc[0]

                default_age = int(row["Age"])

                if isinstance(row["Gender"], str):
                    default_gender = row["Gender"]
                else:
                    default_gender = "Male" if int(row["Gender"]) == 0 else "Female"

                default_sys = float(row["Avg_Systolic"])
                default_dia = float(row["Avg_Diastolic"])
                default_vol = float(row["BP_Volatility"])
                default_pulse = float(row["Pulse_Pressure"])
                default_readings = int(row["Reading_Count"])
                region_id = int(row["RegionID"])

        except Exception as exc:
            st.error(f"Could not load sample patients: {exc}")
            input_mode = "Manual entry"

    if input_mode == "Manual entry":
        default_age, default_gender = 55, "Male"
        default_sys, default_dia = 135.0, 85.0
        default_vol, default_pulse, default_readings = 12.0, 50.0, 30

    with st.expander("Patient vitals and pattern indicators", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 1, 110, int(default_age))
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0 if default_gender == "Male" else 1,
            )

        with c2:
            systolic = st.number_input("Average systolic BP (mmHg)", 80, 220, int(default_sys))
            diastolic = st.number_input("Average diastolic BP (mmHg)", 50, 140, int(default_dia))

        with c3:
            volatility = st.number_input("BP volatility / variation", 0.0, 40.0, float(default_vol))
            pulse = st.number_input("Pulse pressure", 20, 120, int(default_pulse))
            readings = st.number_input("Reading count", 1, 100, int(default_readings))

    if not is_cloud_mode():
        rc1, rc2 = st.columns(2)
        with rc1:
            researcher_name = st.text_input("Reviewer name (optional)", placeholder="e.g. Clinical analyst")
        with rc2:
            research_notes = st.text_input("Review notes (optional)", placeholder="e.g. demo review")
    else:
        researcher_name = None
        research_notes = None

    if st.button("Run risk screening pipeline", use_container_width=True, type="primary"):
        if not models:
            st.error("No trained models found. Run `python main_engine.py` locally and push the `models/` folder.")
        else:
            patient, gender_label = build_patient_row(
                age,
                gender,
                systolic,
                diastolic,
                volatility,
                pulse,
                readings,
                region_id,
            )

            result = assess_patient(patient, models)
            render_assessment_result(result, patient_row=patient)

            if db_ok and not is_cloud_mode():
                try:
                    source = "Dataset" if selected_patient_id else "Manual"

                    assessment_id = save_research_assessment(
                        patient_id=selected_patient_id,
                        input_source=source,
                        patient_row=patient,
                        gender_label=gender_label,
                        overall=result["overall"],
                        steps=result["steps"],
                        researcher_name=researcher_name or None,
                        notes=research_notes or None,
                    )

                    st.success(f"Saved to local SQL research log — Assessment #{assessment_id}")
                    load_patients_from_db.clear()

                except Exception as exc:
                    st.warning(f"Assessment ran but could not save to local database: {exc}")
            else:
                st.caption("Cloud/demo mode: assessment is displayed but not saved to a local SQL database.")


# ============================================================
# TAB 2 — RISK EXPLANATION
# ============================================================

with tab2:
    st.title("Risk Explanation Guide")

    st.markdown(
        """
        This page explains how to read the dashboard results. The goal is to make each output understandable
        without requiring the user to inspect the source code.
        """
    )

    st.subheader("Composite Pipeline Risk Score")
    st.write(
        "The Composite Pipeline Risk Score is the final weighted score used to assign the overall risk level. "
        "It is not a clinical measurement, not a percentage, and not a diagnosis."
    )
    st.dataframe(RISK_SCALE, use_container_width=True, hide_index=True)

    st.subheader("Score formula")
    formula_df = pd.DataFrame(
        [
            {"Component": "Blood Pressure Stage", "Contribution Rule": "BP_Stage × 0.8"},
            {"Component": "Health Risk Tier", "Contribution Rule": "Health_Risk_Tier × 0.6"},
            {"Component": "Hypertensive Crisis Risk Signal", "Contribution Rule": "+3.0 if active"},
            {"Component": "Emergency-Care Risk Signal", "Contribution Rule": "+3.0 if active"},
            {"Component": "Stroke Risk Signal", "Contribution Rule": "+2.5 if active"},
            {"Component": "Cardiac Event Risk Signal", "Contribution Rule": "+2.5 if active"},
            {"Component": "Chronic Hypertension Pattern", "Contribution Rule": "+1.5 if active"},
            {"Component": "Cardiovascular Strain Signal", "Contribution Rule": "+1.5 if active"},
            {"Component": "Hypertensive Pattern Flag", "Contribution Rule": "+1.0 if active"},
            {"Component": "Medication Review Signal", "Contribution Rule": "+0.5 if active"},
        ]
    )
    st.dataframe(formula_df, use_container_width=True, hide_index=True)

    st.subheader("Important distinction")
    st.warning(
        "The Internal Hemodynamic Risk Index is a model-generated intermediate value. "
        "It is different from the final Composite Pipeline Risk Score used for the overall risk level."
    )

    st.subheader("Recommended wording for interpretation")
    st.markdown(
        """
        - Use **risk signal active** instead of saying the patient definitely has a condition.
        - Use **clinical review recommended** instead of treatment instruction.
        - Use **model-based screening output** instead of diagnosis.
        - Use **demo/synthetic data** unless a reviewed clinical data source is connected.
        """
    )


# ============================================================
# TAB 3 — PIPELINE METHODOLOGY
# ============================================================

with tab3:
    st.title("Pipeline Methodology")

    st.markdown(
        """
        This dashboard uses an interconnected machine learning pipeline. Earlier stages produce outputs that become
        inputs for later stages, allowing downstream risks to reflect upstream blood-pressure patterns.
        """
    )

    st.subheader("Input features")
    feature_df = pd.DataFrame(
        [
            {"Feature": "Age", "Description": "Patient age used as a demographic risk indicator."},
            {"Feature": "Gender", "Description": "Encoded demographic variable used by the model."},
            {"Feature": "RegionID", "Description": "Synthetic/sample regional grouping variable."},
            {"Feature": "Average Systolic BP", "Description": "Average systolic blood pressure across readings."},
            {"Feature": "Average Diastolic BP", "Description": "Average diastolic blood pressure across readings."},
            {"Feature": "BP Volatility", "Description": "Variation of systolic readings over time."},
            {"Feature": "Pulse Pressure", "Description": "Difference between systolic and diastolic pressure."},
            {"Feature": "Reading Count", "Description": "Number of available readings used to summarize the patient."},
        ]
    )
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.subheader("Pipeline stages")
    if config:
        flow_rows = []

        for s in PIPELINE_STAGES:
            flow_rows.append({
                "Stage": STAGE_FRIENDLY_NAMES.get(s["name"], s["name"].replace("_", " ")),
                "Type": s["type"],
                "Depends on": ", ".join(
                    STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " "))
                    for d in s["depends_on"]
                ) or "Patient vitals only",
                "Description": s["description"],
            })

        st.dataframe(pd.DataFrame(flow_rows), use_container_width=True, hide_index=True)

        st.subheader("Dependency diagram")

        dot_lines = [
            "digraph clinical_pipeline {",
            '    rankdir=TB; node [shape=box style=rounded];',
            '    Vitals [label="Patient Vitals" fillcolor="#e3f2fd" style="filled,rounded"];',
            '    Overall [label="Composite Pipeline Risk Score\\n(weighted final score)" fillcolor="#fce4ec" style="filled,rounded"];',
        ]

        for stage in PIPELINE_STAGES:
            safe = stage["name"]
            label = STAGE_FRIENDLY_NAMES.get(stage["name"], stage["name"]).replace('"', "")
            dot_lines.append(f'    {safe} [label="{label}"];')

            if not stage["depends_on"]:
                dot_lines.append(f"    Vitals -> {safe};")

            for dep in stage["depends_on"]:
                dot_lines.append(f"    {dep} -> {safe};")

        for src in (
            "Hypertensive_Crisis_Risk",
            "Emergency_Visit_Risk",
            "Stroke_Risk",
            "Heart_Attack_Risk",
            "BP_Stage",
            "Health_Risk_Tier",
        ):
            dot_lines.append(f"    {src} -> Overall;")

        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines))
    else:
        st.warning("Pipeline configuration was not found. Train and deploy the model artifacts first.")

    st.subheader("Data and validation status")
    st.info(
        "Current public deployment is configured for demonstration data/model artifacts. "
        "Before clinical or operational use, the model should be externally validated on appropriate real-world data, "
        "calibrated, reviewed for bias, and governed under the relevant medical software requirements."
    )


# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================

with tab4:
    st.title("Model Performance")

    st.markdown(
        """
        Model performance metrics help users understand how each stage performed during evaluation.
        These metrics should be interpreted in the context of the training data and validation design.
        """
    )

    if metadata:
        selected = st.selectbox(
            "Select stage",
            [s["name"] for s in PIPELINE_STAGES],
            format_func=lambda x: STAGE_FRIENDLY_NAMES.get(x, x.replace("_", " ")),
        )

        meta = metadata.get(selected, {})

        if meta:
            st.subheader(STAGE_FRIENDLY_NAMES.get(selected, selected.replace("_", " ")))
            st.caption(meta.get("description", ""))

            if meta.get("depends_on"):
                st.markdown(
                    "**Upstream inputs:** "
                    + ", ".join(
                        STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " "))
                        for d in meta["depends_on"]
                    )
                )
            else:
                st.markdown("**Upstream inputs:** Patient vitals only")

            st.info(STAGE_EXPLANATIONS.get(selected, "No plain-language explanation available."))

            if "accuracy" in meta:
                m1, m2, m3, m4 = st.columns(4)

                m1.markdown(
                    f'<div class="metric-box"><p>Accuracy</p><h3>{meta["accuracy"]:.1%}</h3></div>',
                    unsafe_allow_html=True,
                )
                m2.markdown(
                    f'<div class="metric-box"><p>Precision</p><h3>{meta["precision"]:.1%}</h3></div>',
                    unsafe_allow_html=True,
                )
                m3.markdown(
                    f'<div class="metric-box"><p>Recall</p><h3>{meta["recall"]:.1%}</h3></div>',
                    unsafe_allow_html=True,
                )
                m4.markdown(
                    f'<div class="metric-box"><p>F1</p><h3>{meta["f1_score"]:.1%}</h3></div>',
                    unsafe_allow_html=True,
                )

                y_test = np.array(meta.get("y_test", []))
                y_probs = meta.get("y_probs")

                if y_probs and len(np.unique(y_test)) == 2:
                    c1, c2 = st.columns(2)

                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            name=f"AUC {auc(fpr, tpr):.3f}",
                            line=dict(width=2),
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            line=dict(dash="dash"),
                            name="Random baseline",
                        )
                    )
                    fig_roc.update_layout(title="ROC Curve", template="plotly_white")
                    c1.plotly_chart(fig_roc, use_container_width=True)

                    p, r, _ = precision_recall_curve(y_test, y_probs)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=r, y=p, line=dict(width=2)))
                    fig_pr.update_layout(title="Precision-Recall Curve", template="plotly_white")
                    c2.plotly_chart(fig_pr, use_container_width=True)

            elif "rmse" in meta:
                st.metric("RMSE", f"{meta['rmse']:.4f}")
                st.caption("Lower RMSE means the regression model prediction is closer to the target value.")

        st.markdown("---")
        st.subheader("All model stages summary")

        rows = []
        for name, m in metadata.items():
            row = {
                "Stage": STAGE_FRIENDLY_NAMES.get(name, name.replace("_", " ")),
                "Depends on": ", ".join(
                    STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " "))
                    for d in m.get("depends_on", [])
                ) or "Patient vitals only",
            }

            if "accuracy" in m:
                row["Type"] = m.get("stage_type", "classification")
                row["Score"] = f"{m['accuracy']:.1%}"
            elif "rmse" in m:
                row["Type"] = "regression"
                row["Score"] = f"RMSE {m['rmse']:.3f}"

            rows.append(row)

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    else:
        st.warning("No evaluation metadata found. Run `python main_engine.py` and deploy the `models/` folder.")


# ============================================================
# TAB 5 — RESEARCH LOG
# ============================================================

with tab5:
    st.title("Research Log")

    if is_cloud_mode():
        st.info(
            "Cloud deployment uses demo/CSV mode and does not write new assessments to your local SQL Server. "
            "Use the local deployment if persistent SQL research logging is required."
        )

    if not db_ok:
        st.warning("Database not connected. Research history is unavailable in this mode.")
    else:
        try:
            summary = fetch_research_summary()

            if summary and (summary.get("Total_Assessments", 0) or 0) > 0:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total assessments", int(summary["Total_Assessments"]))
                s2.metric("Review flags", int(summary["Alert_Count"] or 0))
                s3.metric(
                    "Avg composite score",
                    f"{summary['Avg_Risk_Score']:.2f}" if summary["Avg_Risk_Score"] else "—",
                )
                s4.metric("Unique patients", int(summary["Unique_Patients"] or 0))

            history = fetch_research_history(limit=200)

            if history.empty:
                st.info("No research assessments found.")
            else:
                st.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Assessed_At": st.column_config.DatetimeColumn("Assessed at"),
                        "Overall_Risk_Score": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

                assessment_ids = history["AssessmentID"].tolist()
                chosen = st.selectbox("Inspect assessment steps", assessment_ids)

                if chosen:
                    steps_df = fetch_assessment_steps(chosen)
                    st.subheader(f"Pipeline steps for assessment #{chosen}")
                    st.dataframe(steps_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Model registry")

            registry = fetch_model_registry()

            if registry.empty:
                st.caption("No model registry rows found.")
            else:
                st.dataframe(registry, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Could not load research data: {exc}")


# ============================================================
# TAB 6 — ABOUT AND LIMITATIONS
# ============================================================

with tab6:
    st.title("About & Limitations")

    st.subheader("Intended use")
    st.write(INTENDED_USE)

    st.subheader("Safety notice")
    st.warning(SAFETY_NOTICE)

    st.subheader("Privacy notice")
    st.warning(PRIVACY_NOTICE)

    st.subheader("Current deployment status")
    deployment_rows = [
        {"Item": "Dashboard version", "Value": DASHBOARD_VERSION},
        {"Item": "Model version", "Value": MODEL_VERSION},
        {"Item": "Last updated", "Value": LAST_UPDATED},
        {"Item": "Deployment mode", "Value": "Cloud demo" if is_cloud_mode() else "Local / SQL-capable"},
        {"Item": "Data mode", "Value": "Synthetic/demo or approved de-identified data"},
        {"Item": "Persistence", "Value": "Cloud demo does not write to local SQL Server"},
    ]
    st.dataframe(pd.DataFrame(deployment_rows), use_container_width=True, hide_index=True)

    st.subheader("Known limitations")
    limitations = pd.DataFrame(
        [
            {"Limitation": "Validation", "Description": "The public demo should not be considered clinically validated without external validation on appropriate real-world data."},
            {"Limitation": "Data source", "Description": "If synthetic/demo data is used, outputs demonstrate workflow behavior rather than real-world clinical performance."},
            {"Limitation": "Clinical use", "Description": "The dashboard should not be used to diagnose, treat, or replace clinical judgment."},
            {"Limitation": "Generalization", "Description": "Model behavior may not generalize to populations not represented in the training data."},
            {"Limitation": "Calibration", "Description": "Risk scores and probabilities should be calibrated and monitored before operational use."},
            {"Limitation": "Governance", "Description": "Any clinical deployment would require privacy, security, bias, safety, and regulatory review."},
        ]
    )
    st.dataframe(limitations, use_container_width=True, hide_index=True)

    st.subheader("Recommended next professional upgrades")
    st.markdown(
        """
        - Add externally validated clinical datasets.
        - Add model calibration and confidence intervals.
        - Add SHAP or feature-attribution explanations.
        - Add bias and subgroup performance checks.
        - Add secure authentication and role-based access.
        - Add audit logging for every assessment.
        - Add monitoring for model drift and data quality.
        - Add a formal model card and data sheet.
        """
    )
