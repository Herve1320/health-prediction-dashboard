import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from clinical_pipeline import PIPELINE_STAGES, assess_patient, load_pipeline_models


# ============================================================
# APP CONSTANTS
# ============================================================

APP_TITLE = "Hypertension Risk Screening Dashboard"
APP_SUBTITLE = "Clinical Risk Support Pipeline"
DASHBOARD_VERSION = "2.0.0"
MODEL_VERSION = "v2.0-calibrated"
LAST_UPDATED = "June 2026"
OVERALL_SCORE_MAX = 19.1

INTENDED_USE = (
    "This dashboard supports blood-pressure-related risk screening by combining patient vitals, "
    "blood pressure pattern indicators, calibrated model confidence, and a multi-stage clinical pipeline."
)

SAFETY_NOTICE = (
    "This system is not intended to diagnose, treat, prevent, or replace professional medical judgment. "
    "Outputs should be reviewed by a qualified professional before any clinical decision."
)

PRIVACY_NOTICE = (
    "Use synthetic, sample, or approved de-identified data only. Do not enter identifiable real patient "
    "information in a public demo environment."
)


# ============================================================
# CONFIG IMPORTS
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
# PAGE SETUP + STYLES
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f5f7fb;
        --card: #ffffff;
        --border: #dce6f2;
        --text: #172033;
        --muted: #64748b;
        --blue: #2563eb;
        --green: #16a34a;
        --yellow: #eab308;
        --orange: #f97316;
        --red: #dc2626;
        --purple: #7c3aed;
    }

    .main {
        background: var(--bg);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        color: var(--text);
        font-size: 36px;
        line-height: 1.1;
    }

    .hero p {
        color: var(--muted);
        font-size: 16px;
        margin: 10px 0 0 0;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 16px;
    }

    .compact-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        margin-bottom: 12px;
    }

    .risk-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 9px solid var(--blue);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 16px;
    }

    .section-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .pill {
        display: inline-block;
        background: #eef4ff;
        border: 1px solid #d7e6ff;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 5px 11px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 13px;
        font-weight: 600;
    }

    .pill-green {
        background: #ecfdf5;
        border-color: #bbf7d0;
        color: #15803d;
    }

    .pill-red {
        background: #fef2f2;
        border-color: #fecaca;
        color: #b91c1c;
    }

    .pill-purple {
        background: #f5f3ff;
        border-color: #ddd6fe;
        color: #6d28d9;
    }

    .score-box {
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
    }

    .score-box h2 {
        margin: 4px 0;
    }

    .muted {
        color: var(--muted);
        font-size: 14px;
    }

    .stage-result {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }

    .footer-note {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        border-radius: 14px;
        padding: 14px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LABELS + EXPLANATIONS
# ============================================================

LABELS = {
    "BP_Stage": {
        0: ("Normal", "#16a34a"),
        1: ("Elevated", "#eab308"),
        2: ("Hypertension Stage 1", "#f97316"),
        3: ("Hypertension Stage 2", "#dc2626"),
    },
    "Health_Risk_Tier": {
        0: ("Low Tier", "#16a34a"),
        1: ("Moderate Tier", "#f97316"),
        2: ("Critical Tier", "#dc2626"),
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
    "BP_Stage": "Classifies the blood pressure pattern using average systolic and diastolic values.",
    "Health_Risk_Tier": "Summarizes general risk using blood pressure stage and blood pressure volatility.",
    "Future_Systolic": "Projects future systolic blood pressure from the current pattern and upstream stage outputs.",
    "Future_Diastolic": "Projects future diastolic blood pressure from the current pattern and upstream stage outputs.",
    "Chronic_Hypertension_Development": "Flags whether the profile resembles a longer-term hypertension pattern.",
    "Hypertensive_Event": "Flags whether the profile resembles an active hypertensive pattern.",
    "Hypertensive_Crisis_Risk": "Flags a severe blood-pressure-related risk signal.",
    "BP_Medication_Recommendation": "Flags whether the profile may need professional medication review.",
    "Cardiovascular_Event_Risk": "Flags cardiovascular strain from BP stage, pulse pressure, and volatility.",
    "Stroke_Risk": "Combines chronic hypertension, crisis, and cardiovascular strain indicators.",
    "Heart_Attack_Risk": "Combines cardiovascular strain, BP stage, and crisis signal indicators.",
    "Emergency_Visit_Risk": "Estimates whether the profile is associated with higher emergency-care risk.",
    "Risk_Score": "Internal numeric index used by the pipeline; it is not the final composite score.",
    "Probability_Hypertension": "Probability-like model output estimating association with hypertension pattern.",
}

RISK_SCALE = pd.DataFrame(
    [
        {"Level": "Low", "Score Range": "0.0 – 1.9", "Meaning": "Few or no major risk signals activated."},
        {"Level": "Moderate", "Score Range": "2.0 – 3.9", "Meaning": "Early warning signals are present."},
        {"Level": "High", "Score Range": "4.0 – 6.9", "Meaning": "Multiple important risk signals are active."},
        {"Level": "Critical", "Score Range": "7.0+", "Meaning": "Several high-severity signals are active together."},
    ]
)


# ============================================================
# AUTH + LOADING
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


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def format_stage_output(stage_name, value):
    if stage_name in LABELS:
        return LABELS[stage_name].get(safe_int(value), ("Unknown", "#64748b"))

    if stage_name in BINARY_STAGES or stage_name == "Overall_Risk_Flag":
        value_int = safe_int(value)
        if value_int == 1:
            return "Risk Signal Active", "#dc2626"
        return "Not Flagged", "#16a34a"

    if stage_name == "Probability_Hypertension":
        return f"{safe_float(value):.1%}", "#2563eb"

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.1f}", "#2563eb"

    return str(value), "#2563eb"


def confidence_band(confidence):
    if confidence is None:
        return "Not available", "#64748b"
    conf = safe_float(confidence)
    if conf >= 0.80:
        return "High", "#16a34a"
    if conf >= 0.60:
        return "Medium", "#f97316"
    return "Low", "#dc2626"


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


def summarize_patient_inputs(patient):
    return {
        "Age": safe_int(patient.get("Age")),
        "Gender": "Male" if safe_int(patient.get("Gender")) == 0 else "Female",
        "Avg Systolic": safe_float(patient.get("Avg_Systolic")),
        "Avg Diastolic": safe_float(patient.get("Avg_Diastolic")),
        "BP Volatility": safe_float(patient.get("BP_Volatility")),
        "Pulse Pressure": safe_float(patient.get("Pulse_Pressure")),
        "Reading Count": safe_int(patient.get("Reading_Count")),
    }


def get_overall_interpretation(level):
    if level == "Low":
        return "Few or no major risk signals were activated."
    if level == "Moderate":
        return "Early warning signals are present, but severe indicators are limited."
    if level == "High":
        return "Multiple clinically important risk signals were activated."
    return "Several high-severity risk signals were activated together."


def get_triggered_risk_factors(preds):
    triggered = []
    for name, weight in HIGH_SEVERITY_STAGES.items():
        active = safe_int(preds.get(name, 0)) == 1
        if active:
            triggered.append({
                "Stage": STAGE_FRIENDLY_NAMES.get(name, name.replace("_", " ")),
                "Weight": weight,
                "Explanation": STAGE_EXPLANATIONS.get(name, ""),
            })
    return triggered


def explain_score_contribution(preds):
    rows = []

    bp_stage = safe_float(preds.get("BP_Stage", 0))
    health_tier = safe_float(preds.get("Health_Risk_Tier", 0))

    rows.append({
        "Component": "Blood Pressure Stage",
        "Output": bp_stage,
        "Rule": "stage × 0.8",
        "Contribution": round(bp_stage * 0.8, 2),
    })

    rows.append({
        "Component": "Health Risk Tier",
        "Output": health_tier,
        "Rule": "tier × 0.6",
        "Contribution": round(health_tier * 0.6, 2),
    })

    for item in get_triggered_risk_factors(preds):
        rows.append({
            "Component": item["Stage"],
            "Output": "Active",
            "Rule": f"+{item['Weight']}",
            "Contribution": item["Weight"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Contribution", ascending=False, ignore_index=True)
    return df


def flag_input_drivers(patient):
    drivers = []

    systolic = safe_float(patient.get("Avg_Systolic"))
    diastolic = safe_float(patient.get("Avg_Diastolic"))
    volatility = safe_float(patient.get("BP_Volatility"))
    pulse = safe_float(patient.get("Pulse_Pressure"))
    age = safe_int(patient.get("Age"))

    if systolic >= 140:
        drivers.append(("Average systolic BP", systolic, "Hypertension Stage 2 systolic range."))
    elif systolic >= 130:
        drivers.append(("Average systolic BP", systolic, "Hypertension Stage 1 systolic range."))
    elif systolic >= 120:
        drivers.append(("Average systolic BP", systolic, "Elevated systolic range."))

    if diastolic >= 90:
        drivers.append(("Average diastolic BP", diastolic, "Hypertension Stage 2 diastolic range."))
    elif diastolic >= 80:
        drivers.append(("Average diastolic BP", diastolic, "Hypertension Stage 1 diastolic range."))

    if volatility >= 15:
        drivers.append(("BP volatility", volatility, "High variation across readings."))
    elif volatility >= 10:
        drivers.append(("BP volatility", volatility, "Moderate variation across readings."))

    if pulse >= 60:
        drivers.append(("Pulse pressure", pulse, "Elevated hemodynamic strain signal."))
    elif pulse >= 50:
        drivers.append(("Pulse pressure", pulse, "Moderate hemodynamic strain signal."))

    if age >= 65:
        drivers.append(("Age", age, "Older age group in the risk profile."))

    if not drivers:
        drivers.append(("Input profile", "Within expected range", "No major elevated input drivers detected."))

    return pd.DataFrame(drivers, columns=["Input Factor", "Value", "Interpretation"])


def get_confidence_summary(steps):
    rows = []
    for step in steps:
        stage = step.get("stage")
        conf = step.get("confidence")
        interval = step.get("prediction_interval")
        label, _ = format_stage_output(stage, step.get("value"))
        band, _ = confidence_band(conf)

        if conf is not None:
            uncertainty = f"{safe_float(conf):.1%}"
        elif interval:
            lower = interval.get("lower")
            upper = interval.get("upper")
            uncertainty = f"{safe_float(lower):.1f} to {safe_float(upper):.1f}"
        else:
            uncertainty = "Not available"

        rows.append({
            "Stage": STAGE_FRIENDLY_NAMES.get(stage, stage.replace("_", " ")),
            "Result": label,
            "Confidence / Interval": uncertainty,
            "Confidence Band": band if conf is not None else "Interval",
            "Calibrated": "Yes" if step.get("calibrated") else "No / N/A",
        })
    return pd.DataFrame(rows)


def render_status_pills():
    mode = "Cloud demo" if is_cloud_mode() else "Local SQL capable"
    st.markdown(
        f"""
        <span class="pill">Dashboard {DASHBOARD_VERSION}</span>
        <span class="pill pill-purple">Model {MODEL_VERSION}</span>
        <span class="pill">{mode}</span>
        <span class="pill pill-green">Confidence enabled</span>
        """,
        unsafe_allow_html=True,
    )


def render_score_gauge(score, level):
    color_map = {
        "Low": "#16a34a",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Critical": "#dc2626",
    }
    score = safe_float(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": f" / {OVERALL_SCORE_MAX}", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, OVERALL_SCORE_MAX]},
                "bar": {"color": color_map.get(level, "#2563eb")},
                "steps": [
                    {"range": [0, 2], "color": "#dcfce7"},
                    {"range": [2, 4], "color": "#fef9c3"},
                    {"range": [4, 7], "color": "#ffedd5"},
                    {"range": [7, OVERALL_SCORE_MAX], "color": "#fee2e2"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
    return fig


# ============================================================
# RENDER ASSESSMENT RESULTS
# ============================================================

def render_assessment_result(result, patient_row=None):
    overall = result["overall"]
    preds = result["predictions"]
    steps = result["steps"]

    level = overall["Overall_Risk_Level"]
    score = safe_float(overall["Overall_Risk_Score"])
    active_signals = get_triggered_risk_factors(preds)
    contribution_df = explain_score_contribution(preds)
    confidence_df = get_confidence_summary(steps)

    level_colors = {
        "Low": "#16a34a",
        "Moderate": "#eab308",
        "High": "#f97316",
        "Critical": "#dc2626",
    }
    risk_color = level_colors.get(level, "#2563eb")

    st.markdown("---")
    st.subheader("Assessment Result")

    summary_col, gauge_col = st.columns([1.4, 1])

    with summary_col:
        st.markdown(
            f"""
            <div class="risk-card" style="border-left-color:{risk_color};">
                <div class="section-label">Composite screening output</div>
                <h2 style="margin:0;color:{risk_color};">Overall Risk Level: {level}</h2>
                <p style="font-size:18px;margin:10px 0 0 0;">
                    Composite Pipeline Risk Score: <b>{score:.1f}</b> / {OVERALL_SCORE_MAX}
                </p>
                <p style="font-size:15px;margin:10px 0 0 0;color:#475569;">
                    {get_overall_interpretation(level)}
                </p>
                <p style="font-size:13px;margin:10px 0 0 0;color:#64748b;">
                    This is a weighted model-based screening score, not a diagnosis or treatment instruction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Review status", "Review needed" if overall["Overall_Risk_Flag"] else "Clear")
        m2.metric("Active risk signals", len(active_signals))
        m3.metric("Pipeline stages", len(steps))

    with gauge_col:
        st.plotly_chart(render_score_gauge(score, level), use_container_width=True)

    result_tabs = st.tabs([
        "Summary",
        "Confidence",
        "Why this result?",
        "Pipeline details",
        "Clinical review notes",
    ])

    with result_tabs[0]:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Patient profile used")
            if patient_row is not None:
                st.dataframe(
                    pd.DataFrame([summarize_patient_inputs(patient_row)]),
                    use_container_width=True,
                    hide_index=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Active risk signals")
            if active_signals:
                st.dataframe(pd.DataFrame(active_signals), use_container_width=True, hide_index=True)
            else:
                st.success("No major downstream risk signals were active.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Score scale")
        st.dataframe(RISK_SCALE, use_container_width=True, hide_index=True)

    with result_tabs[1]:
        st.markdown("#### Prediction confidence and uncertainty")
        st.info(
            "Classification stages show calibrated model confidence when available. "
            "Regression stages show approximate prediction intervals when available."
        )

        available_conf = confidence_df[
            confidence_df["Confidence / Interval"].astype(str).str.lower() != "not available"
        ]
        if available_conf.empty:
            st.warning(
                "Confidence values are not available yet. Re-run `python main_engine.py` with the calibrated "
                "`clinical_pipeline.py`, then restart the dashboard."
            )
        else:
            st.dataframe(confidence_df, use_container_width=True, hide_index=True)

            confidence_values = []
            for step in steps:
                if step.get("confidence") is not None:
                    confidence_values.append(safe_float(step.get("confidence")))

            if confidence_values:
                avg_conf = float(np.mean(confidence_values))
                band, band_color = confidence_band(avg_conf)
                st.markdown(
                    f"""
                    <div class="compact-card">
                        <div class="section-label">Average classification confidence</div>
                        <h2 style="margin:0;color:{band_color};">{avg_conf:.1%} — {band}</h2>
                        <p class="muted">This summarizes confidence across classification stages only.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with result_tabs[2]:
        st.markdown("#### Key input drivers")
        if patient_row is not None:
            st.dataframe(flag_input_drivers(patient_row), use_container_width=True, hide_index=True)

        st.markdown("#### Score contribution breakdown")
        st.dataframe(contribution_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        if not contribution_df.empty:
            fig.add_trace(
                go.Bar(
                    x=contribution_df["Contribution"],
                    y=contribution_df["Component"],
                    orientation="h",
                    text=contribution_df["Contribution"],
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Contribution to Composite Pipeline Risk Score",
                xaxis_title="Score contribution",
                yaxis_title="",
                height=420,
                template="plotly_white",
                margin=dict(l=10, r=10, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with result_tabs[3]:
        st.markdown("#### Stage-by-stage pipeline details")
        for i, step in enumerate(steps, 1):
            stage_name = step["stage"]
            friendly = STAGE_FRIENDLY_NAMES.get(stage_name, stage_name.replace("_", " "))
            label, color = format_stage_output(stage_name, step["value"])
            conf = step.get("confidence")
            band, band_color = confidence_band(conf)
            interval = step.get("prediction_interval")

            with st.expander(f"Step {i}: {friendly} → {label}", expanded=(i <= 3)):
                c1, c2 = st.columns([2, 1])

                with c1:
                    st.markdown(f"**Purpose:** {step.get('description', '')}")
                    st.markdown(f"**Plain-language explanation:** {STAGE_EXPLANATIONS.get(stage_name, 'No explanation available.')}")

                    if step.get("depends_on"):
                        dep_vals = []
                        for dep in step["depends_on"]:
                            dep_label = STAGE_FRIENDLY_NAMES.get(dep, dep.replace("_", " "))
                            dep_value = step.get("upstream_snapshot", {}).get(dep)
                            dep_vals.append(f"**{dep_label}** = `{dep_value}`")
                        st.markdown("**Influenced by earlier outputs:**")
                        st.markdown(", ".join(dep_vals))
                    else:
                        st.markdown("**Based on:** patient vitals only.")

                    if stage_name == "Risk_Score":
                        st.info(
                            "This is the Internal Hemodynamic Risk Index. It is different from the final "
                            "Composite Pipeline Risk Score."
                        )

                with c2:
                    st.markdown(
                        f"""
                        <div class="stage-result">
                            <p style="margin:0;color:#64748b;">Stage Result</p>
                            <h4 style="margin:6px 0;color:{color};">{label}</h4>
                            <p style="margin:0;color:#64748b;font-size:13px;">Raw output: {step.get('value')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if conf is not None:
                        st.markdown(
                            f"""
                            <div class="stage-result" style="margin-top:10px;">
                                <p style="margin:0;color:#64748b;">Confidence</p>
                                <h4 style="margin:6px 0;color:{band_color};">{safe_float(conf):.1%}</h4>
                                <p style="margin:0;color:#64748b;font-size:13px;">Band: {band}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif interval:
                        st.markdown(
                            f"""
                            <div class="stage-result" style="margin-top:10px;">
                                <p style="margin:0;color:#64748b;">Approx. 95% Interval</p>
                                <h4 style="margin:6px 0;color:#2563eb;">
                                    {safe_float(interval.get('lower')):.1f} – {safe_float(interval.get('upper')):.1f}
                                </h4>
                                <p style="margin:0;color:#64748b;font-size:13px;">Based on evaluation RMSE</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with result_tabs[4]:
        st.markdown("#### Clinical review notes")
        notes = []

        if safe_int(preds.get("BP_Medication_Recommendation", 0)) == 1:
            notes.append("Medication review signal is active. This suggests the profile may require professional medication review, not automatic medication selection.")
        if safe_int(preds.get("Hypertensive_Crisis_Risk", 0)) == 1:
            notes.append("Hypertensive crisis signal is active. The projected blood-pressure pattern should be reviewed carefully.")
        if safe_int(preds.get("Emergency_Visit_Risk", 0)) == 1:
            notes.append("Emergency-care risk signal is active. The model associates this profile with higher urgent-care risk.")
        if safe_int(preds.get("Stroke_Risk", 0)) == 1:
            notes.append("Stroke risk signal is active. Chronic hypertension and cardiovascular strain signals should be reviewed together.")
        if safe_int(preds.get("Heart_Attack_Risk", 0)) == 1:
            notes.append("Cardiac event risk signal is active. Cardiovascular strain and acute pressure indicators contributed to this result.")

        if not notes:
            notes.append("No major downstream acute-risk signals were active in this assessment.")

        for note in notes:
            st.info(note)

        st.warning(SAFETY_NOTICE)


# ============================================================
# MAIN APP
# ============================================================

if not check_access():
    st.stop()

config, models, metadata = load_assets()
db_ok, db_message = test_connection()


# -----------------------------
# SIDEBAR
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
    st.sidebar.error("No trained models found")

st.sidebar.markdown("---")
st.sidebar.caption(f"Dashboard: {DASHBOARD_VERSION}")
st.sidebar.caption(f"Model: {MODEL_VERSION}")
st.sidebar.caption(f"Updated: {LAST_UPDATED}")
st.sidebar.markdown("---")
st.sidebar.warning("Use synthetic or approved de-identified data only.")


# -----------------------------
# MAIN TABS
# -----------------------------
tab_overview, tab_assess, tab_method, tab_perf, tab_log, tab_gov = st.tabs([
    "Overview",
    "Run Assessment",
    "Methodology",
    "Model Performance",
    "Research Log",
    "Governance",
])


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    st.markdown(
        f"""
        <div class="hero">
            <div class="section-label">Clinical risk support</div>
            <h1>{APP_TITLE}</h1>
            <p>{INTENDED_USE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_status_pills()

    st.markdown("### System overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">Purpose</div>
                <h3>Risk screening</h3>
                <p class="muted">Generates model-based risk signals from blood pressure patterns and patient vitals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">Output</div>
                <h3>Composite score</h3>
                <p class="muted">Combines pipeline outputs into a weighted score from 0 to approximately 19.1.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="section-label">Trust layer</div>
                <h3>Confidence + guardrails</h3>
                <p class="muted">Displays model confidence and prevents normal profiles from being escalated incorrectly.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Score interpretation")
    st.dataframe(RISK_SCALE, use_container_width=True, hide_index=True)

    st.markdown("### Important notices")
    st.warning(SAFETY_NOTICE)
    st.warning(PRIVACY_NOTICE)


# ============================================================
# RUN ASSESSMENT
# ============================================================

with tab_assess:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Assessment workflow</div>
            <h1>Run Patient Risk Screening</h1>
            <p>Enter patient vitals manually or load a sample patient from the available dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Input source",
        ["Manual entry", "Load from dataset"],
        horizontal=True,
    )

    selected_patient_id = None
    region_id = 1

    if input_mode == "Load from dataset":
        try:
            patients_df = load_patients_from_db()
            if patients_df.empty:
                st.warning("No sample patient data found. Falling back to manual entry.")
                input_mode = "Manual entry"
            else:
                selected_patient_id = st.selectbox("Select sample patient", patients_df["PatientID"].tolist())
                row = patients_df[patients_df["PatientID"] == selected_patient_id].iloc[0]

                default_age = safe_int(row["Age"])
                default_gender = row["Gender"] if isinstance(row["Gender"], str) else ("Male" if safe_int(row["Gender"]) == 0 else "Female")
                default_sys = safe_float(row["Avg_Systolic"])
                default_dia = safe_float(row["Avg_Diastolic"])
                default_vol = safe_float(row["BP_Volatility"])
                default_pulse = safe_float(row["Pulse_Pressure"])
                default_readings = safe_int(row["Reading_Count"])
                region_id = safe_int(row["RegionID"], 1)
        except Exception as exc:
            st.error(f"Could not load sample patients: {exc}")
            input_mode = "Manual entry"

    if input_mode == "Manual entry":
        default_age, default_gender = 20, "Male"
        default_sys, default_dia = 115.0, 75.0
        default_vol, default_pulse, default_readings = 5.0, 40.0, 30

    st.markdown("### Patient vitals")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=int(default_age))
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0 if default_gender == "Male" else 1,
            )

        with c2:
            systolic = st.number_input("Average systolic BP (mmHg)", min_value=80, max_value=220, value=int(default_sys))
            diastolic = st.number_input("Average diastolic BP (mmHg)", min_value=50, max_value=140, value=int(default_dia))

        with c3:
            volatility = st.number_input("BP volatility / variation", min_value=0.0, max_value=40.0, value=float(default_vol))
            pulse = st.number_input("Pulse pressure", min_value=20, max_value=120, value=int(default_pulse))
            readings = st.number_input("Reading count", min_value=1, max_value=100, value=int(default_readings))

    if not is_cloud_mode():
        r1, r2 = st.columns(2)
        with r1:
            reviewer_name = st.text_input("Reviewer name (optional)", placeholder="e.g. Clinical analyst")
        with r2:
            review_notes = st.text_input("Review notes (optional)", placeholder="e.g. demo review")
    else:
        reviewer_name = None
        review_notes = None

    run_clicked = st.button("Run risk screening", use_container_width=True, type="primary")

    if run_clicked:
        if not models:
            st.error("No trained models found. Run `python main_engine.py` and deploy the `models/` folder.")
        else:
            patient, gender_label = build_patient_row(
                age=age,
                gender=gender,
                systolic=systolic,
                diastolic=diastolic,
                volatility=volatility,
                pulse=pulse,
                readings=readings,
                region_id=region_id,
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
                        researcher_name=reviewer_name or None,
                        notes=review_notes or None,
                    )
                    st.success(f"Saved to local SQL research log — Assessment #{assessment_id}")
                    load_patients_from_db.clear()
                except Exception as exc:
                    st.warning(f"Assessment ran but could not save to local database: {exc}")
            else:
                st.caption("Cloud/demo mode: assessment is displayed but not saved to local SQL Server.")


# ============================================================
# METHODOLOGY
# ============================================================

with tab_method:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Technical transparency</div>
            <h1>Pipeline Methodology</h1>
            <p>This section explains what the system uses, how stages connect, and how the composite score is calculated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    method_tabs = st.tabs(["Inputs", "Pipeline", "Score formula", "Confidence"])

    with method_tabs[0]:
        feature_df = pd.DataFrame(
            [
                {"Feature": "Age", "Description": "Patient age used as a demographic risk indicator."},
                {"Feature": "Gender", "Description": "Encoded demographic variable used by the model."},
                {"Feature": "RegionID", "Description": "Synthetic/sample regional grouping variable."},
                {"Feature": "Average Systolic BP", "Description": "Average systolic pressure across readings."},
                {"Feature": "Average Diastolic BP", "Description": "Average diastolic pressure across readings."},
                {"Feature": "BP Volatility", "Description": "Variation of blood pressure readings over time."},
                {"Feature": "Pulse Pressure", "Description": "Difference between systolic and diastolic pressure."},
                {"Feature": "Reading Count", "Description": "Number of readings summarized for the patient."},
            ]
        )
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    with method_tabs[1]:
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

            dot_lines = [
                "digraph clinical_pipeline {",
                'rankdir=TB; node [shape=box style=rounded];',
                'Vitals [label="Patient Vitals" fillcolor="#e3f2fd" style="filled,rounded"];',
                'Overall [label="Composite Pipeline Risk Score" fillcolor="#fce4ec" style="filled,rounded"];',
            ]

            for stage in PIPELINE_STAGES:
                safe = stage["name"]
                label = STAGE_FRIENDLY_NAMES.get(stage["name"], stage["name"]).replace('"', "")
                dot_lines.append(f'{safe} [label="{label}"];')
                if not stage["depends_on"]:
                    dot_lines.append(f"Vitals -> {safe};")
                for dep in stage["depends_on"]:
                    dot_lines.append(f"{dep} -> {safe};")

            for src in ("Hypertensive_Crisis_Risk", "Emergency_Visit_Risk", "Stroke_Risk", "Heart_Attack_Risk", "BP_Stage", "Health_Risk_Tier"):
                dot_lines.append(f"{src} -> Overall;")

            dot_lines.append("}")
            st.graphviz_chart("\n".join(dot_lines))
        else:
            st.warning("Pipeline configuration was not found.")

    with method_tabs[2]:
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
        st.dataframe(RISK_SCALE, use_container_width=True, hide_index=True)

    with method_tabs[3]:
        st.markdown(
            """
            Classification models can return confidence values when probability outputs are available.
            Regression stages can return approximate 95% intervals based on evaluation RMSE.
            """
        )
        st.info(
            "These confidence values and intervals are model uncertainty indicators. They are not clinical certainty "
            "and should not be interpreted as a guarantee."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with tab_perf:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Evaluation</div>
            <h1>Model Performance</h1>
            <p>Review stage-level metrics, curves, and confidence summaries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metadata:
        selected = st.selectbox(
            "Select model stage",
            [s["name"] for s in PIPELINE_STAGES],
            format_func=lambda x: STAGE_FRIENDLY_NAMES.get(x, x.replace("_", " ")),
        )

        meta = metadata.get(selected, {})

        if meta:
            st.subheader(STAGE_FRIENDLY_NAMES.get(selected, selected.replace("_", " ")))
            st.info(STAGE_EXPLANATIONS.get(selected, "No plain-language explanation available."))

            if meta.get("depends_on"):
                st.markdown(
                    "**Upstream inputs:** "
                    + ", ".join(STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " ")) for d in meta["depends_on"])
                )
            else:
                st.markdown("**Upstream inputs:** Patient vitals only")

            if "accuracy" in meta:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{meta.get('accuracy', 0):.1%}")
                m2.metric("Precision", f"{meta.get('precision', 0):.1%}")
                m3.metric("Recall", f"{meta.get('recall', 0):.1%}")
                m4.metric("F1", f"{meta.get('f1_score', 0):.1%}")
                mean_conf = meta.get("mean_confidence")
                m5.metric("Mean confidence", f"{mean_conf:.1%}" if mean_conf is not None else "N/A")

                y_test = np.array(meta.get("y_test", []))
                y_probs = meta.get("y_probs")

                if y_probs and len(np.unique(y_test)) == 2:
                    c1, c2 = st.columns(2)

                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC {auc(fpr, tpr):.3f}"))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random baseline"))
                    fig_roc.update_layout(title="ROC Curve", template="plotly_white")
                    c1.plotly_chart(fig_roc, use_container_width=True)

                    p, r, _ = precision_recall_curve(y_test, y_probs)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=r, y=p))
                    fig_pr.update_layout(title="Precision-Recall Curve", template="plotly_white")
                    c2.plotly_chart(fig_pr, use_container_width=True)

            elif "rmse" in meta:
                st.metric("RMSE", f"{meta['rmse']:.4f}")
                st.caption("Lower RMSE means the regression model prediction is closer to the target value.")

        st.markdown("---")
        st.subheader("All model stages")

        rows = []
        for name, m in metadata.items():
            row = {
                "Stage": STAGE_FRIENDLY_NAMES.get(name, name.replace("_", " ")),
                "Type": m.get("stage_type", "regression" if "rmse" in m else "classification"),
                "Depends on": ", ".join(STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " ")) for d in m.get("depends_on", [])) or "Patient vitals only",
            }

            if "accuracy" in m:
                row["Primary Metric"] = f"Accuracy {m.get('accuracy', 0):.1%}"
                row["Mean Confidence"] = f"{m['mean_confidence']:.1%}" if m.get("mean_confidence") is not None else "N/A"
            elif "rmse" in m:
                row["Primary Metric"] = f"RMSE {m.get('rmse', 0):.3f}"
                row["Mean Confidence"] = "Interval-based"

            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.warning("No evaluation metadata found. Run `python main_engine.py` and deploy the `models/` folder.")


# ============================================================
# RESEARCH LOG
# ============================================================

with tab_log:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Audit trail</div>
            <h1>Research Log</h1>
            <p>Local SQL deployments can store assessment history. Cloud demo mode displays results without local database persistence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_cloud_mode():
        st.info("Cloud deployment uses demo/CSV mode and does not write new assessments to your local SQL Server.")

    if not db_ok:
        st.warning("Database not connected. Research history is unavailable in this mode.")
    else:
        try:
            summary = fetch_research_summary()
            if summary and (summary.get("Total_Assessments", 0) or 0) > 0:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total assessments", safe_int(summary.get("Total_Assessments")))
                s2.metric("Review flags", safe_int(summary.get("Alert_Count")))
                avg_score = summary.get("Avg_Risk_Score")
                s3.metric("Avg composite score", f"{avg_score:.2f}" if avg_score else "—")
                s4.metric("Unique patients", safe_int(summary.get("Unique_Patients")))

            history = fetch_research_history(limit=200)
            if history.empty:
                st.info("No research assessments found.")
            else:
                st.dataframe(history, use_container_width=True, hide_index=True)
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
# GOVERNANCE
# ============================================================

with tab_gov:
    st.markdown(
        """
        <div class="hero">
            <div class="section-label">Responsible use</div>
            <h1>Governance, Safety & Limitations</h1>
            <p>Professional AI systems need clear intended use, privacy controls, validation, and limitations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g1, g2 = st.columns(2)

    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Intended use")
        st.write(INTENDED_USE)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Version information")
        version_df = pd.DataFrame(
            [
                {"Item": "Dashboard version", "Value": DASHBOARD_VERSION},
                {"Item": "Model version", "Value": MODEL_VERSION},
                {"Item": "Last updated", "Value": LAST_UPDATED},
                {"Item": "Deployment mode", "Value": "Cloud demo" if is_cloud_mode() else "Local / SQL-capable"},
                {"Item": "Data mode", "Value": "Synthetic/demo or approved de-identified data"},
            ]
        )
        st.dataframe(version_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.warning(SAFETY_NOTICE)
    st.warning(PRIVACY_NOTICE)

    st.subheader("Known limitations")
    limitations = pd.DataFrame(
        [
            {"Area": "Validation", "Limitation": "The public demo should not be considered clinically validated without external validation on appropriate real-world data."},
            {"Area": "Data source", "Limitation": "Synthetic/demo data demonstrates workflow behavior rather than real-world clinical performance."},
            {"Area": "Clinical use", "Limitation": "The dashboard should not be used to diagnose, treat, or replace clinical judgment."},
            {"Area": "Calibration", "Limitation": "Confidence values are model uncertainty indicators and require monitoring before operational use."},
            {"Area": "Generalization", "Limitation": "Model behavior may not generalize to populations not represented in training and validation data."},
            {"Area": "Governance", "Limitation": "Clinical deployment would require privacy, security, bias, safety, and regulatory review."},
        ]
    )
    st.dataframe(limitations, use_container_width=True, hide_index=True)

    st.subheader("Recommended next professional upgrades")
    st.markdown(
        """
        - Add externally validated clinical datasets.
        - Add SHAP or feature-attribution explanations.
        - Add bias and subgroup performance checks.
        - Add secure authentication and role-based access.
        - Add cloud audit logging for every assessment.
        - Add monitoring for model drift and data quality.
        - Add a formal model card and data sheet.
        """
    )
