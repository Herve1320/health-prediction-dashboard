import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from clinical_pipeline import PIPELINE_STAGES, assess_patient, load_pipeline_models

try:
    from db_config import DASHBOARD_PASSWORD, APP_MODE
except Exception:
    DASHBOARD_PASSWORD = ""
    APP_MODE = "auto"

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
        return False, "Database service not available. Running in CSV/cloud mode."

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


st.set_page_config(page_title="Health Prediction — Clinical Pipeline", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    .report-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border-left: 7px solid #007bff;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-box {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    "Future_Systolic": "Predicted Future Systolic BP",
    "Future_Diastolic": "Predicted Future Diastolic BP",
    "Chronic_Hypertension_Development": "Chronic Hypertension Development",
    "Hypertensive_Event": "Hypertensive Event",
    "Hypertensive_Crisis_Risk": "Hypertensive Crisis Risk",
    "BP_Medication_Recommendation": "BP Medication Recommendation",
    "Cardiovascular_Event_Risk": "Cardiovascular Event Risk",
    "Stroke_Risk": "Stroke Risk",
    "Heart_Attack_Risk": "Heart Attack Risk",
    "Emergency_Visit_Risk": "Emergency Visit Risk",
    "Risk_Score": "Internal Pipeline Risk Index",
    "Probability_Hypertension": "Estimated Probability of Hypertension",
}

STAGE_EXPLANATIONS = {
    "BP_Stage": "Classifies the patient's blood pressure level using average systolic and diastolic values.",
    "Health_Risk_Tier": "Summarizes general risk using blood pressure stage and blood pressure volatility.",
    "Future_Systolic": "Forecasts systolic blood pressure based on the patient's current pattern.",
    "Future_Diastolic": "Forecasts diastolic blood pressure based on current vitals and earlier pipeline outputs.",
    "Chronic_Hypertension_Development": "Flags whether the patient's pattern looks consistent with longer-term hypertension risk.",
    "Hypertensive_Event": "Flags whether the current blood pressure pattern suggests an active hypertensive event.",
    "Hypertensive_Crisis_Risk": "Estimates severe blood pressure risk based on BP stage and projected systolic pressure.",
    "BP_Medication_Recommendation": "Flags whether the profile may need medication review. It does not prescribe treatment.",
    "Cardiovascular_Event_Risk": "Estimates cardiovascular strain from BP stage, pulse pressure, and volatility.",
    "Stroke_Risk": "Combines chronic hypertension, crisis risk, and cardiovascular indicators.",
    "Heart_Attack_Risk": "Uses cardiovascular risk, BP stage, and crisis risk to flag heart attack-related risk.",
    "Emergency_Visit_Risk": "Estimates whether the pattern may be associated with higher emergency-care risk.",
    "Risk_Score": "Internal numeric index used inside the pipeline; different from the final Overall Risk Score.",
    "Probability_Hypertension": "Estimated probability-like output between 0 and 1 for hypertension association.",
}


def check_access():
    if not DASHBOARD_PASSWORD:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("Health Prediction — Sign in")
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


def format_stage_output(stage_name, value):
    if stage_name in LABELS:
        return LABELS[stage_name].get(int(value), ("Unknown", "#6c757d"))
    if stage_name in BINARY_STAGES or stage_name == "Overall_Risk_Flag":
        try:
            value_int = int(value)
        except Exception:
            value_int = 1 if value is True else 0
        return ("ALERT — Risk Flagged", "#dc3545") if value_int == 1 else ("Stable / Not Flagged", "#28a745")
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
        return "Low risk: few or no major risk stages were activated."
    if level == "Moderate":
        return "Moderate risk: early warning signs are present, but severe stages are limited."
    if level == "High":
        return "High risk: multiple clinically important risk stages were activated."
    return "Critical risk: several high-severity risk stages were activated together."


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
            })
    return triggered


def explain_score_contribution(preds):
    rows = []
    bp_stage = float(preds.get("BP_Stage", 0))
    health_tier = float(preds.get("Health_Risk_Tier", 0))
    rows.append({
        "Component": "Blood Pressure Stage",
        "Value": bp_stage,
        "Weight Used": "stage × 0.8",
        "Score Contribution": round(bp_stage * 0.8, 2),
    })
    rows.append({
        "Component": "Health Risk Tier",
        "Value": health_tier,
        "Weight Used": "tier × 0.6",
        "Score Contribution": round(health_tier * 0.6, 2),
    })
    for item in get_triggered_risk_factors(preds):
        rows.append({
            "Component": item["label"],
            "Value": "Flagged",
            "Weight Used": item["weight"],
            "Score Contribution": item["weight"],
        })
    return pd.DataFrame(rows)


def summarize_patient_inputs(patient):
    return {
        "Age": patient.get("Age"),
        "Gender": "Male" if int(patient.get("Gender", 0)) == 0 else "Female",
        "Avg Systolic": patient.get("Avg_Systolic"),
        "Avg Diastolic": patient.get("Avg_Diastolic"),
        "BP Volatility": patient.get("BP_Volatility"),
        "Pulse Pressure": patient.get("Pulse_Pressure"),
        "Reading Count": patient.get("Reading_Count"),
    }


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
        <div class="report-card" style="border-left-color:{level_colors.get(level, '#007bff')};">
            <h2 style="margin:0;color:{level_colors.get(level, '#007bff')};">
                Overall Risk: {level}
            </h2>
            <p style="font-size:17px;margin:8px 0 0 0;">
                Final Overall Risk Score: <b>{score}</b> on an approximate <b>0 to 19.1</b> scale.
            </p>
            <p style="font-size:15px;margin:8px 0 0 0;">
                <b>Interpretation:</b> {interpretation}
            </p>
            <p style="font-size:13px;margin:8px 0 0 0;color:#6c757d;">
                This dashboard is a student research prototype. It is not a medical diagnosis tool.
                Results should be interpreted only as model-based risk indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    oc1, oc2, oc3 = st.columns(3)
    oc1.metric("Overall flag", "ALERT" if overall["Overall_Risk_Flag"] else "Clear")
    oc2.metric("Overall risk score", score)
    oc3.metric("Stages evaluated", len(result["steps"]))

    with st.expander("What does this overall score mean?", expanded=True):
        st.markdown(
            """
            The **Overall Risk Score** is the final composite score created from the outputs of the
            connected clinical pipeline.

            It is **not** a blood pressure reading and it is **not** a percentage. It is a weighted
            model score based on which risk stages were activated.
            """
        )

        st.markdown(
            """
            **Score scale used by this dashboard**

            - **0 to 1.9:** Low risk
            - **2 to 3.9:** Moderate risk
            - **4 to 6.9:** High risk
            - **7 or higher:** Critical risk
            - **Maximum possible score:** approximately **19.1**
            """
        )

        if triggered:
            st.markdown("**Main risk factors that increased this patient's score:**")
            for item in triggered:
                st.markdown(f"- **{item['label']}** — added weight: `{item['weight']}`")
        else:
            st.markdown("No major binary risk stages were flagged.")

    with st.expander("Why did the model produce this score?", expanded=True):
        if patient_row is not None:
            st.markdown("**Patient inputs used by the pipeline:**")
            st.dataframe(pd.DataFrame([summarize_patient_inputs(patient_row)]), use_container_width=True, hide_index=True)

        st.markdown("**Score contribution breakdown:**")
        st.dataframe(contribution_df, use_container_width=True, hide_index=True)

        st.caption(
            "The final score is calculated from blood pressure stage, health risk tier, and "
            "weighted alerts such as crisis risk, stroke risk, heart attack risk, and emergency visit risk."
        )

    st.subheader("Clinical cascade: step-by-step explanation")

    for i, step in enumerate(result["steps"], 1):
        stage_name = step["stage"]
        friendly = STAGE_FRIENDLY_NAMES.get(stage_name, stage_name.replace("_", " "))
        label, color = format_stage_output(stage_name, step["value"])

        with st.expander(f"Step {i}: {friendly} → {label}", expanded=(i <= 4)):
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown(f"**Purpose:** {step['description']}")
                st.markdown(f"**Plain-language explanation:** {STAGE_EXPLANATIONS.get(stage_name, 'No explanation available.')}")

                if step["depends_on"]:
                    dep_vals = []
                    for d in step["depends_on"]:
                        dep_label = STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " "))
                        dep_value = step["upstream_snapshot"].get(d)
                        dep_vals.append(f"**{dep_label}** = `{dep_value}`")
                    st.markdown("**Based partly on earlier stage outputs:**")
                    st.markdown(", ".join(dep_vals))
                else:
                    st.markdown("**Based on:** patient vitals only.")

            with c2:
                st.markdown(
                    f"""
                    <div style="background:#ffffff;border:1px solid #dee2e6;border-radius:10px;padding:14px;text-align:center;">
                        <p style="margin:0;color:#6c757d;">Result</p>
                        <h4 style="margin:6px 0;color:{color};">{label}</h4>
                        <p style="margin:0;color:#6c757d;font-size:13px;">Raw output: {step['value']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if stage_name == "Risk_Score":
                st.info(
                    "Note: this is the Internal Pipeline Risk Index. It is different from the final "
                    "Overall Risk Score shown at the top of the page."
                )

    st.subheader("Treatment implications / follow-up interpretation")

    implications = []
    if preds.get("BP_Medication_Recommendation") == 1:
        implications.append("Medication review flag: the model suggests this profile may need professional medication review.")
    if preds.get("Hypertensive_Crisis_Risk") == 1:
        implications.append("Crisis risk flag: the projected blood pressure pattern is concerning and should be reviewed by a healthcare professional.")
    if preds.get("Emergency_Visit_Risk") == 1:
        implications.append("Emergency visit risk flag: the model associates this profile with higher emergency-care risk.")
    if preds.get("Stroke_Risk") == 1:
        implications.append("Stroke risk flag: chronic and cardiovascular risk indicators were activated together.")
    if preds.get("Heart_Attack_Risk") == 1:
        implications.append("Heart attack risk flag: cardiovascular strain and acute pressure indicators were activated.")
    if not implications:
        implications.append("No major acute risk stages were flagged. Continue routine monitoring in this prototype context.")

    for note in implications:
        st.info(note)


if not check_access():
    st.stop()

config, models, metadata = load_assets()
db_ok, db_message = test_connection()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=90)
st.sidebar.title("Health Prediction")
st.sidebar.caption("Interconnected Clinical Pipeline v2.0")
st.sidebar.markdown("---")

if str(APP_MODE).lower() in ("cloud", "csv", "streamlit"):
    st.sidebar.info("Cloud mode: using CSV/model files")
else:
    if db_ok:
        st.sidebar.success("SQL Server connected")
    else:
        st.sidebar.warning("SQL Server offline / CSV fallback")
        st.sidebar.caption(db_message)

if models:
    st.sidebar.success(f"{len(models)} linked models loaded")
else:
    st.sidebar.error("No trained models found. Run `python main_engine.py` first.")

st.sidebar.markdown("---")
public_url_path = os.path.join(os.path.dirname(__file__), "public_url.txt")
if os.path.exists(public_url_path):
    with open(public_url_path, encoding="utf-8") as f:
        public_share_url = f.read().strip()
    st.sidebar.markdown("**Public link**")
    st.sidebar.code(public_share_url, language=None)
    st.sidebar.success("Use this URL, not localhost.")
else:
    if str(APP_MODE).lower() in ("cloud", "csv", "streamlit"):
        st.sidebar.markdown("**Deployment**")
        st.sidebar.caption("Running on Streamlit Cloud.")
    else:
        st.sidebar.markdown("**Local network share**")
        st.sidebar.code("http://<your-ip>:8501", language=None)
        st.sidebar.caption("Run `run_public.bat` for a public ngrok link.")

st.sidebar.markdown("---")
st.sidebar.caption("Prototype notice: this dashboard is for academic demonstration and should not be used as a medical diagnosis tool.")

tab1, tab2, tab3, tab4 = st.tabs(["Patient Journey", "Research Log", "Pipeline Map", "Model Metrics"])


with tab1:
    st.title("Interconnected Patient Assessment")

    st.info(
        "This dashboard estimates hypertension-related risk using patient vitals, blood pressure patterns, "
        "blood pressure volatility, pulse pressure, and outputs from 14 connected prediction stages. "
        "The result is a model-based risk estimate, not a medical diagnosis."
    )

    input_mode = st.radio("Input source", ["Manual entry", "Load from database"], horizontal=True, disabled=False)

    selected_patient_id = None
    region_id = 1

    if input_mode == "Load from database":
        try:
            patients_df = load_patients_from_db()
            if patients_df.empty:
                st.warning("No patient data found. Run `data_generator.py` locally or include `generated_data/ml_dataset.csv` for cloud mode.")
                input_mode = "Manual entry"
            else:
                options = patients_df["PatientID"].tolist()
                selected_patient_id = st.selectbox("Select patient", options)
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
            st.error(f"Could not load patients: {exc}")
            input_mode = "Manual entry"

    if input_mode == "Manual entry":
        default_age, default_gender = 55, "Male"
        default_sys, default_dia = 135.0, 85.0
        default_vol, default_pulse, default_readings = 12.0, 50.0, 30

    with st.expander("Patient vitals", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 110, int(default_age))
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_gender == "Male" else 1)
        with c2:
            systolic = st.number_input("Avg Systolic (mmHg)", 80, 220, int(default_sys))
            diastolic = st.number_input("Avg Diastolic (mmHg)", 50, 140, int(default_dia))
        with c3:
            volatility = st.number_input("BP Volatility (SD)", 0.0, 40.0, float(default_vol))
            pulse = st.number_input("Pulse Pressure", 20, 120, int(default_pulse))
            readings = st.number_input("Reading count", 1, 100, int(default_readings))

    rc1, rc2 = st.columns(2)
    with rc1:
        researcher_name = st.text_input("Researcher name (optional)", placeholder="e.g. Dr. Smith")
    with rc2:
        research_notes = st.text_input("Research notes (optional)", placeholder="e.g. control group trial")

    if st.button("Run full clinical pipeline", use_container_width=True, type="primary"):
        if not models:
            st.error("No trained models found. Run `python main_engine.py` locally and push the `models/` folder.")
        else:
            patient, gender_label = build_patient_row(age, gender, systolic, diastolic, volatility, pulse, readings, region_id)
            result = assess_patient(patient, models)
            render_assessment_result(result, patient_row=patient)

            if db_ok and str(APP_MODE).lower() not in ("cloud", "csv", "streamlit"):
                try:
                    source = "Database" if selected_patient_id else "Manual"
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
                    st.success(f"Saved to SQL — Research_Assessments #{assessment_id}")
                    load_patients_from_db.clear()
                except Exception as exc:
                    st.warning(f"Assessment ran but could not save to database: {exc}")
                    st.info("Run the research SQL tables locally if you need SQL saving.")
            else:
                st.caption("Cloud/CSV mode: assessment is displayed but not saved to local SQL Server.")


with tab2:
    st.title("Research tracking")

    if str(APP_MODE).lower() in ("cloud", "csv", "streamlit"):
        st.info("Cloud mode uses CSV/model files and does not save new assessments to your local SQL Server. Use the local version if you need permanent SQL research logging.")

    if not db_ok:
        st.warning("Database not connected. Research history is unavailable in this mode.")
    else:
        try:
            summary = fetch_research_summary()
            if summary and (summary.get("Total_Assessments", 0) or 0) > 0:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total assessments", int(summary["Total_Assessments"]))
                s2.metric("Alerts flagged", int(summary["Alert_Count"] or 0))
                s3.metric("Avg risk score", f"{summary['Avg_Risk_Score']:.2f}" if summary["Avg_Risk_Score"] else "—")
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


with tab3:
    st.title("How the stages connect")

    st.info(
        "The pipeline is interconnected: early outputs such as Blood Pressure Stage influence later "
        "outputs such as chronic hypertension, crisis risk, stroke risk, and emergency visit risk."
    )

    if config:
        flow_rows = []
        for s in PIPELINE_STAGES:
            flow_rows.append({
                "Stage": STAGE_FRIENDLY_NAMES.get(s["name"], s["name"].replace("_", " ")),
                "Type": s["type"],
                "Depends on": ", ".join(STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " ")) for d in s["depends_on"]) or "Patient vitals only",
                "Description": s["description"],
            })

        st.dataframe(pd.DataFrame(flow_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Dependency diagram")

        dot_lines = [
            "digraph clinical_pipeline {",
            '    rankdir=TB; node [shape=box style=rounded];',
            '    Vitals [label="Patient Vitals" fillcolor="#e3f2fd" style="filled,rounded"];',
            '    Overall [label="Overall Risk\\n(weighted final score)" fillcolor="#fce4ec" style="filled,rounded"];',
        ]

        for stage in PIPELINE_STAGES:
            safe = stage["name"]
            label = STAGE_FRIENDLY_NAMES.get(stage["name"], stage["name"]).replace('"', "")
            dot_lines.append(f'    {safe} [label="{label}"];')
            if not stage["depends_on"]:
                dot_lines.append(f"    Vitals -> {safe};")
            for dep in stage["depends_on"]:
                dot_lines.append(f"    {dep} -> {safe};")

        for src in ("Hypertensive_Crisis_Risk", "Emergency_Visit_Risk", "Stroke_Risk", "Heart_Attack_Risk", "BP_Stage", "Health_Risk_Tier"):
            dot_lines.append(f"    {src} -> Overall;")

        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines))

        with st.expander("How the final score is calculated"):
            st.markdown(
                """
                The final **Overall Risk Score** combines:

                - Blood Pressure Stage × 0.8
                - Health Risk Tier × 0.6
                - Hypertensive Crisis Risk: +3.0 if flagged
                - Emergency Visit Risk: +3.0 if flagged
                - Stroke Risk: +2.5 if flagged
                - Heart Attack Risk: +2.5 if flagged
                - Chronic Hypertension Development: +1.5 if flagged
                - Cardiovascular Event Risk: +1.5 if flagged
                - Hypertensive Event: +1.0 if flagged
                - Medication Recommendation: +0.5 if flagged

                The maximum possible score is approximately **19.1**.
                """
            )
    else:
        st.warning("Pipeline config not found. Train models first.")


with tab4:
    st.title("Model performance")

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
                st.markdown("**Upstream inputs:** " + ", ".join(STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " ")) for d in meta["depends_on"]))
            else:
                st.markdown("**Upstream inputs:** Patient vitals only")

            st.info(STAGE_EXPLANATIONS.get(selected, "No plain-language explanation available."))

            if "accuracy" in meta:
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(f'<div class="metric-box"><p>Accuracy</p><h3>{meta["accuracy"]:.1%}</h3></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-box"><p>Precision</p><h3>{meta["precision"]:.1%}</h3></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-box"><p>Recall</p><h3>{meta["recall"]:.1%}</h3></div>', unsafe_allow_html=True)
                m4.markdown(f'<div class="metric-box"><p>F1</p><h3>{meta["f1_score"]:.1%}</h3></div>', unsafe_allow_html=True)

                y_test = np.array(meta.get("y_test", []))
                y_probs = meta.get("y_probs")

                if y_probs and len(np.unique(y_test)) == 2:
                    c1, c2 = st.columns(2)

                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC {auc(fpr, tpr):.3f}", line=dict(width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random baseline"))
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
                "Depends on": ", ".join(STAGE_FRIENDLY_NAMES.get(d, d.replace("_", " ")) for d in m.get("depends_on", [])) or "Patient vitals only",
            }
            if "accuracy" in m:
                row["Type"] = m.get("stage_type", "classification")
                row["Score"] = f"{m['accuracy']:.1%}"
            elif "rmse" in m:
                row["Type"] = "regression"
                row["Score"] = f"RMSE {m['rmse']:.3f}"
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.warning("No evaluation metadata found. Run `python main_engine.py` and push the `models/` folder.")
