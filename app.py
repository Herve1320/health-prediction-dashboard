import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from clinical_pipeline import PIPELINE_STAGES, assess_patient, load_pipeline_models
from db_config import DASHBOARD_PASSWORD
from db_service import (
    fetch_assessment_steps,
    fetch_model_registry,
    fetch_patients,
    fetch_research_history,
    fetch_research_summary,
    save_research_assessment,
    test_connection,
)

st.set_page_config(page_title="Health Prediction — Clinical Pipeline", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .report-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #007bff;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .stage-card {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

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
    "Chronic_Hypertension_Development", "Hypertensive_Event", "Hypertensive_Crisis_Risk",
    "BP_Medication_Recommendation", "Cardiovascular_Event_Risk", "Stroke_Risk",
    "Heart_Attack_Risk", "Emergency_Visit_Risk",
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
        label, color = LABELS[stage_name].get(int(value), ("Unknown", "#6c757d"))
        return label, color
    if stage_name in BINARY_STAGES or stage_name == "Overall_Risk_Flag":
        if int(value) == 1:
            return "ALERT — Action Required", "#dc3545"
        return "Stable", "#28a745"
    if isinstance(value, float):
        return f"{value:.1f}", "#007bff"
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


def render_assessment_result(result):
    overall = result["overall"]
    level = overall["Overall_Risk_Level"]
    level_colors = {"Low": "#28a745", "Moderate": "#ffc107", "High": "#fd7e14", "Critical": "#dc3545"}
    st.markdown(f"""
        <div class="report-card" style="border-left-color:{level_colors.get(level, '#007bff')};">
            <h2 style="margin:0;color:{level_colors.get(level, '#007bff')};">
                Overall Risk: {level}
            </h2>
            <p style="font-size:17px;margin:8px 0 0 0;">
                Composite score <b>{overall['Overall_Risk_Score']}</b> —
                aggregated from all upstream clinical stages.
            </p>
        </div>
    """, unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)
    oc1.metric("Overall flag", "ALERT" if overall["Overall_Risk_Flag"] else "Clear")
    oc2.metric("Risk score", overall["Overall_Risk_Score"])
    oc3.metric("Stages evaluated", len(result["steps"]))

    st.subheader("Clinical cascade (step by step)")
    for i, step in enumerate(result["steps"], 1):
        label, color = format_stage_output(step["stage"], step["value"])
        dep_html = ""
        if step["depends_on"]:
            dep_vals = ", ".join(
                f"{d.replace('_', ' ')}={step['upstream_snapshot'].get(d)}"
                for d in step["depends_on"]
            )
            dep_html = f'<p style="margin:4px 0 0 0;font-size:13px;color:#6c757d;">Influenced by: {dep_vals}</p>'

        st.markdown(f"""
            <div class="stage-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <small style="color:#6c757d;">Step {i}</small>
                        <h4 style="margin:4px 0;">{step['stage'].replace('_', ' ')}</h4>
                        <p style="margin:0;font-size:14px;">{step['description']}</p>
                        {dep_html}
                    </div>
                    <div style="text-align:right;">
                        <h3 style="margin:0;color:{color};">{label}</h3>
                        <small>raw: {step['value']}</small>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("Treatment implications")
    preds = result["predictions"]
    implications = []
    if preds.get("BP_Medication_Recommendation") == 1:
        implications.append("Consider antihypertensive therapy (stage/chronic risk elevated).")
    if preds.get("Hypertensive_Crisis_Risk") == 1:
        implications.append("Monitor for hypertensive crisis — projected BP trajectory is concerning.")
    if preds.get("Emergency_Visit_Risk") == 1:
        implications.append("Elevated ER visit risk — evaluate urgently.")
    if preds.get("Stroke_Risk") == 1:
        implications.append("Stroke risk flagged — neurological assessment advised.")
    if preds.get("Heart_Attack_Risk") == 1:
        implications.append("Cardiac event risk — cardiology follow-up recommended.")
    if not implications:
        implications.append("No acute interventions indicated. Continue routine monitoring.")

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

if db_ok:
    st.sidebar.success("SQL Server connected")
else:
    st.sidebar.error("SQL Server offline")
    st.sidebar.caption(db_message)

if models:
    st.sidebar.success(f"{len(models)} linked models loaded")
else:
    st.sidebar.error("Run `python main_engine.py` to train models.")

st.sidebar.markdown("---")
public_url_path = os.path.join(os.path.dirname(__file__), "public_url.txt")
if os.path.exists(public_url_path):
    with open(public_url_path, encoding="utf-8") as f:
        public_share_url = f.read().strip()
    st.sidebar.markdown("**Public link (phone / internet)**")
    st.sidebar.code(public_share_url, language=None)
    st.sidebar.success("Use this URL on your phone — not localhost.")
    st.sidebar.caption("From `run_public.bat`. Link works while that terminal stays open.")
else:
    st.sidebar.markdown("**Local network share**")
    st.sidebar.code("http://<your-ip>:8501", language=None)
    st.sidebar.caption("Run `run_public.bat` for a phone/internet link anyone can open.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Patient Journey",
    "Research Log",
    "Pipeline Map",
    "Model Metrics",
])

with tab1:
    st.title("Interconnected Patient Assessment")
    st.caption("Loads patients from SQL Server and saves each run to the research audit tables.")

    input_mode = st.radio(
        "Input source",
        ["Manual entry", "Load from database"],
        horizontal=True,
        disabled=not db_ok,
    )

    selected_patient_id = None
    region_id = 1

    if input_mode == "Load from database" and db_ok:
        try:
            patients_df = load_patients_from_db()
            if patients_df.empty:
                st.warning("No patients in database. Run data_generator.py first.")
            else:
                options = patients_df["PatientID"].tolist()
                selected_patient_id = st.selectbox("Select patient", options)
                row = patients_df[patients_df["PatientID"] == selected_patient_id].iloc[0]
                default_age = int(row["Age"])
                default_gender = row["Gender"] if isinstance(row["Gender"], str) else ("Male" if row["Gender"] == 0 else "Female")
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
            age = st.number_input("Age", 1, 110, default_age)
            gender = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if default_gender == "Male" else 1)
        with c2:
            systolic = st.number_input("Avg Systolic (mmHg)", 80, 220, int(default_sys))
            diastolic = st.number_input("Avg Diastolic (mmHg)", 50, 140, int(default_dia))
        with c3:
            volatility = st.number_input("BP Volatility (SD)", 0.0, 40.0, float(default_vol))
            pulse = st.number_input("Pulse Pressure", 20, 120, int(default_pulse))
            readings = st.number_input("Reading count", 1, 100, default_readings)

    rc1, rc2 = st.columns(2)
    with rc1:
        researcher_name = st.text_input("Researcher name (optional)", placeholder="e.g. Dr. Smith")
    with rc2:
        research_notes = st.text_input("Research notes (optional)", placeholder="e.g. control group trial")

    if st.button("Run full clinical pipeline", use_container_width=True, type="primary"):
        if not models:
            st.error("No trained models found. Run main_engine.py first.")
        else:
            patient, gender_label = build_patient_row(
                age, gender, systolic, diastolic, volatility, pulse, readings, region_id
            )
            result = assess_patient(patient, models)
            render_assessment_result(result)

            if db_ok:
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
                    st.info("Run sql/add_research_tables.sql in SSMS if tables are missing.")
            else:
                st.warning("Database offline — results not saved to research tables.")

with tab2:
    st.title("Research tracking")
    st.caption("Every dashboard assessment is stored in `Research_Assessments` and `Research_Pipeline_Steps`.")

    if not db_ok:
        st.error("Connect to SQL Server to view research history.")
    else:
        try:
            summary = fetch_research_summary()
            if summary.get("Total_Assessments", 0) or 0 > 0:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total assessments", int(summary["Total_Assessments"]))
                s2.metric("Alerts flagged", int(summary["Alert_Count"] or 0))
                s3.metric("Avg risk score", f"{summary['Avg_Risk_Score']:.2f}" if summary["Avg_Risk_Score"] else "—")
                s4.metric("Unique patients", int(summary["Unique_Patients"] or 0))

            history = fetch_research_history(limit=200)
            if history.empty:
                st.info("No research assessments yet. Run a pipeline assessment on the Patient Journey tab.")
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
            st.subheader("Model registry (from SQL)")
            registry = fetch_model_registry()
            if registry.empty:
                st.caption("No models in Model_Registry. Run main_engine.py.")
            else:
                st.dataframe(registry, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not load research data: {exc}")
            st.info("Run `sql/add_research_tables.sql` in SSMS on HealthAI_Project.")

with tab3:
    st.title("How stages connect")
    if config:
        flow_rows = [{
            "Stage": s["name"].replace("_", " "),
            "Type": s["type"],
            "Depends on": ", ".join(d.replace("_", " ") for d in s["depends_on"]) or "Patient vitals only",
            "Description": s["description"],
        } for s in PIPELINE_STAGES]
        st.dataframe(pd.DataFrame(flow_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Dependency diagram")
        dot_lines = [
            "digraph clinical_pipeline {",
            '    rankdir=TB; node [shape=box style=rounded];',
            '    Vitals [label="Patient Vitals" fillcolor="#e3f2fd" style="filled,rounded"];',
            '    Overall [label="Overall Risk\\n(aggregated)" fillcolor="#fce4ec" style="filled,rounded"];',
        ]
        for stage in PIPELINE_STAGES:
            safe = stage["name"]
            if not stage["depends_on"]:
                dot_lines.append(f"    Vitals -> {safe};")
            for dep in stage["depends_on"]:
                dot_lines.append(f"    {dep} -> {safe};")
        for src in ("Hypertensive_Crisis_Risk", "Emergency_Visit_Risk", "Stroke_Risk", "BP_Stage"):
            dot_lines.append(f"    {src} -> Overall;")
        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines))
    else:
        st.warning("Pipeline config not found. Train models first.")

with tab4:
    st.title("Model performance")
    if metadata:
        selected = st.selectbox(
            "Select stage",
            [s["name"] for s in PIPELINE_STAGES],
            format_func=lambda x: x.replace("_", " "),
        )
        meta = metadata.get(selected, {})
        if meta:
            st.caption(meta.get("description", ""))
            if meta.get("depends_on"):
                st.markdown(f"**Upstream inputs:** {', '.join(meta['depends_on'])}")

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
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC {auc(fpr, tpr):.3f}", line=dict(color="#007bff", width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray")))
                    fig_roc.update_layout(title="ROC Curve", template="plotly_white")
                    c1.plotly_chart(fig_roc, use_container_width=True)

                    p, r, _ = precision_recall_curve(y_test, y_probs)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=r, y=p, line=dict(color="#28a745", width=2)))
                    fig_pr.update_layout(title="Precision-Recall", template="plotly_white")
                    c2.plotly_chart(fig_pr, use_container_width=True)
            elif "rmse" in meta:
                st.metric("RMSE", f"{meta['rmse']:.4f}")

        st.markdown("---")
        rows = []
        for name, m in metadata.items():
            row = {"Stage": name.replace("_", " "), "Depends on": ", ".join(m.get("depends_on", []))}
            if "accuracy" in m:
                row["Type"] = m.get("stage_type", "classification")
                row["Score"] = f"{m['accuracy']:.1%}"
            elif "rmse" in m:
                row["Type"] = "regression"
                row["Score"] = f"RMSE {m['rmse']:.3f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No evaluation metadata. Run main_engine.py.")
