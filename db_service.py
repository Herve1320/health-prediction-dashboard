"""
Cloud-ready database service.

This file keeps the same function names used by app.py.

Local mode:
- If SQL Server + pyodbc are available, it uses SQL Server.

Cloud mode:
- If SQL Server is unavailable, it automatically reads generated_data/ml_dataset.csv.
- Research history is kept only in Streamlit session state, so it resets when the app restarts.
"""

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from db_config import APP_MODE, CSV_PATIENTS_PATH, get_connection_string

PIPELINE_VERSION = "v2"


def _try_import_pyodbc():
    try:
        import pyodbc
        return pyodbc
    except Exception:
        return None


def _use_sql_allowed():
    return APP_MODE not in ("cloud", "csv", "streamlit")


@contextmanager
def get_connection():
    pyodbc = _try_import_pyodbc()
    if pyodbc is None:
        raise RuntimeError("pyodbc is not available; running in cloud CSV mode.")

    conn = pyodbc.connect(get_connection_string())
    try:
        yield conn
    finally:
        conn.close()


def _csv_path():
    return Path(CSV_PATIENTS_PATH)


def _load_csv_patients():
    path = _csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Add generated_data/ml_dataset.csv to your GitHub repo."
        )

    df = pd.read_csv(path)

    expected = [
        "PatientID", "Age", "Gender", "RegionID",
        "Avg_Systolic", "Avg_Diastolic",
        "BP_Volatility", "Pulse_Pressure", "Reading_Count",
    ]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return df[expected].copy()


def test_connection():
    if _use_sql_allowed():
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            return True, "Connected to SQL Server"
        except Exception:
            pass

    try:
        rows = len(_load_csv_patients())
        return True, f"Cloud CSV mode active — loaded {rows} patients"
    except Exception as exc:
        return False, str(exc)


def fetch_patients():
    if _use_sql_allowed():
        try:
            query = """
            SELECT p.PatientID, p.Age, p.Gender, p.RegionID,
                   a.Avg_Systolic, a.Avg_Diastolic,
                   a.BP_Volatility, a.Pulse_Pressure, a.Reading_Count
            FROM Patients p
            INNER JOIN Aggregated_Stats a ON p.PatientID = a.PatientID
            ORDER BY p.PatientID
            """
            with get_connection() as conn:
                return pd.read_sql(query, conn)
        except Exception:
            pass

    return _load_csv_patients()


def fetch_model_registry():
    if _use_sql_allowed():
        try:
            query = """
            SELECT ModelID, ModelName, ModelType, Version, TrainingDate, Accuracy, RMSE
            FROM Model_Registry
            ORDER BY TrainingDate DESC, ModelName
            """
            with get_connection() as conn:
                return pd.read_sql(query, conn)
        except Exception:
            pass

    # Cloud fallback: lightweight registry from model eval files.
    model_dir = Path("models")
    rows = []
    if model_dir.exists():
        for eval_file in sorted(model_dir.glob("*_eval.pkl")):
            name = eval_file.name.replace("_eval.pkl", "")
            rows.append({
                "ModelName": name,
                "ModelType": "Saved sklearn model",
                "Version": PIPELINE_VERSION,
                "TrainingDate": "Loaded from models/",
                "Accuracy": None,
                "RMSE": None,
            })
    return pd.DataFrame(rows)


def _history_key():
    if "research_history_cloud" not in st.session_state:
        st.session_state["research_history_cloud"] = []
    if "research_steps_cloud" not in st.session_state:
        st.session_state["research_steps_cloud"] = {}
    return "research_history_cloud", "research_steps_cloud"


def save_research_assessment(
    patient_id,
    input_source,
    patient_row,
    gender_label,
    overall,
    steps,
    researcher_name=None,
    notes=None,
):
    if _use_sql_allowed():
        try:
            insert_assessment = """
            INSERT INTO Research_Assessments (
                PatientID, ResearcherName, InputSource,
                Age, Gender, RegionID,
                Avg_Systolic, Avg_Diastolic, BP_Volatility, Pulse_Pressure, Reading_Count,
                Overall_Risk_Level, Overall_Risk_Score, Overall_Risk_Flag,
                Pipeline_Version, Notes
            )
            OUTPUT INSERTED.AssessmentID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            insert_step = """
            INSERT INTO Research_Pipeline_Steps (
                AssessmentID, StepOrder, StageName, StageType,
                Predicted_Value, Depends_On_JSON
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """

            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    insert_assessment,
                    (
                        patient_id,
                        researcher_name,
                        input_source,
                        int(patient_row["Age"]),
                        gender_label,
                        int(patient_row.get("RegionID", 1)),
                        float(patient_row["Avg_Systolic"]),
                        float(patient_row["Avg_Diastolic"]),
                        float(patient_row["BP_Volatility"]),
                        float(patient_row["Pulse_Pressure"]),
                        int(patient_row["Reading_Count"]),
                        overall["Overall_Risk_Level"],
                        float(overall["Overall_Risk_Score"]),
                        int(overall["Overall_Risk_Flag"]),
                        PIPELINE_VERSION,
                        notes,
                    ),
                )
                assessment_id = cursor.fetchone()[0]

                for order, step in enumerate(steps, 1):
                    cursor.execute(
                        insert_step,
                        (
                            assessment_id,
                            order,
                            step["stage"],
                            step["type"],
                            float(step["value"]),
                            json.dumps(step.get("upstream_snapshot", {}), default=str),
                        ),
                    )

                conn.commit()
                return assessment_id
        except Exception:
            pass

    # Cloud session-only fallback.
    hist_key, steps_key = _history_key()
    assessment_id = len(st.session_state[hist_key]) + 1

    st.session_state[hist_key].insert(0, {
        "AssessmentID": assessment_id,
        "Assessed_At": datetime.now(),
        "InputSource": input_source,
        "ResearcherName": researcher_name,
        "PatientID": patient_id,
        "Age": int(patient_row["Age"]),
        "Gender": gender_label,
        "Avg_Systolic": float(patient_row["Avg_Systolic"]),
        "Avg_Diastolic": float(patient_row["Avg_Diastolic"]),
        "Overall_Risk_Level": overall["Overall_Risk_Level"],
        "Overall_Risk_Score": float(overall["Overall_Risk_Score"]),
        "Overall_Risk_Flag": int(overall["Overall_Risk_Flag"]),
        "Pipeline_Version": PIPELINE_VERSION,
        "Notes": notes,
        "Stage_Count": len(steps),
    })

    st.session_state[steps_key][assessment_id] = [
        {
            "StepOrder": i,
            "StageName": step["stage"],
            "StageType": step["type"],
            "Predicted_Value": float(step["value"]),
            "Depends_On_JSON": json.dumps(step.get("upstream_snapshot", {}), default=str),
        }
        for i, step in enumerate(steps, 1)
    ]

    return assessment_id


def fetch_research_history(limit=100):
    if _use_sql_allowed():
        try:
            query = f"""
            SELECT TOP ({int(limit)})
                a.AssessmentID,
                a.Assessed_At,
                a.InputSource,
                a.ResearcherName,
                a.PatientID,
                a.Age,
                a.Gender,
                a.Avg_Systolic,
                a.Avg_Diastolic,
                a.Overall_Risk_Level,
                a.Overall_Risk_Score,
                a.Overall_Risk_Flag,
                a.Pipeline_Version,
                a.Notes,
                (SELECT COUNT(*) FROM Research_Pipeline_Steps s WHERE s.AssessmentID = a.AssessmentID) AS Stage_Count
            FROM Research_Assessments a
            ORDER BY a.Assessed_At DESC
            """
            with get_connection() as conn:
                return pd.read_sql(query, conn)
        except Exception:
            pass

    hist_key, _ = _history_key()
    return pd.DataFrame(st.session_state[hist_key][:limit])


def fetch_assessment_steps(assessment_id):
    if _use_sql_allowed():
        try:
            query = """
            SELECT StepOrder, StageName, StageType, Predicted_Value, Depends_On_JSON
            FROM Research_Pipeline_Steps
            WHERE AssessmentID = ?
            ORDER BY StepOrder
            """
            with get_connection() as conn:
                return pd.read_sql(query, conn, params=[assessment_id])
        except Exception:
            pass

    _, steps_key = _history_key()
    return pd.DataFrame(st.session_state[steps_key].get(assessment_id, []))


def fetch_research_summary():
    if _use_sql_allowed():
        try:
            query = """
            SELECT
                COUNT(*) AS Total_Assessments,
                SUM(CASE WHEN Overall_Risk_Flag = 1 THEN 1 ELSE 0 END) AS Alert_Count,
                AVG(Overall_Risk_Score) AS Avg_Risk_Score,
                COUNT(DISTINCT PatientID) AS Unique_Patients,
                MAX(Assessed_At) AS Last_Assessment
            FROM Research_Assessments
            """
            with get_connection() as conn:
                return pd.read_sql(query, conn).iloc[0].to_dict()
        except Exception:
            pass

    hist = fetch_research_history()
    if hist.empty:
        return {
            "Total_Assessments": 0,
            "Alert_Count": 0,
            "Avg_Risk_Score": None,
            "Unique_Patients": 0,
            "Last_Assessment": None,
        }

    return {
        "Total_Assessments": len(hist),
        "Alert_Count": int(hist["Overall_Risk_Flag"].sum()),
        "Avg_Risk_Score": float(hist["Overall_Risk_Score"].mean()),
        "Unique_Patients": int(hist["PatientID"].nunique()) if "PatientID" in hist else 0,
        "Last_Assessment": hist["Assessed_At"].max(),
    }
