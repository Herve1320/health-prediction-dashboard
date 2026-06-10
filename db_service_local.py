"""
Database access for the dashboard and research tracking.
"""

import json
from contextlib import contextmanager

import pandas as pd
import pyodbc

from db_config import get_connection_string

PIPELINE_VERSION = "v2"


@contextmanager
def get_connection():
    conn = pyodbc.connect(get_connection_string())
    try:
        yield conn
    finally:
        conn.close()


def test_connection():
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return True, "Connected to SQL Server"
    except Exception as exc:
        return False, str(exc)


def fetch_patients():
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


def fetch_model_registry():
    query = """
    SELECT ModelID, ModelName, ModelType, Version, TrainingDate, Accuracy, RMSE
    FROM Model_Registry
    ORDER BY TrainingDate DESC, ModelName
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


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
    """
    Persist one full pipeline run to Research_Assessments + Research_Pipeline_Steps.
    Returns AssessmentID or None on failure.
    """
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
                    json.dumps(step.get("upstream_snapshot", {}), default=lambda o: int(o) if hasattr(o, "item") else o),
                ),
            )

        conn.commit()
        return assessment_id


def fetch_research_history(limit=100):
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


def fetch_assessment_steps(assessment_id):
    query = """
    SELECT StepOrder, StageName, StageType, Predicted_Value, Depends_On_JSON
    FROM Research_Pipeline_Steps
    WHERE AssessmentID = ?
    ORDER BY StepOrder
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=[assessment_id])


def fetch_research_summary():
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
