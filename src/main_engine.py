import pandas as pd
import pyodbc

from config.db_config import get_connection_string
from src.clinical_pipeline import MODEL_DIR, ClinicalPipeline, preprocess_for_pipeline
from src.paths import CSV_PATIENTS_PATH

USE_SQL = True
CSV_PATH = str(CSV_PATIENTS_PATH)


def load_data():
    conn = None

    if USE_SQL:
        try:
            conn = pyodbc.connect(get_connection_string())
            query = """
            SELECT p.PatientID, p.Age, p.Gender, p.RegionID,
                   a.Avg_Systolic, a.Avg_Diastolic,
                   a.BP_Volatility, a.Pulse_Pressure, a.Reading_Count
            FROM Patients p
            JOIN Aggregated_Stats a ON p.PatientID = a.PatientID
            """
            df = pd.read_sql(query, conn)
            print("Loaded from SQL")
            return df, conn
        except Exception as e:
            print("SQL failed, fallback to CSV:", e)
            if conn:
                conn.close()

    df = pd.read_csv(CSV_PATH)
    print("Loaded from CSV")
    return df, None


def get_sql_model_type(entry):
    """
    SQL table Model_Registry has a CHECK constraint that only accepts:
    LogisticRegression, RandomForestClassifier, RandomForestRegressor.

    Calibrated classification models are wrapped by sklearn as CalibratedClassifierCV,
    so we map calibrated binary/multiclass models back to RandomForestClassifier
    for registry compatibility.
    """
    stage_type = entry.get("stage_type") or entry.get("type")

    # In the pipeline, classification stages can be binary or multiclass.
    if stage_type in ("binary", "multiclass", "classification"):
        return "RandomForestClassifier"

    # Regression stages use RandomForestRegressor.
    if stage_type == "regression":
        return "RandomForestRegressor"

    # Fallback from actual model class name.
    raw_type = type(entry.get("model")).__name__

    if raw_type == "CalibratedClassifierCV":
        return "RandomForestClassifier"

    if raw_type in ("LogisticRegression", "RandomForestClassifier", "RandomForestRegressor"):
        return raw_type

    # Safe fallback for SQL constraint.
    return "RandomForestRegressor"


def get_metric_values(entry):
    """
    Pull stored evaluation metrics when available.
    Keeps SQL inserts valid even if a metric is missing.
    """
    metadata = entry.get("metadata", {}) or {}

    accuracy = metadata.get("accuracy", 0.0)
    rmse = metadata.get("rmse", 0.0)

    try:
        accuracy = float(accuracy)
    except Exception:
        accuracy = 0.0

    try:
        rmse = float(rmse)
    except Exception:
        rmse = 0.0

    # SQL checks require accuracy between 0 and 1 and RMSE >= 0.
    accuracy = max(0.0, min(1.0, accuracy))
    rmse = max(0.0, rmse)

    return accuracy, rmse


if __name__ == "__main__":
    df_raw, conn = load_data()
    df = preprocess_for_pipeline(df_raw)

    pipeline = ClinicalPipeline(df)

    print("\nTraining interconnected pipeline (14 models + aggregated overall risk)...")
    pipeline.train_all()

    if conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Model_Registry WHERE Version='v2'")
        conn.commit()

        print("\nSaving model registry to SQL...")
        for name, entry in pipeline.models.items():
            model_type = get_sql_model_type(entry)
            accuracy, rmse = get_metric_values(entry)
            deps = ", ".join(entry.get("depends_on", [])) or "vitals only"

            cursor.execute(
                """
                INSERT INTO Model_Registry (ModelName, ModelType, Version, Accuracy, RMSE)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, model_type, "v2", accuracy, rmse),
            )

            print(f"  {name} ({model_type}, depends on: {deps})")

        conn.commit()
        conn.close()
        print("\nPipeline models registered in database")

    print(f"\nDone — {len(pipeline.models)} interconnected models in '{MODEL_DIR}/'")
