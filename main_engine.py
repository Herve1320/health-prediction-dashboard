import pandas as pd
import pyodbc

from clinical_pipeline import (
    MODEL_DIR,
    ClinicalPipeline,
    preprocess_for_pipeline,
)
from db_config import get_connection_string

USE_SQL = True
CSV_PATH = "generated_data/ml_dataset.csv"


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


if __name__ == "__main__":
    df_raw, conn = load_data()
    df = preprocess_for_pipeline(df_raw)

    pipeline = ClinicalPipeline(df)

    print("\nTraining interconnected pipeline (14 models + aggregated overall risk)...")
    pipeline.train_all()

    print("\nGenerating cascade predictions for all patients...")
    predictions_df = pipeline.predict_all_patients()
    predictions_df.to_csv("full_patient_predictions.csv", index=False)
    print("Predictions saved to full_patient_predictions.csv")

    if conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Model_Registry WHERE Version='v2'")
        conn.commit()

        print("\nSaving model registry to SQL...")
        for name, entry in pipeline.models.items():
            model_type = type(entry["model"]).__name__
            deps = ", ".join(entry["depends_on"]) or "vitals only"
            cursor.execute(
                """
                INSERT INTO Model_Registry (ModelName, ModelType, Version, Accuracy, RMSE)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, model_type, "v2", 0.0, 0.0),
            )
            print(f"  {name} (depends on: {deps})")

        conn.commit()
        conn.close()
        print("\nPipeline models registered in database")

    print(f"\nDone — {len(pipeline.models)} interconnected models in '{MODEL_DIR}/'")
