import random
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.db_config import get_connection_string
from src.paths import DATA_DIR

# ============================================
# CONFIG
# ============================================
NUM_PATIENTS = 1200
DAYS_HISTORY = 90
OUTPUT_DIR = str(DATA_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# DATABASE CONNECTION
# ============================================
conn = pyodbc.connect(get_connection_string())
cursor = conn.cursor()

# ============================================
# CLEAN DATABASE
# ============================================
print("Cleaning tables...")

tables = [
    "Research_Pipeline_Steps", "Research_Assessments",
    "Prediction_Results", "Model_Registry", "Emergency_Logs",
    "Medication_Records", "Clinical_Events", "Aggregated_Stats",
    "Biometric_History", "BloodPressure_Logs", "Patients", "Geographic_Regions",
]

for t in tables:
    cursor.execute(f"DELETE FROM {t}")

cursor.execute("DBCC CHECKIDENT ('Geographic_Regions', RESEED, 0)")
cursor.execute("DBCC CHECKIDENT ('Patients', RESEED, 0)")
conn.commit()

# ============================================
# INSERT REGIONS
# ============================================
regions = [
    ("Urban", 0.8, 70),
    ("Suburban", 0.6, 40),
    ("Rural", 0.4, 20)
]

for r in regions:
    cursor.execute("""
        INSERT INTO Geographic_Regions (RegionName, SocioEconomicIndex, PollutionLevel)
        VALUES (?, ?, ?)
    """, r)

conn.commit()

cursor.execute("SELECT RegionID FROM Geographic_Regions")
region_ids = [r[0] for r in cursor.fetchall()]

# ============================================
# INSERT PATIENTS
# ============================================
print("Generating patients...")

patients_data = []

for _ in range(NUM_PATIENTS):
    age = random.randint(18, 90)
    gender = random.choice(['Male', 'Female'])
    region_id = random.choice(region_ids)

    cursor.execute("""
        INSERT INTO Patients (Age, Gender, RegionID)
        VALUES (?, ?, ?)
    """, (age, gender, region_id))

conn.commit()

cursor.execute("SELECT PatientID, Age, Gender, RegionID FROM Patients")
patients = cursor.fetchall()

# ============================================
# GENERATION LOOP
# ============================================
print("Generating realistic medical data...")

for p in patients:
    pid, age, gender, region_id = p

    # Patient profile
    profile = random.choices(
        ["healthy", "moderate", "high"],
        weights=[0.4, 0.35, 0.25]
    )[0]

    if profile == "healthy":
        base_sys = random.randint(100, 120)
        base_dia = random.randint(65, 80)
        volatility = random.uniform(5, 10)

    elif profile == "moderate":
        base_sys = random.randint(120, 140)
        base_dia = random.randint(75, 90)
        volatility = random.uniform(8, 15)

    else:
        base_sys = random.randint(140, 170)
        base_dia = random.randint(85, 105)
        volatility = random.uniform(10, 20)

    base_weight = random.uniform(55, 110)
    height = random.uniform(1.55, 1.9)

    systolic_values = []
    diastolic_values = []

    for d in range(DAYS_HISTORY):
        date = datetime.now() - timedelta(days=d)

        trend = np.sin(d / 10) * 2

        weight = base_weight + random.uniform(-2, 2)
        bmi = weight / (height ** 2)

        systolic = int(np.random.normal(base_sys + trend + (bmi/3), volatility))
        diastolic = int(np.random.normal(base_dia + trend + (bmi/4), volatility/2))
        pulse = random.randint(60, 100)

        # ✅ FIX: CLAMP VALUES (PREVENT SQL ERRORS)
        systolic = max(90, min(systolic, 200))
        diastolic = max(60, min(diastolic, 120))

        systolic_values.append(systolic)
        diastolic_values.append(diastolic)

        cursor.execute("""
            INSERT INTO BloodPressure_Logs
            (PatientID, ReadingDate, Systolic, Diastolic, Pulse)
            VALUES (?, ?, ?, ?, ?)
        """, (pid, date, systolic, diastolic, pulse))

        cursor.execute("""
            INSERT INTO Biometric_History
            (PatientID, RecordDate, Weight, BMI)
            VALUES (?, ?, ?, ?)
        """, (pid, date, weight, round(bmi,2)))

    # Aggregation
    avg_sys = np.mean(systolic_values)
    avg_dia = np.mean(diastolic_values)
    bp_vol = np.std(systolic_values)
    pulse_pressure = avg_sys - avg_dia

    cursor.execute("""
        INSERT INTO Aggregated_Stats
        (PatientID, Avg_Systolic, Avg_Diastolic, BP_Volatility, Pulse_Pressure, Reading_Count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (pid, avg_sys, avg_dia, bp_vol, pulse_pressure, DAYS_HISTORY))

    # Clinical events
    if avg_sys > 140 and random.random() < 0.4:
        cursor.execute("""
            INSERT INTO Clinical_Events (PatientID, EventDate, EventType)
            VALUES (?, GETDATE(), 'Hypertension')
        """, (pid,))
    # Medication
    if avg_sys > 135 and random.random() < 0.5:
        cursor.execute("""
            INSERT INTO Medication_Records
            (PatientID, MedicationName, Dosage, StartDate)
            VALUES (?, 'Amlodipine', '5mg', GETDATE())
        """, (pid,))
    # Emergency
    if avg_sys > 150 and random.random() < 0.3:
        severity = random.choice(['Medium','High'])
        cursor.execute("""
            INSERT INTO Emergency_Logs
            (PatientID, VisitDate, Reason, Severity)
            VALUES (?, GETDATE(), 'Hypertensive Crisis', ?)
        """, (pid, severity))
conn.commit()

# ============================================
# EXPORT CSV
# ============================================
print("Exporting CSV...")

patients_df = pd.read_sql("SELECT * FROM Patients", conn)
agg_df = pd.read_sql("SELECT * FROM Aggregated_Stats", conn)

merged = patients_df.merge(agg_df, on="PatientID")
merged.to_csv(f"{OUTPUT_DIR}/ml_dataset.csv", index=False)

print("✅ DATA GENERATION COMPLETE")
print("📁 CSV saved in:", OUTPUT_DIR)



