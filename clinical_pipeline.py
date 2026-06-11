"""
Interconnected clinical decision pipeline.

Models run in dependency order: each stage uses patient vitals plus outputs
from earlier stages, mirroring how BP classification, chronic risk, treatment,
and acute events relate in real clinical practice.
"""

import os
import warnings

# Keep training output clean.
# This suppresses repeated sklearn/joblib warnings during calibration and model fitting.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
MODEL_DIR = "models"

BASE_VITALS = [
    "Age", "Gender", "RegionID",
    "Avg_Systolic", "Avg_Diastolic",
    "BP_Volatility", "Pulse_Pressure", "Reading_Count",
]

RISK_FACTORS = [
    "Age", "Gender", "RegionID",
    "BP_Volatility", "Pulse_Pressure", "Reading_Count",
]

# Pipeline definition: order matters; depends_on must appear earlier in the list.
PIPELINE_STAGES = [
    {
        "name": "BP_Stage",
        "type": "multiclass",
        "depends_on": [],
        "features": BASE_VITALS,
        "description": "Classify blood pressure stage from current vitals",
    },
    {
        "name": "Health_Risk_Tier",
        "type": "multiclass",
        "depends_on": ["BP_Stage"],
        "features": RISK_FACTORS,
        "description": "Overall health tier influenced by BP stage and volatility",
    },
    {
        "name": "Future_Systolic",
        "type": "regression",
        "depends_on": ["BP_Stage"],
        "features": BASE_VITALS,
        "description": "Forecast systolic BP based on current readings and stage",
    },
    {
        "name": "Future_Diastolic",
        "type": "regression",
        "depends_on": ["BP_Stage", "Future_Systolic"],
        "features": BASE_VITALS,
        "description": "Forecast diastolic BP using stage and projected systolic",
    },
    {
        "name": "Chronic_Hypertension_Development",
        "type": "binary",
        "depends_on": ["BP_Stage", "Health_Risk_Tier"],
        "features": RISK_FACTORS,
        "description": "Long-term hypertension risk driven by sustained elevated BP",
    },
    {
        "name": "Hypertensive_Event",
        "type": "binary",
        "depends_on": ["BP_Stage"],
        "features": RISK_FACTORS,
        "description": "Active hypertensive event linked to current BP stage",
    },
    {
        "name": "Hypertensive_Crisis_Risk",
        "type": "binary",
        "depends_on": ["BP_Stage", "Future_Systolic"],
        "features": RISK_FACTORS,
        "description": "Crisis risk from stage plus projected systolic trajectory",
    },
    {
        "name": "BP_Medication_Recommendation",
        "type": "binary",
        "depends_on": ["BP_Stage", "Chronic_Hypertension_Development", "Health_Risk_Tier"],
        "features": RISK_FACTORS,
        "description": "Medication need based on stage, chronic risk, and health tier",
    },
    {
        "name": "Cardiovascular_Event_Risk",
        "type": "binary",
        "depends_on": ["BP_Stage", "Health_Risk_Tier"],
        "features": RISK_FACTORS,
        "description": "Cardiovascular strain from BP stage and hemodynamic instability",
    },
    {
        "name": "Stroke_Risk",
        "type": "binary",
        "depends_on": ["Chronic_Hypertension_Development", "Hypertensive_Crisis_Risk", "Cardiovascular_Event_Risk"],
        "features": RISK_FACTORS,
        "description": "Stroke risk compounds chronic, crisis, and cardiovascular factors",
    },
    {
        "name": "Heart_Attack_Risk",
        "type": "binary",
        "depends_on": ["Cardiovascular_Event_Risk", "BP_Stage", "Hypertensive_Crisis_Risk"],
        "features": RISK_FACTORS,
        "description": "Heart attack risk from cardiovascular load and acute pressure spikes",
    },
    {
        "name": "Emergency_Visit_Risk",
        "type": "binary",
        "depends_on": ["Hypertensive_Crisis_Risk", "Future_Systolic", "Stroke_Risk"],
        "features": RISK_FACTORS,
        "description": "ER visit likelihood from crisis trajectory and stroke alert",
    },
    {
        "name": "Risk_Score",
        "type": "regression",
        "depends_on": ["BP_Stage", "Health_Risk_Tier", "Future_Systolic"],
        "features": BASE_VITALS,
        "description": "Composite numeric risk score from stage, tier, and forecast",
    },
    {
        "name": "Probability_Hypertension",
        "type": "regression",
        "depends_on": ["BP_Stage", "Chronic_Hypertension_Development", "Risk_Score"],
        "features": BASE_VITALS,
        "description": "Probability estimate informed by chronic progression and risk score",
    },
]

STAGE_NAMES = {s["name"]: s for s in PIPELINE_STAGES}


def bp_stage_label(systolic, diastolic=None):
    """
    Blood pressure stage logic using systolic and diastolic values.

    0 = Normal
    1 = Elevated
    2 = Hypertension Stage 1
    3 = Hypertension Stage 2
    """
    systolic = float(systolic)

    if diastolic is None:
        diastolic = 0.0
    else:
        diastolic = float(diastolic)

    if systolic >= 140 or diastolic >= 90:
        return 3

    if systolic >= 130 or diastolic >= 80:
        return 2

    if systolic >= 120 and diastolic < 80:
        return 1

    return 0


def health_risk_tier_label(bp_stage, volatility):
    if bp_stage <= 0 and volatility < 10:
        return 0
    if bp_stage <= 1:
        return 1
    return 2


def build_cascading_labels(df):
    """Create interconnected targets where downstream labels depend on upstream clinical state."""
    df = df.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    df["BP_Stage"] = df.apply(
        lambda r: bp_stage_label(r["Avg_Systolic"], r["Avg_Diastolic"]),
        axis=1,
    )
    df["Health_Risk_Tier"] = df.apply(
        lambda r: health_risk_tier_label(r["BP_Stage"], r["BP_Volatility"]), axis=1
    )

    stage_boost = df["BP_Stage"] * rng.normal(4, 2, len(df))
    df["Future_Systolic"] = df["Avg_Systolic"] + stage_boost + rng.normal(2, 3, len(df))
    df["Future_Diastolic"] = (
        df["Avg_Diastolic"]
        + df["BP_Stage"] * rng.normal(2, 1, len(df))
        + rng.normal(1, 2, len(df))
    )

    df["Chronic_Hypertension_Development"] = (
        (df["BP_Stage"] >= 2) & (df["BP_Volatility"] > 10) & (df["Health_Risk_Tier"] >= 1)
    ).astype(int)

    df["Hypertensive_Event"] = (df["BP_Stage"] >= 2).astype(int)

    df["Hypertensive_Crisis_Risk"] = (
        (df["BP_Stage"] >= 3) | (df["Future_Systolic"] > 160)
    ).astype(int)

    df["BP_Medication_Recommendation"] = (
        (df["BP_Stage"] >= 2)
        | (df["Chronic_Hypertension_Development"] == 1)
        | (df["Health_Risk_Tier"] >= 2)
    ).astype(int)

    df["Cardiovascular_Event_Risk"] = (
        (df["BP_Stage"] >= 2)
        | (df["BP_Volatility"] > 15)
        | (df["Pulse_Pressure"] > 60)
        | (df["Health_Risk_Tier"] >= 2)
    ).astype(int)

    df["Stroke_Risk"] = (
        (df["Chronic_Hypertension_Development"] == 1)
        & (
            (df["Hypertensive_Crisis_Risk"] == 1)
            | (df["Cardiovascular_Event_Risk"] == 1)
        )
    ).astype(int)

    df["Heart_Attack_Risk"] = (
        (df["Cardiovascular_Event_Risk"] == 1)
        & (df["Pulse_Pressure"] > 55)
        & ((df["BP_Stage"] >= 2) | (df["Hypertensive_Crisis_Risk"] == 1))
    ).astype(int)

    df["Emergency_Visit_Risk"] = (
        (df["Hypertensive_Crisis_Risk"] == 1)
        | (df["Future_Systolic"] > 155)
        | (df["Stroke_Risk"] == 1)
    ).astype(int)

    df["Risk_Score"] = (
        df["Avg_Systolic"] * 0.4
        + df["Pulse_Pressure"] * 0.3
        + df["BP_Stage"] * 15
        + df["Health_Risk_Tier"] * 10
        + df["Future_Systolic"] * 0.1
    )

    chronic_factor = df["Chronic_Hypertension_Development"] * 0.15
    df["Probability_Hypertension"] = np.clip(
        df["Avg_Systolic"] / 200 + df["BP_Stage"] * 0.08 + chronic_factor + df["Risk_Score"] / 500,
        0,
        1,
    )

    return df


def apply_clinical_consistency_guardrails(predictions, patient_row):
    """
    Adjust clearly inconsistent downstream signals before the final score is computed.

    Normal BP + low volatility + normal pulse pressure should not activate crisis,
    emergency, stroke, or cardiac acute-risk signals only because of model noise.
    """
    if patient_row is None:
        return predictions

    predictions = dict(predictions)

    systolic = float(patient_row.get("Avg_Systolic", 0))
    diastolic = float(patient_row.get("Avg_Diastolic", 0))
    volatility = float(patient_row.get("BP_Volatility", 0))
    pulse_pressure = float(patient_row.get("Pulse_Pressure", 0))
    age = int(patient_row.get("Age", 0))

    normal_bp_profile = (
        systolic < 120
        and diastolic < 80
        and volatility < 10
        and pulse_pressure < 50
    )

    mild_bp_profile = (
        systolic < 130
        and diastolic < 85
        and volatility < 12
        and pulse_pressure < 55
    )

    if normal_bp_profile:
        predictions["BP_Stage"] = 0
        predictions["Health_Risk_Tier"] = 0

        for key in [
            "Hypertensive_Event",
            "Hypertensive_Crisis_Risk",
            "BP_Medication_Recommendation",
            "Cardiovascular_Event_Risk",
            "Stroke_Risk",
            "Heart_Attack_Risk",
            "Emergency_Visit_Risk",
        ]:
            if key in predictions:
                predictions[key] = 0

        if age < 60 and "Chronic_Hypertension_Development" in predictions:
            predictions["Chronic_Hypertension_Development"] = 0

    elif mild_bp_profile:
        for key in [
            "Hypertensive_Crisis_Risk",
            "Emergency_Visit_Risk",
            "Stroke_Risk",
            "Heart_Attack_Risk",
        ]:
            if key in predictions:
                predictions[key] = 0

    return predictions


def synchronize_steps_with_guardrails(steps, guarded_predictions):
    """
    Update the displayed step values so the dashboard does not show old
    pre-guardrail predictions after the clinical guardrail corrected them.
    """
    if not steps:
        return steps

    updated_steps = []
    for step in steps:
        step = dict(step)
        stage = step.get("stage")

        if stage in guarded_predictions:
            step["value"] = guarded_predictions[stage]

            # If a severe binary risk signal was suppressed, lower confidence display safely.
            if guarded_predictions[stage] == 0 and stage in [
                "Hypertensive_Event",
                "Hypertensive_Crisis_Risk",
                "BP_Medication_Recommendation",
                "Cardiovascular_Event_Risk",
                "Stroke_Risk",
                "Heart_Attack_Risk",
                "Emergency_Visit_Risk",
            ]:
                step["confidence"] = None
                step["calibrated"] = False
                step["guardrail_adjusted"] = True

        updated_steps.append(step)

    return updated_steps


def compute_overall_risk(predictions, patient_row=None):
    """
    Compute a clinically consistent Composite Pipeline Risk Score.

    This score is not a diagnosis. It is a weighted screening score.
    Guardrails prevent normal or near-normal input profiles from being escalated
    to high/critical risk only because downstream models are noisy.
    """
    bp_stage = int(predictions.get("BP_Stage", 0))
    health_tier = int(predictions.get("Health_Risk_Tier", 0))

    max_score_cap = None
    forced_level = None

    if patient_row is not None:
        systolic = float(patient_row.get("Avg_Systolic", 0))
        diastolic = float(patient_row.get("Avg_Diastolic", 0))
        volatility = float(patient_row.get("BP_Volatility", 0))
        pulse_pressure = float(patient_row.get("Pulse_Pressure", 0))
        age = int(patient_row.get("Age", 0))

        normal_bp_profile = (
            systolic < 120
            and diastolic < 80
            and volatility < 10
            and pulse_pressure < 50
        )

        mild_bp_profile = (
            systolic < 130
            and diastolic < 85
            and volatility < 12
            and pulse_pressure < 55
        )

        # Normal BP + stable readings should never become High or Critical.
        if normal_bp_profile:
            if age < 60:
                forced_level = "Low"
                max_score_cap = 1.9
            else:
                forced_level = "Moderate"
                max_score_cap = 3.9

        # Mild/elevated profiles should not become High/Critical unless raw vitals support it.
        elif mild_bp_profile:
            forced_level = "Moderate"
            max_score_cap = 3.9

    weights = {
        "Hypertensive_Crisis_Risk": 3.0,
        "Emergency_Visit_Risk": 3.0,
        "Stroke_Risk": 2.5,
        "Heart_Attack_Risk": 2.5,
        "Chronic_Hypertension_Development": 1.5,
        "Cardiovascular_Event_Risk": 1.5,
        "Hypertensive_Event": 1.0,
        "BP_Medication_Recommendation": 0.5,
    }

    score = 0.0
    score += float(bp_stage) * 0.8
    score += float(health_tier) * 0.6

    for key, weight in weights.items():
        value = predictions.get(key, 0)
        try:
            active = int(value) == 1
        except Exception:
            active = value is True

        if active:
            score += weight

    score = round(score, 2)

    if score >= 7:
        level = "Critical"
    elif score >= 4:
        level = "High"
    elif score >= 2:
        level = "Moderate"
    else:
        level = "Low"

    if max_score_cap is not None and score > max_score_cap:
        score = max_score_cap
        level = forced_level

    flag = 1 if level in ("High", "Critical") else 0

    return {
        "Overall_Risk_Flag": flag,
        "Overall_Risk_Score": round(score, 2),
        "Overall_Risk_Level": level,
    }



def _build_feature_matrix(row_or_df, stage_def, upstream_values):
    """Combine base features with upstream pipeline outputs for one stage."""
    if isinstance(row_or_df, pd.Series):
        base = row_or_df[stage_def["features"]].to_frame().T
    else:
        base = row_or_df[stage_def["features"]].copy()

    for dep in stage_def["depends_on"]:
        base[dep] = upstream_values[dep]

    return base


def _fit_with_optional_calibration(base_model, X_train, y_train, stage_type):
    """Fit classifiers with probability calibration when data is sufficient."""
    if stage_type not in ("binary", "multiclass"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model.fit(X_train, y_train)
        return base_model, False

    y_series = pd.Series(y_train)
    class_counts = y_series.value_counts()

    # Need at least 2 classes and enough samples per class for 3-fold calibration.
    if len(class_counts) < 2 or class_counts.min() < 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model.fit(X_train, y_train)
        return base_model, False

    try:
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated.fit(X_train, y_train)
        return calibrated, True
    except TypeError:
        # Compatibility fallback for older sklearn versions.
        calibrated = CalibratedClassifierCV(
            base_estimator=base_model,
            method="sigmoid",
            cv=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated.fit(X_train, y_train)
        return calibrated, True
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model.fit(X_train, y_train)
        return base_model, False


def _prediction_confidence(model, X, raw):
    """Return predicted-class probability for classifiers when available."""
    if not hasattr(model, "predict_proba"):
        return None, None

    try:
        probs = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", range(len(probs))))
        if raw in classes:
            idx = classes.index(raw)
        else:
            idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        prob_map = {str(classes[i]): float(probs[i]) for i in range(len(probs))}
        return confidence, prob_map
    except Exception:
        return None, None


def _regression_interval(pred_value, rmse):
    """Approximate 95% prediction interval using evaluation RMSE when available."""
    if rmse is None:
        return None
    try:
        pred_value = float(pred_value)
        rmse = float(rmse)
        return {
            "lower": round(pred_value - 1.96 * rmse, 2),
            "upper": round(pred_value + 1.96 * rmse, 2),
            "rmse": round(rmse, 4),
            "method": "approx_95_percent_interval_from_eval_rmse",
        }
    except Exception:
        return None


class ClinicalPipeline:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.stage_order = [s["name"] for s in PIPELINE_STAGES]

    def _get_model(self, stage_type):
        if stage_type == "binary":
            return LogisticRegression(max_iter=1000, solver="liblinear")
        if stage_type == "multiclass":
            return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)
        return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)

    def _save_eval_metadata(self, model, target, X_test, y_test, stage_type):
        os.makedirs(MODEL_DIR, exist_ok=True)
        eval_data = {
            "y_test": y_test.tolist(),
            "depends_on": STAGE_NAMES[target]["depends_on"],
            "stage_type": stage_type,
            "description": STAGE_NAMES[target]["description"],
        }

        if stage_type in ("binary", "multiclass"):
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                if stage_type == "binary" and probs.shape[1] > 1:
                    eval_data["y_probs"] = probs[:, 1].tolist()
                else:
                    eval_data["y_probs"] = None
                eval_data["mean_confidence"] = float(np.max(probs, axis=1).mean())
                eval_data["min_confidence"] = float(np.max(probs, axis=1).min())
            else:
                eval_data["y_probs"] = None
                eval_data["mean_confidence"] = None
                eval_data["min_confidence"] = None
            eval_data["accuracy"] = report["accuracy"]
            eval_data["precision"] = report["weighted avg"]["precision"]
            eval_data["recall"] = report["weighted avg"]["recall"]
            eval_data["f1_score"] = report["weighted avg"]["f1-score"]
        else:
            preds = model.predict(X_test)
            eval_data["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))

        joblib.dump(eval_data, os.path.join(MODEL_DIR, f"{target}_eval.pkl"))
        return eval_data

    def _training_features(self, stage_def):
        """Use ground-truth upstream labels during training."""
        upstream = {dep: self.df[dep] for dep in stage_def["depends_on"]}
        return _build_feature_matrix(self.df, stage_def, upstream)

    def train_all(self):
        for stage_def in PIPELINE_STAGES:
            name = stage_def["name"]
            X = self._training_features(stage_def)
            y = self.df[name]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )

            base_model = self._get_model(stage_def["type"])
            model, calibrated = _fit_with_optional_calibration(
                base_model, X_train, y_train, stage_def["type"]
            )

            eval_data = self._save_eval_metadata(model, name, X_test, y_test, stage_def["type"])
            self.models[name] = {
                "model": model,
                "feature_columns": list(X.columns),
                "depends_on": stage_def["depends_on"],
                "type": stage_def["type"],
                "calibrated": calibrated,
                "rmse": eval_data.get("rmse") if isinstance(eval_data, dict) else None,
            }
            print(f"  [{len(stage_def['depends_on'])} deps] {name}")

        self._save_pipeline_config()

    def _save_pipeline_config(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        config = {
            "stages": PIPELINE_STAGES,
            "stage_order": self.stage_order,
            "version": "2.0-pipeline",
        }
        joblib.dump(config, os.path.join(MODEL_DIR, "pipeline_config.pkl"))
        for name, entry in self.models.items():
            joblib.dump(entry["model"], os.path.join(MODEL_DIR, f"{name}.pkl"))

    def predict_patient(self, patient_row):
        """
        Run full interconnected assessment for one patient.
        Returns ordered steps with inputs used and downstream impact notes.
        """
        upstream = {}
        steps = []

        for stage_def in PIPELINE_STAGES:
            name = stage_def["name"]
            entry = self.models[name]
            X = _build_feature_matrix(patient_row, stage_def, upstream)
            X = X[entry["feature_columns"]]

            raw = entry["model"].predict(X)[0]
            upstream[name] = raw

            confidence, probability_map = _prediction_confidence(entry["model"], X, raw)
            interval = _regression_interval(raw, entry.get("rmse")) if stage_def["type"] == "regression" else None

            step = {
                "stage": name,
                "value": float(raw) if isinstance(raw, (np.floating, float)) else int(raw),
                "depends_on": stage_def["depends_on"],
                "upstream_snapshot": {d: upstream.get(d) for d in stage_def["depends_on"]},
                "description": stage_def["description"],
                "type": stage_def["type"],
                "confidence": confidence,
                "probabilities": probability_map,
                "prediction_interval": interval,
                "calibrated": entry.get("calibrated", False),
            }
            steps.append(step)

        upstream = apply_clinical_consistency_guardrails(upstream, patient_row)
        overall = compute_overall_risk(upstream, patient_row)
        upstream.update(overall)

        return {"steps": steps, "predictions": upstream, "overall": overall}

    def predict_all_patients(self):
        rows = []
        for _, row in self.df.iterrows():
            assessment = self.predict_patient(row)
            record = {"PatientID": row["PatientID"]}
            record.update(assessment["predictions"])
            rows.append(record)
        return pd.DataFrame(rows)


def assess_patient(patient_row, models_registry):
    """
    Run pipeline inference using models loaded from disk (for the dashboard).
    models_registry: dict from load_pipeline_models()[1]
    """
    upstream = {}
    steps = []

    for stage_def in PIPELINE_STAGES:
        name = stage_def["name"]
        entry = models_registry[name]
        X = _build_feature_matrix(patient_row, stage_def, upstream)
        feature_cols = entry["features"] + stage_def["depends_on"]
        X = X[feature_cols]

        raw = entry["model"].predict(X)[0]
        upstream[name] = raw

        confidence, probability_map = _prediction_confidence(entry["model"], X, raw)
        interval = _regression_interval(raw, entry.get("rmse")) if stage_def["type"] == "regression" else None

        steps.append({
            "stage": name,
            "value": float(raw) if isinstance(raw, (np.floating, float)) else int(raw),
            "depends_on": stage_def["depends_on"],
            "upstream_snapshot": {d: upstream.get(d) for d in stage_def["depends_on"]},
            "description": stage_def["description"],
            "type": stage_def["type"],
            "confidence": confidence,
            "probabilities": probability_map,
            "prediction_interval": interval,
            "calibrated": entry.get("calibrated", False),
        })

    upstream = apply_clinical_consistency_guardrails(upstream, patient_row)
    steps = synchronize_steps_with_guardrails(steps, upstream)
    overall = compute_overall_risk(upstream, patient_row)
    upstream.update(overall)
    return {"steps": steps, "predictions": upstream, "overall": overall}


def preprocess_for_pipeline(df):
    df = df.copy()

    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"male": 0, "m": 0, "0": 0, "female": 1, "f": 1, "1": 1})
            .fillna(0)
            .astype(int)
        )

    numeric_cols = [
        "Age", "Gender", "RegionID", "Avg_Systolic", "Avg_Diastolic",
        "BP_Volatility", "Pulse_Pressure", "Reading_Count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])

    return build_cascading_labels(df)


def load_pipeline_models():
    """Load trained models and config for the Streamlit dashboard."""
    config_path = os.path.join(MODEL_DIR, "pipeline_config.pkl")
    if not os.path.exists(config_path):
        return None, {}, {}

    config = joblib.load(config_path)
    models = {}
    metadata = {}

    for stage_def in config["stages"]:
        name = stage_def["name"]
        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        eval_path = os.path.join(MODEL_DIR, f"{name}_eval.pkl")

        eval_data = None
        if os.path.exists(eval_path):
            eval_data = joblib.load(eval_path)
            metadata[name] = eval_data

        if os.path.exists(model_path):
            models[name] = {
                "model": joblib.load(model_path),
                "depends_on": stage_def["depends_on"],
                "features": stage_def["features"],
                "type": stage_def["type"],
                "description": stage_def["description"],
                "rmse": eval_data.get("rmse") if isinstance(eval_data, dict) else None,
                "mean_confidence": eval_data.get("mean_confidence") if isinstance(eval_data, dict) else None,
                "calibrated": stage_def["type"] in ("binary", "multiclass"),
            }

    return config, models, metadata


def self_test_normal_profile():
    """
    Quick internal sanity test for normal BP profile.
    Expected: Low risk, score <= 1.9, acute risk signals suppressed.
    """
    patient_row = pd.Series({
        "Age": 20,
        "Gender": 0,
        "RegionID": 1,
        "Avg_Systolic": 115,
        "Avg_Diastolic": 75,
        "BP_Volatility": 5,
        "Pulse_Pressure": 40,
        "Reading_Count": 30,
    })

    noisy_predictions = {
        "BP_Stage": 3,
        "Health_Risk_Tier": 2,
        "Hypertensive_Event": 1,
        "Hypertensive_Crisis_Risk": 1,
        "BP_Medication_Recommendation": 1,
        "Cardiovascular_Event_Risk": 1,
        "Stroke_Risk": 1,
        "Heart_Attack_Risk": 1,
        "Emergency_Visit_Risk": 1,
    }

    guarded = apply_clinical_consistency_guardrails(noisy_predictions, patient_row)
    overall = compute_overall_risk(guarded, patient_row)
    return guarded, overall


