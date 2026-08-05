"""Project root and asset paths (local dev and Streamlit Cloud)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CSV_PATIENTS_PATH = DATA_DIR / "ml_dataset.csv"

# String form for os.path.join compatibility in older code paths.
MODEL_DIR = str(MODELS_DIR)
