"""
Cloud-ready shared configuration.

Local mode:
- Uses SQL Server if pyodbc and the ODBC driver are available.

Cloud mode:
- Uses CSV/model files and does not require SQL Server.
- Reads DASHBOARD_PASSWORD from environment variables or Streamlit secrets.
"""

import os


def _get_secret(name: str, default: str = "") -> str:
    """Read from environment first, then Streamlit secrets when available."""
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


DB_SERVER = os.getenv("DB_SERVER", "DESKTOP-MGMKQLP")
DB_NAME = os.getenv("DB_NAME", "HealthAI_Project")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED = os.getenv("DB_TRUSTED_CONNECTION", "yes").lower() in ("1", "true", "yes")

# Optional simple password gate for the public dashboard.
DASHBOARD_PASSWORD = _get_secret("DASHBOARD_PASSWORD", "")

# Force cloud CSV mode when deployed.
APP_MODE = _get_secret("APP_MODE", os.getenv("APP_MODE", "auto")).lower()

# CSV used when SQL Server is unavailable.
CSV_PATIENTS_PATH = _get_secret("CSV_PATIENTS_PATH", "generated_data/ml_dataset.csv")


def get_connection_string():
    parts = [
        f"DRIVER={{{DB_DRIVER}}};",
        f"SERVER={DB_SERVER};",
        f"DATABASE={DB_NAME};",
    ]

    if DB_TRUSTED:
        parts.append("Trusted_Connection=yes;")
    else:
        user = os.getenv("DB_USER", "")
        password = os.getenv("DB_PASSWORD", "")
        if not user or not password:
            raise ValueError("DB_USER and DB_PASSWORD required when Trusted_Connection is disabled")
        parts.append(f"UID={user};PWD={password};")

    return """.join(parts)
