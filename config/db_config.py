"""
Cloud-ready shared configuration.

Local mode:
- Uses SQL Server when DB_SERVER is set and pyodbc is available.

Cloud mode:
- Uses bundled CSV/model files; no SQL Server required.
- Auto-detected on Streamlit Community Cloud (USER=appuser).
"""

import os

from src.paths import CSV_PATIENTS_PATH as DEFAULT_CSV_PATH


def _get_secret(name: str, default: str = "") -> str:
    """Read from environment first, then Streamlit secrets when available."""
    value = os.getenv(name)
    if value is not None and value != "":
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name, default)
        if secret_value is not None and str(secret_value) != "":
            return str(secret_value)
    except Exception:
        pass

    return default


def _is_streamlit_cloud() -> bool:
    """Streamlit Community Cloud runs apps as the appuser Linux user."""
    return os.getenv("USER") == "appuser"


def _resolve_app_mode() -> str:
    explicit = _get_secret("APP_MODE", os.getenv("APP_MODE", "auto")).lower().strip()

    if explicit in ("cloud", "csv", "streamlit"):
        return explicit

    if explicit == "auto":
        if _is_streamlit_cloud():
            return "cloud"
        try:
            import pyodbc  # noqa: F401
        except ImportError:
            return "cloud"
        if not os.getenv("DB_SERVER", "").strip():
            return "cloud"

    return explicit


DB_SERVER = os.getenv("DB_SERVER", "")
DB_NAME = os.getenv("DB_NAME", "HealthAI_Project")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED = os.getenv("DB_TRUSTED_CONNECTION", "yes").lower() in ("1", "true", "yes")

DASHBOARD_PASSWORD = _get_secret("DASHBOARD_PASSWORD", "")
APP_MODE = _resolve_app_mode()
CSV_PATIENTS_PATH = _get_secret("CSV_PATIENTS_PATH", str(DEFAULT_CSV_PATH))


def sql_configured() -> bool:
    return bool(DB_SERVER.strip())


def get_connection_string():
    if not sql_configured():
        raise ValueError("DB_SERVER is not configured")

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

    return "".join(parts)
