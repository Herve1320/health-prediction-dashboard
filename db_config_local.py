"""
Shared SQL Server connection settings.

Override via environment variables (recommended for deployment):
  DB_SERVER, DB_NAME, DB_DRIVER, DB_TRUSTED_CONNECTION, DB_USER, DB_PASSWORD
"""

import os

DB_SERVER = os.getenv("DB_SERVER", "DESKTOP-MGMKQLP")
DB_NAME = os.getenv("DB_NAME", "HealthAI_Project")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED = os.getenv("DB_TRUSTED_CONNECTION", "yes").lower() in ("1", "true", "yes")

# Optional simple password gate for the public dashboard (leave empty to disable)
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

# Public tunnel name (used in URL: healt-prediction.ngrok-free.app)
PUBLIC_URL_NAME = os.getenv("PUBLIC_URL_NAME", "healt-prediction").lower().replace("_", "-")


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
    return "".join(parts)
