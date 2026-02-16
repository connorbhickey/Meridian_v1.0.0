"""Application configuration: QSettings persistence + .env loading."""

import os
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import QSettings

from portopt.constants import APP_NAME, APP_ORG


# Load .env from project root if it exists
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def get_settings() -> QSettings:
    """Return the global QSettings instance (INI format in user config dir)."""
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, APP_ORG, APP_NAME)


def get_data_dir() -> Path:
    """Return the local data directory for caches, state, etc."""
    base = Path(os.environ.get("PORTOPT_DATA_DIR", ""))
    if not base.is_absolute():
        base = Path.home() / ".portopt"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_cache_db_path() -> Path:
    return get_data_dir() / "cache.db"


def get_fidelity_state_dir() -> Path:
    d = get_data_dir() / "fidelity_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_alpha_vantage_key() -> str | None:
    return os.environ.get("ALPHA_VANTAGE_API_KEY") or None
