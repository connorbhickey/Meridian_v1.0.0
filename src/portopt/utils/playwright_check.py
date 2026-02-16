"""Playwright browser detection and installation helpers."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def is_playwright_firefox_installed() -> bool:
    """Check if Playwright's bundled Firefox browser is installed."""
    local_app = os.environ.get("LOCALAPPDATA", "")
    if not local_app:
        # Non-Windows: check default Playwright path
        pw_dir = Path.home() / ".cache" / "ms-playwright"
    else:
        pw_dir = Path(local_app) / "ms-playwright"

    if not pw_dir.exists():
        return False

    # Playwright stores browsers as firefox-<version> directories
    return any(d.name.startswith("firefox-") for d in pw_dir.iterdir() if d.is_dir())


def install_playwright_firefox() -> tuple[bool, str]:
    """Install Playwright Firefox browser.

    Returns (success, output_text).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "firefox"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        if success:
            logger.info("Playwright Firefox installed successfully")
        else:
            logger.error("Playwright Firefox install failed: %s", output)
        return success, output.strip()
    except Exception as e:
        msg = f"Failed to install Playwright Firefox: {e}"
        logger.error(msg)
        return False, msg


def is_fidelity_api_installed() -> bool:
    """Check if the fidelity-api package is installed."""
    try:
        import importlib.util
        return importlib.util.find_spec("fidelity") is not None
    except Exception:
        return False
