"""Encrypted credential storage using the OS keyring (Windows Credential Manager)."""

from __future__ import annotations

import logging

import keyring

logger = logging.getLogger(__name__)

SERVICE_NAME = "Meridian"


def store_credential(key: str, value: str):
    """Store a credential securely in the OS keyring."""
    try:
        keyring.set_password(SERVICE_NAME, key, value)
    except Exception as e:
        logger.error("Failed to store credential '%s': %s", key, e)
        raise


def get_credential(key: str) -> str | None:
    """Retrieve a credential from the OS keyring."""
    try:
        return keyring.get_password(SERVICE_NAME, key)
    except Exception as e:
        logger.warning("Failed to retrieve credential '%s': %s", key, e)
        return None


def delete_credential(key: str):
    """Delete a credential from the OS keyring."""
    try:
        keyring.delete_password(SERVICE_NAME, key)
    except keyring.errors.PasswordDeleteError:
        pass  # Already gone
    except Exception as e:
        logger.warning("Failed to delete credential '%s': %s", key, e)


def has_credential(key: str) -> bool:
    """Check if a credential exists."""
    return get_credential(key) is not None


# Convenience keys
FIDELITY_USERNAME = "fidelity_username"
FIDELITY_PASSWORD = "fidelity_password"
FIDELITY_TOTP_SECRET = "fidelity_totp_secret"
ANTHROPIC_API_KEY = "anthropic_api_key"
FRED_API_KEY = "fred_api_key"
TIINGO_API_KEY = "tiingo_api_key"
ALPHA_VANTAGE_API_KEY = "alpha_vantage_api_key"
