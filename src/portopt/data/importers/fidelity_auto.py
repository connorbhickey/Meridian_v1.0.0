"""Automated Fidelity account connection using fidelity-api (Playwright).

Provides persistent session login, 2FA handling, and position fetching
without requiring manual CSV downloads.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from portopt.config import get_fidelity_state_dir
from portopt.data.models import (
    AccountSummary, Asset, AssetType, Holding, Portfolio,
)

logger = logging.getLogger(__name__)

# The fidelity-api library stores its state as "Fidelity.json" inside the profile dir
_LIBRARY_STATE_FILE = "Fidelity.json"


class PlaywrightNotInstalledError(RuntimeError):
    """Raised when Playwright Firefox is not installed."""


class FidelityAutoImporter:
    """Automated Fidelity connection using fidelity-api (Playwright browser automation).

    Uses a persistent browser profile to maintain login sessions across app restarts.
    After first-time login + 2FA, subsequent launches reuse the saved session.
    """

    def __init__(self, state_dir: Path | None = None, headless: bool = True):
        self._state_dir = state_dir or get_fidelity_state_dir()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._fidelity = None
        self._logged_in = False
        self._headless = headless

    @property
    def has_saved_session(self) -> bool:
        """Check if a saved browser session exists.

        The fidelity-api library stores its state as Fidelity.json inside the profile dir.
        """
        return (self._state_dir / _LIBRARY_STATE_FILE).exists()

    def _check_playwright(self):
        """Verify Playwright Firefox is available before launching a browser."""
        from portopt.utils.playwright_check import is_playwright_firefox_installed
        if not is_playwright_firefox_installed():
            raise PlaywrightNotInstalledError(
                "Playwright Firefox is not installed. "
                "Run 'python -m playwright install firefox' or use the setup wizard."
            )

    def _create_automation(self, headless: bool = True):
        """Create a new FidelityAutomation instance with persistent profile.

        Bug fix: profile_path must be a DIRECTORY — the library appends
        'Fidelity.json' to it internally.
        """
        # Close any existing browser first to avoid orphaned processes
        if self._fidelity:
            try:
                self._fidelity.close_browser()
            except Exception:
                pass
            self._fidelity = None

        self._check_playwright()
        from fidelity.fidelity import FidelityAutomation
        self._fidelity = FidelityAutomation(
            headless=headless if headless is not None else self._headless,
            save_state=True,
            profile_path=str(self._state_dir),
        )

    def login(self, username: str, password: str, totp_secret: str | None = None) -> tuple[bool, bool]:
        """Login to Fidelity.

        Returns (success, needs_2fa):
            - (True, False): Login complete, no 2FA needed
            - (True, True): Login started, call complete_2fa() with the code
            - (False, False): Login failed

        Note: The fidelity-api library returns inverted semantics:
            - (True, True) = fully logged in (success, no 2FA needed)
            - (True, False) = needs 2FA
            - (False, False) = failed
        We map those to Meridian's convention above.
        """
        try:
            self._create_automation()
            lib_success, lib_logged_in = self._fidelity.login(
                username=username,
                password=password,
                totp_secret=totp_secret,
                save_device=False,
            )

            if lib_success and lib_logged_in:
                # Library: (True, True) = fully logged in → Meridian: (True, False)
                self._logged_in = True
                logger.info("Fidelity login successful (no 2FA needed)")
                return True, False
            elif lib_success and not lib_logged_in:
                # Library: (True, False) = needs 2FA → Meridian: (True, True)
                logger.info("Fidelity login requires 2FA code")
                return True, True
            else:
                # Library: (False, False) = failed → Meridian: (False, False)
                logger.warning("Fidelity login failed")
                return False, False
        except PlaywrightNotInstalledError:
            raise
        except Exception as e:
            logger.error("Fidelity login error: %s", e)
            raise  # Surface real error to controller/UI

    def login_interactive(self, timeout_sec: int = 300) -> bool:
        """Open a visible browser and let the user log in manually.

        Opens Firefox to Fidelity's login page. The user handles credentials
        and 2FA themselves. We poll until the browser leaves the login page,
        then navigate to the positions page ourselves.

        Returns True if the user successfully logged in.
        """
        try:
            self._create_automation(headless=False)
            self._fidelity.page.goto(
                "https://digital.fidelity.com/prgw/digital/login/full-page",
                timeout=60000,
            )
            logger.info("Interactive login: browser opened, waiting for user to log in...")

            # Poll until the user leaves the login page.
            # After successful login, Fidelity may redirect to:
            #   - /ftgw/digital/portfolio/summary
            #   - www.fidelity.com (homepage)
            #   - /ftgw/digital/portfolio/positions
            #   - or other authenticated pages
            # We detect success by checking the URL is no longer the login page.
            import time
            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline:
                url = self._fidelity.page.url
                # Still on login page — keep waiting
                if "login" in url and "digital.fidelity.com" in url:
                    self._fidelity.page.wait_for_timeout(1000)
                    continue
                # Left the login page — login succeeded
                logger.info("Interactive login: detected post-login URL: %s", url)
                break
            else:
                raise TimeoutError(
                    f"Login was not completed within {timeout_sec} seconds. "
                    "Close the browser and try again."
                )

            # Now navigate to the positions page so getAccountInfo() works
            logger.info("Interactive login: navigating to portfolio positions page...")
            self._fidelity.page.goto(
                "https://digital.fidelity.com/ftgw/digital/portfolio/positions",
                timeout=60000,
            )
            self._logged_in = True
            logger.info("Interactive login: success — ready to fetch positions")
            return True
        except PlaywrightNotInstalledError:
            raise
        except Exception as e:
            logger.error("Interactive login failed: %s", e)
            raise

    def complete_2fa(self, code: str, save_device: bool = True) -> bool:
        """Complete 2FA with the code sent to the user's phone.

        Args:
            code: The 6-digit verification code
            save_device: If True, Fidelity will remember this device
        """
        if not self._fidelity:
            logger.error("Cannot complete 2FA: no active login session")
            return False
        try:
            result = self._fidelity.login_2FA(code=code, save_device=save_device)
            if result:
                self._logged_in = True
                logger.info("Fidelity 2FA successful")
            return bool(result)
        except Exception as e:
            logger.error("Fidelity 2FA error: %s", e)
            return False

    def connect_with_saved_session(self) -> bool:
        """Attempt to connect using saved browser session (no login needed).

        Returns True if the saved session is still valid and positions were loaded.
        """
        if not self.has_saved_session:
            return False
        try:
            self._create_automation()
            # getAccountInfo() returns None when session is expired
            result = self._fidelity.getAccountInfo()
            if result is None:
                logger.warning("Saved Fidelity session expired (getAccountInfo returned None)")
                self._logged_in = False
                return False
            self._logged_in = True
            logger.info("Fidelity connected with saved session")
            return True
        except PlaywrightNotInstalledError:
            raise
        except Exception as e:
            logger.warning("Saved Fidelity session expired or invalid: %s", e)
            self._logged_in = False
            return False

    def get_positions(self) -> Portfolio:
        """Fetch all positions across all accounts.

        Must be logged in first (via login() or connect_with_saved_session()).
        """
        if not self._fidelity or not self._logged_in:
            raise RuntimeError("Not logged in to Fidelity")

        try:
            # Fetch account info (populates account_dict)
            result = self._fidelity.getAccountInfo()
            if result is None:
                raise RuntimeError(
                    "Failed to fetch account info from Fidelity. "
                    "The session may have expired — try disconnecting and logging in again."
                )
            account_dict = self._fidelity.account_dict

            holdings = []
            accounts = []

            for acct_num, acct_data in account_dict.items():
                balance = acct_data.get("balance", 0.0)
                nickname = acct_data.get("nickname", acct_num)
                stocks = acct_data.get("stocks", [])

                accounts.append(AccountSummary(
                    account_id=acct_num,
                    account_name=nickname,
                    total_value=float(balance) if balance else 0.0,
                    holdings_count=len(stocks),
                    last_updated=datetime.now(),
                ))

                for stock_entry in stocks:
                    ticker = stock_entry.get("ticker", "")
                    if not ticker or ticker.upper() in ("PENDING", "CASH", "SPAXX**"):
                        continue

                    quantity = float(stock_entry.get("quantity", 0))
                    last_price = float(stock_entry.get("last_price", 0))
                    value = float(stock_entry.get("value", 0))
                    cost_basis = float(stock_entry.get("cost_basis", 0)) if "cost_basis" in stock_entry else 0.0

                    asset = Asset(
                        symbol=ticker.upper(),
                        name=stock_entry.get("name", ticker),
                        asset_type=AssetType.STOCK,
                    )
                    holdings.append(Holding(
                        asset=asset,
                        quantity=quantity,
                        cost_basis=cost_basis,
                        current_price=last_price,
                        account=nickname,
                    ))

            return Portfolio(
                name="Fidelity Portfolio",
                holdings=holdings,
                accounts=accounts,
                last_updated=datetime.now(),
            )

        except Exception as e:
            logger.error("Failed to fetch Fidelity positions: %s", e)
            raise

    def get_summary_holdings(self) -> dict[str, dict]:
        """Get aggregated holdings across all accounts.

        Returns dict like: {'AAPL': {'quantity': 10, 'last_price': 150.0, 'value': 1500.0}}
        """
        if not self._fidelity or not self._logged_in:
            raise RuntimeError("Not logged in to Fidelity")
        return self._fidelity.summary_holdings()

    def get_account_list(self) -> list[AccountSummary]:
        """Get list of all Fidelity accounts."""
        if not self._fidelity or not self._logged_in:
            raise RuntimeError("Not logged in to Fidelity")

        self._fidelity.get_list_of_accounts(set_flag=True)
        accounts = []
        for acct_num, acct_data in self._fidelity.account_dict.items():
            accounts.append(AccountSummary(
                account_id=acct_num,
                account_name=acct_data.get("nickname", acct_num),
                total_value=float(acct_data.get("balance", 0)) if acct_data.get("balance") else 0.0,
                last_updated=datetime.now(),
            ))
        return accounts

    def save_session(self):
        """Explicitly save the current browser session state."""
        if self._fidelity:
            try:
                self._fidelity.save_storage_state()
                logger.info("Fidelity session state saved")
            except Exception as e:
                logger.warning("Failed to save Fidelity session: %s", e)

    def close(self):
        """Close the browser and save state."""
        if self._fidelity:
            try:
                self.save_session()
                self._fidelity.close_browser()
            except Exception as e:
                logger.warning("Error closing Fidelity browser: %s", e)
            finally:
                self._fidelity = None
                self._logged_in = False

    @property
    def is_connected(self) -> bool:
        return self._logged_in and self._fidelity is not None

    def clear_saved_session(self):
        """Delete the saved session file (Fidelity.json created by the library)."""
        state_file = self._state_dir / _LIBRARY_STATE_FILE
        if state_file.exists():
            state_file.unlink()
            logger.info("Fidelity saved session cleared")
