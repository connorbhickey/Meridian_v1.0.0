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
    Transaction, TransactionSource, TransactionStatus,
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

    def get_activity(self, days: int = 90) -> list[Transaction]:
        """Scrape transaction/activity history from Fidelity.

        Uses a tiered approach:
        1. Intercept XHR/JSON API responses from the Activity page
        2. Fall back to DOM scraping if no JSON endpoint found
        3. Return empty list on failure (never crashes)

        Must be logged in first.
        """
        if not self._fidelity or not self._logged_in:
            raise RuntimeError("Not logged in to Fidelity")

        try:
            page = self._fidelity.page
            captured_data = []

            # Step 1: Set up network interception to capture JSON API responses
            def _on_response(response):
                try:
                    url = response.url
                    if response.status == 200 and (
                        "activity" in url.lower()
                        or "history" in url.lower()
                        or "transaction" in url.lower()
                        or "order" in url.lower()
                    ):
                        content_type = response.headers.get("content-type", "")
                        if "json" in content_type or "javascript" in content_type:
                            try:
                                body = response.json()
                                captured_data.append(("json", url, body))
                            except Exception:
                                pass
                except Exception:
                    pass

            page.on("response", _on_response)

            # Navigate to the Activity & Orders page
            logger.info("Navigating to Fidelity Activity page...")
            page.goto(
                "https://digital.fidelity.com/ftgw/digital/portfolio/activity",
                timeout=60000,
            )
            page.wait_for_load_state("networkidle", timeout=30000)

            # Give extra time for async data loads
            page.wait_for_timeout(3000)

            # Remove the listener
            page.remove_listener("response", _on_response)

            # Step 2: Try to parse captured JSON data
            transactions = self._parse_captured_activity(captured_data)
            if transactions:
                logger.info("Parsed %d Fidelity transactions from XHR data", len(transactions))
                return transactions

            # Step 3: Fall back to DOM scraping
            logger.info("No JSON activity data captured, falling back to DOM scraping...")
            transactions = self._scrape_activity_dom(page)
            if transactions:
                logger.info("Scraped %d Fidelity transactions from DOM", len(transactions))
                return transactions

            logger.warning("Could not extract Fidelity activity data (no JSON or DOM data found)")
            return []

        except Exception as e:
            logger.warning("Failed to fetch Fidelity activity: %s", e)
            return []

    def _parse_captured_activity(self, captured_data: list) -> list[Transaction]:
        """Parse transactions from intercepted JSON responses."""
        transactions = []
        for data_type, url, body in captured_data:
            if data_type != "json" or not isinstance(body, (dict, list)):
                continue

            # Look for transaction-like arrays in the response
            items = []
            if isinstance(body, list):
                items = body
            elif isinstance(body, dict):
                # Common patterns: body["activities"], body["orders"], body["transactions"]
                for key in ("activities", "orders", "transactions", "activity",
                            "data", "results", "items"):
                    val = body.get(key)
                    if isinstance(val, list) and val:
                        items = val
                        break

            for item in items:
                if not isinstance(item, dict):
                    continue
                txn = self._map_fidelity_activity_item(item)
                if txn:
                    transactions.append(txn)

        return transactions

    def _scrape_activity_dom(self, page) -> list[Transaction]:
        """Fall back: scrape the Activity page DOM for transaction rows."""
        try:
            # Try to find a table or list of activity items
            rows_data = page.evaluate("""() => {
                const results = [];
                // Try common table/row selectors
                const selectors = [
                    'table tbody tr',
                    '[data-testid*="activity"] tr',
                    '.activity-row',
                    '[class*="activity"] [class*="row"]',
                    '[class*="transaction"] [class*="row"]',
                ];
                for (const sel of selectors) {
                    const rows = document.querySelectorAll(sel);
                    if (rows.length > 0) {
                        rows.forEach((row, idx) => {
                            const cells = row.querySelectorAll('td, [class*="cell"]');
                            const cellTexts = Array.from(cells).map(c => c.textContent.trim());
                            if (cellTexts.length >= 3) {
                                results.push({
                                    index: idx,
                                    cells: cellTexts,
                                    html: row.innerHTML.substring(0, 500),
                                });
                            }
                        });
                        break;
                    }
                }
                return results;
            }""")

            transactions = []
            for row in (rows_data or []):
                cells = row.get("cells", [])
                if len(cells) < 3:
                    continue

                txn = self._parse_dom_row(cells)
                if txn:
                    transactions.append(txn)

            return transactions

        except Exception as e:
            logger.warning("DOM scraping failed: %s", e)
            return []

    def _map_fidelity_activity_item(self, item: dict) -> Transaction | None:
        """Map a Fidelity JSON activity item to a Transaction."""
        try:
            # Try various field name patterns Fidelity might use
            txn_id = (
                item.get("activityId") or item.get("orderId")
                or item.get("id") or item.get("transactionId")
                or f"fid_{hash(json.dumps(item, default=str)) & 0xFFFFFFFF:08x}"
            )
            acct = item.get("accountNumber") or item.get("account") or item.get("accountId") or ""
            acct_name = item.get("accountName") or item.get("accountNickname") or acct

            # Date
            date_str = item.get("date") or item.get("settleDate") or item.get("tradeDate") or ""
            txn_date = None
            if date_str:
                for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        txn_date = datetime.strptime(str(date_str).split("T")[0], fmt.split("T")[0]).date()
                        break
                    except ValueError:
                        continue

            # Amount
            amount_raw = item.get("amount") or item.get("netAmount") or item.get("total") or 0
            amount = float(str(amount_raw).replace("$", "").replace(",", "").replace("+", "")) if amount_raw else 0

            # Description / type
            description = item.get("description") or item.get("activityDescription") or item.get("name") or ""
            txn_type = item.get("type") or item.get("activityType") or item.get("transactionType") or ""

            # Status
            status_str = (item.get("status") or item.get("settlementStatus") or "").lower()
            if "pending" in status_str or "open" in status_str:
                status = TransactionStatus.PENDING
                pending = True
            else:
                status = TransactionStatus.POSTED
                pending = False

            # Sign convention: buys/contributions are positive (debit), sells/dividends negative (credit)
            txn_type_lower = txn_type.lower()
            if any(kw in txn_type_lower for kw in ("sell", "dividend", "distribution", "interest", "credit")):
                amount = -abs(amount) if amount > 0 else amount
            elif any(kw in txn_type_lower for kw in ("buy", "contribution", "fee", "debit")):
                amount = abs(amount)

            return Transaction(
                transaction_id=str(txn_id),
                account_id=str(acct),
                account_name=str(acct_name),
                date=txn_date,
                authorized_date=txn_date,
                amount=amount,
                merchant_name="Fidelity",
                name=str(description),
                category=str(txn_type),
                status=status,
                pending=pending,
                institution_name="Fidelity",
                source=TransactionSource.FIDELITY,
                iso_currency_code="USD",
                metadata={"raw_type": str(txn_type)},
            )
        except Exception as e:
            logger.debug("Failed to map Fidelity activity item: %s", e)
            return None

    def _parse_dom_row(self, cells: list[str]) -> Transaction | None:
        """Parse a DOM table row into a Transaction."""
        try:
            # Heuristic: first cell is usually date, last is amount
            date_str = cells[0] if cells else ""
            description = cells[1] if len(cells) > 1 else ""
            amount_str = cells[-1] if cells else "0"

            txn_date = None
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y"):
                try:
                    txn_date = datetime.strptime(date_str.strip(), fmt).date()
                    break
                except ValueError:
                    continue

            # Parse amount
            clean_amount = amount_str.replace("$", "").replace(",", "").replace("+", "").strip()
            if clean_amount.startswith("(") and clean_amount.endswith(")"):
                clean_amount = "-" + clean_amount[1:-1]
            try:
                amount = float(clean_amount)
            except ValueError:
                amount = 0.0

            txn_id = f"fid_dom_{hash(f'{date_str}{description}{amount_str}') & 0xFFFFFFFF:08x}"

            return Transaction(
                transaction_id=txn_id,
                account_id="",
                account_name="",
                date=txn_date,
                authorized_date=txn_date,
                amount=amount,
                merchant_name="Fidelity",
                name=description.strip(),
                category="",
                status=TransactionStatus.POSTED,
                pending=False,
                institution_name="Fidelity",
                source=TransactionSource.FIDELITY,
                iso_currency_code="USD",
                metadata={"raw_cells": cells},
            )
        except Exception as e:
            logger.debug("Failed to parse DOM row: %s", e)
            return None

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
