"""Core data models used throughout the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto

import numpy as np

from portopt.constants import MCSimMethod


class AssetType(Enum):
    STOCK = auto()
    ETF = auto()
    MUTUAL_FUND = auto()
    INDEX_FUND = auto()
    BOND = auto()
    OPTION = auto()
    CRYPTO = auto()
    MONEY_MARKET = auto()
    OTHER = auto()


@dataclass
class Asset:
    """A single tradeable asset."""
    symbol: str
    name: str = ""
    asset_type: AssetType = AssetType.STOCK
    sector: str = ""
    exchange: str = ""
    currency: str = "USD"

    def __hash__(self):
        return hash(self.symbol)

    def __eq__(self, other):
        if isinstance(other, Asset):
            return self.symbol == other.symbol
        return NotImplemented


@dataclass
class Holding:
    """A position in a portfolio — an asset with quantity and cost basis."""
    asset: Asset
    quantity: float
    cost_basis: float = 0.0           # Total cost basis
    current_price: float = 0.0
    account: str = ""                 # Account name/number
    _weight: float = field(default=0.0, repr=False)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_per_share(self) -> float:
        return self.cost_basis / self.quantity if self.quantity else 0.0

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        return (self.unrealized_pnl / self.cost_basis * 100) if self.cost_basis else 0.0

    @property
    def weight(self) -> float:
        """Portfolio weight — set by Portfolio._update_weights()."""
        return self._weight

    @weight.setter
    def weight(self, value: float):
        self._weight = value


@dataclass
class AccountSummary:
    """Summary of a brokerage account."""
    account_id: str
    account_name: str = ""
    account_type: str = ""            # e.g. "Individual", "Roth IRA", "401k"
    total_value: float = 0.0
    cash_balance: float = 0.0
    holdings_count: int = 0
    last_updated: datetime | None = None


@dataclass
class Portfolio:
    """A collection of holdings representing a portfolio."""
    name: str = "Portfolio"
    holdings: list[Holding] = field(default_factory=list)
    accounts: list[AccountSummary] = field(default_factory=list)
    last_updated: datetime | None = None

    def __post_init__(self):
        self._update_weights()

    def _update_weights(self):
        """Recompute and set weight on each holding."""
        tv = self.total_value
        for h in self.holdings:
            h.weight = h.market_value / tv if tv > 0 else 0.0

    @property
    def total_value(self) -> float:
        return sum(h.market_value for h in self.holdings)

    @property
    def total_cost(self) -> float:
        return sum(h.cost_basis for h in self.holdings)

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.total_cost

    @property
    def total_pnl_pct(self) -> float:
        return (self.total_pnl / self.total_cost * 100) if self.total_cost else 0.0

    @property
    def symbols(self) -> list[str]:
        """Tradeable symbols only (excludes money market / cash positions)."""
        return [
            h.asset.symbol for h in self.holdings
            if h.asset.asset_type != AssetType.MONEY_MARKET
        ]

    @property
    def weights(self) -> dict[str, float]:
        """Current portfolio weights by symbol."""
        tv = self.total_value
        if tv == 0:
            return {}
        return {h.asset.symbol: h.market_value / tv for h in self.holdings}

    def get_holding(self, symbol: str) -> Holding | None:
        for h in self.holdings:
            if h.asset.symbol == symbol:
                return h
        return None


# ── Financial Account / Transaction Models ───────────────────────────


class TransactionStatus(Enum):
    PENDING = auto()
    POSTED = auto()
    REMOVED = auto()


class TransactionType(Enum):
    DEBIT = auto()
    CREDIT = auto()


class PlaidAccountType(Enum):
    CHECKING = auto()
    SAVINGS = auto()
    CREDIT_CARD = auto()
    BROKERAGE = auto()
    IRA = auto()
    CASH_MANAGEMENT = auto()
    OTHER = auto()


class TransactionSource(Enum):
    PLAID = auto()
    FIDELITY = auto()


@dataclass
class Transaction:
    """A financial transaction from any source (Plaid or Fidelity)."""
    transaction_id: str
    account_id: str
    account_name: str = ""
    date: date | None = None
    authorized_date: date | None = None
    amount: float = 0.0                # Signed: positive=debit, negative=credit
    merchant_name: str = ""
    name: str = ""                     # Transaction description
    category: str = ""
    status: TransactionStatus = TransactionStatus.POSTED
    pending: bool = False
    institution_name: str = ""
    source: TransactionSource = TransactionSource.PLAID
    iso_currency_code: str = "USD"
    metadata: dict = field(default_factory=dict)

    @property
    def display_amount(self) -> str:
        """Formatted amount: debit as -$X.XX, credit as +$X.XX."""
        if self.amount >= 0:
            return f"-${self.amount:,.2f}"
        return f"+${abs(self.amount):,.2f}"

    @property
    def is_credit(self) -> bool:
        return self.amount < 0


@dataclass
class PlaidAccount:
    """A financial account linked via Plaid."""
    account_id: str
    item_id: str = ""
    institution_name: str = ""
    name: str = ""
    official_name: str = ""
    account_type: PlaidAccountType = PlaidAccountType.OTHER
    subtype: str = ""
    mask: str = ""                     # Last 4 digits
    current_balance: float = 0.0
    available_balance: float | None = None
    limit: float | None = None         # Credit limit
    last_synced: datetime | None = None

    @property
    def display_name(self) -> str:
        if self.official_name:
            return f"{self.official_name} (***{self.mask})" if self.mask else self.official_name
        if self.name:
            return f"{self.name} (***{self.mask})" if self.mask else self.name
        return f"Account ***{self.mask}" if self.mask else self.account_id


@dataclass
class PlaidItem:
    """A linked institution in Plaid (one Item = one bank/institution)."""
    item_id: str
    institution_id: str = ""
    institution_name: str = ""
    accounts: list[PlaidAccount] = field(default_factory=list)
    last_synced: datetime | None = None
    sync_cursor: str = ""              # For /transactions/sync incremental updates
    error: str = ""


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""
    method: str
    weights: dict[str, float]         # symbol -> weight
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    equity_curve: list[float] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    weights_history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class PriceBar:
    """Single OHLCV price bar."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    adj_close: float | None = None


# ── Monte Carlo ──────────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    """Configuration for a Monte Carlo simulation run."""
    n_sims: int = 1000
    horizon_days: int = 252
    method: MCSimMethod = MCSimMethod.PARAMETRIC
    block_size: int = 20              # for block bootstrap
    initial_value: float = 100_000.0
    spending_rate: float = 0.04       # annual withdrawal rate for shortfall calc
    risk_free_rate: float = 0.04
    frequency: int = 252
    percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)


@dataclass
class MonteCarloResult:
    """Aggregated output from a Monte Carlo simulation."""
    # Fan chart: shape (horizon_days+1, len(percentiles))
    equity_percentiles: np.ndarray
    percentile_labels: tuple[int, ...]
    dates: list[date]

    # Metrics distribution: key -> sorted array of length n_sims
    metrics_distributions: dict[str, np.ndarray]

    # Shortfall analysis
    shortfall_probability: float
    shortfall_threshold: float

    # Summary
    n_sims: int
    method: str
    config: MonteCarloConfig
    metadata: dict = field(default_factory=dict)

    @property
    def median_curve(self) -> np.ndarray:
        """Median equity curve (50th percentile)."""
        idx = list(self.percentile_labels).index(50)
        return self.equity_percentiles[:, idx]

    def percentile_curve(self, pct: int) -> np.ndarray:
        """Get equity curve for a specific percentile."""
        idx = list(self.percentile_labels).index(pct)
        return self.equity_percentiles[:, idx]
