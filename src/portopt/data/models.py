"""Core data models used throughout the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto


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
        """Weight placeholder — set externally by Portfolio."""
        return 0.0


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
