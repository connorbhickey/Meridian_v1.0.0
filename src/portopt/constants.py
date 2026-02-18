"""Application-wide constants, enums, and color definitions."""

from enum import Enum, auto


# ── Application ──────────────────────────────────────────────────────
APP_NAME = "Meridian"
APP_VERSION = "1.0.0"
APP_ORG = "Meridian"
MAX_ASSETS = 100

# ── Colors (Deep Space Palette) ──────────────────────────────────────
class Colors:
    # Backgrounds (Deep Space)
    BG_PRIMARY = "#0a0e14"       # Deepest layer
    BG_SECONDARY = "#0d1117"     # Panel backgrounds
    BG_TERTIARY = "#161b22"      # Elevated surfaces
    BG_INPUT = "#1c2333"         # Input fields
    BG_ELEVATED = "#21283b"      # Hover states

    # Borders
    BORDER = "#30363d"           # Panel borders
    BORDER_LIGHT = "#484f58"     # Active borders, focus rings

    # Text hierarchy
    TEXT_PRIMARY = "#f0f6fc"     # Headings, values
    TEXT_SECONDARY = "#e6edf3"   # Labels
    TEXT_MUTED = "#8b949e"       # Descriptions
    TEXT_DISABLED = "#6e7681"    # Disabled text

    # Accent (Arctic Cyan)
    ACCENT = "#00d4ff"           # Primary accent
    ACCENT_HOVER = "#39ddff"     # Hover states
    ACCENT_LIGHT = "#79e8ff"     # Light accent for charts
    ACCENT_DIM = "#0a3d5c"      # Accent backgrounds
    ACCENT_GLOW = "rgba(0,212,255,0.08)"  # Subtle glow

    # Semantic
    PROFIT = "#00ff88"           # Green (gains)
    PROFIT_DIM = "#0a3d2a"      # Green background
    LOSS = "#ff4757"             # Red (losses)
    LOSS_DIM = "#3d0a15"        # Red background
    WARNING = "#f0b429"          # Amber warnings
    WARNING_DIM = "#3d2e0a"     # Warning background

    # Extended palette
    PURPLE = "#a855f7"
    PINK = "#ec4899"
    GRID = "#2a2a3e"             # Chart grid lines

    # Heatmap gradient
    HEATMAP_NEG = "#ff4757"
    HEATMAP_ZERO = "#0d1117"     # Aligned to BG_SECONDARY
    HEATMAP_POS = "#00ff88"

    # Centralized chart palette (10 colors for multi-series data)
    CHART_PALETTE = [
        "#00d4ff", "#00ff88", "#f0b429", "#a855f7", "#ec4899",
        "#06b6d4", "#f97316", "#84cc16", "#6366f1", "#f43f5e",
    ]

    # Sector colors for network graph
    SECTOR_COLORS = {
        "Technology": "#00d4ff",
        "Healthcare": "#00ff88",
        "Financials": "#f0b429",
        "Consumer Discretionary": "#ec4899",
        "Consumer Staples": "#84cc16",
        "Industrials": "#6366f1",
        "Energy": "#ff4757",
        "Materials": "#f97316",
        "Utilities": "#06b6d4",
        "Real Estate": "#a855f7",
        "Communication Services": "#d946ef",
        "Other": "#8b949e",
    }


# ── Fonts ────────────────────────────────────────────────────────────
class Fonts:
    MONO = "JetBrains Mono"      # Primary monospace
    MONO_FALLBACK = "Consolas"   # Windows fallback
    SANS = "Inter"               # Primary sans-serif
    SANS_FALLBACK = "Segoe UI"  # Windows fallback
    SIZE_SMALL = 9
    SIZE_NORMAL = 10
    SIZE_LARGE = 12
    SIZE_HEADER = 14
    SIZE_TICKER = 11


# ── Optimization Methods ─────────────────────────────────────────────
class OptMethod(Enum):
    INVERSE_VARIANCE = auto()
    MIN_VOLATILITY = auto()
    MAX_SHARPE = auto()
    EFFICIENT_RISK = auto()
    EFFICIENT_RETURN = auto()
    MAX_QUADRATIC_UTILITY = auto()
    MAX_DIVERSIFICATION = auto()
    MAX_DECORRELATION = auto()
    CUSTOM_OBJECTIVE = auto()
    BLACK_LITTERMAN = auto()
    HRP = auto()
    HERC = auto()
    TIC = auto()
    RISK_BUDGET = auto()


class OnlineMethod(Enum):
    BAH = auto()
    BEST_STOCK = auto()
    CRP = auto()
    BCRP = auto()
    EG = auto()
    FTL = auto()
    FTRL = auto()
    PAMR = auto()
    CWMR = auto()
    OLMAR = auto()
    RMR = auto()
    CORN = auto()
    SCORN = auto()
    FCORN = auto()
    FCORN_K = auto()


# ── Covariance Estimators ────────────────────────────────────────────
class CovEstimator(Enum):
    SAMPLE = auto()
    LEDOIT_WOLF = auto()
    DENOISED = auto()             # Marchenko-Pastur
    EXPONENTIAL = auto()


# ── Return Estimators ────────────────────────────────────────────────
class ReturnEstimator(Enum):
    HISTORICAL_MEAN = auto()
    CAPM = auto()
    EXPONENTIAL = auto()


# ── Monte Carlo Simulation Methods ──────────────────────────────────
class MCSimMethod(Enum):
    PARAMETRIC = auto()       # GBM from estimated mu/Sigma
    BOOTSTRAP = auto()        # Block bootstrap from historical returns


# ── Linkage Methods ──────────────────────────────────────────────────
class LinkageMethod(Enum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"


# ── Risk Measures ────────────────────────────────────────────────────
class RiskMeasure(Enum):
    VARIANCE = auto()
    STD_DEV = auto()
    CVAR = auto()
    CDAR = auto()


# ── Rebalance Frequency ─────────────────────────────────────────────
class RebalanceFreq(Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"


# ── Transaction Cost Models ──────────────────────────────────────────
class CostModel(Enum):
    ZERO = auto()
    FIXED = auto()
    PROPORTIONAL = auto()
    TIERED = auto()
    SPREAD = auto()
    COMPOSITE = auto()


# ── Data Sources ─────────────────────────────────────────────────────
class DataSource(Enum):
    YFINANCE = auto()
    ALPHA_VANTAGE = auto()
    CACHE = auto()


# ── Panel IDs ────────────────────────────────────────────────────────
class PanelID(Enum):
    PORTFOLIO = "portfolio"
    WATCHLIST = "watchlist"
    PRICE_CHART = "price_chart"
    CORRELATION = "correlation"
    OPTIMIZATION = "optimization"
    WEIGHTS = "weights"
    FRONTIER = "frontier"
    BACKTEST = "backtest"
    METRICS = "metrics"
    ATTRIBUTION = "attribution"
    NETWORK = "network"
    DENDROGRAM = "dendrogram"
    TRADE_BLOTTER = "trade_blotter"
    RISK = "risk"
    COMPARISON = "comparison"
    SCENARIO = "scenario"
    STRATEGY_LAB = "strategy_lab"
    CONSOLE = "console"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    ROLLING_ANALYTICS = "rolling_analytics"
    COPILOT = "copilot"
    FACTOR_ANALYSIS = "factor_analysis"
    REGIME = "regime"
    RISK_BUDGET = "risk_budget"
    TAX_HARVEST = "tax_harvest"
