"""Application-wide constants, enums, and color definitions."""

from enum import Enum, auto


# ── Application ──────────────────────────────────────────────────────
APP_NAME = "PortOpt"
APP_VERSION = "0.1.0"
APP_ORG = "PortOpt"
MAX_ASSETS = 100

# ── Colors (Trading Terminal Palette) ────────────────────────────────
class Colors:
    BG_PRIMARY = "#0d1117"       # Main background
    BG_SECONDARY = "#161b22"     # Panel backgrounds
    BG_TERTIARY = "#1c2128"      # Elevated surfaces
    BG_INPUT = "#21262d"         # Input fields
    BORDER = "#30363d"           # Panel borders
    BORDER_LIGHT = "#3a424b"     # Subtle separators

    TEXT_PRIMARY = "#e6edf3"     # Main text
    TEXT_SECONDARY = "#8b949e"   # Muted text
    TEXT_MUTED = "#6e7681"       # Dimmed text

    ACCENT = "#00d4ff"           # Cyan accent (selections, highlights)
    ACCENT_DIM = "#0a3d5c"       # Accent background

    PROFIT = "#00ff88"           # Green (gains)
    PROFIT_DIM = "#0d3321"       # Green background
    LOSS = "#ff4444"             # Red (losses)
    LOSS_DIM = "#3d1111"         # Red background

    WARNING = "#f0b429"          # Amber warnings
    GRID = "#2a2a3e"             # Chart grid lines

    # Heatmap gradient
    HEATMAP_NEG = "#ff4444"
    HEATMAP_ZERO = "#1c2128"
    HEATMAP_POS = "#00ff88"


# ── Fonts ────────────────────────────────────────────────────────────
class Fonts:
    MONO = "Consolas"            # Monospace for numbers/data
    SANS = "Segoe UI"            # Sans-serif for labels
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
    CONSOLE = "console"
