# Meridian

**Professional Portfolio Terminal** — quantitative optimization, walk-forward backtesting, and risk analytics in a single desktop app.

Built with Python + PySide6. Dark "deep-space" theme with 33 dockable panels.

![CI](https://github.com/connorbhickey/Meridian_v1.0.0/actions/workflows/ci.yml/badge.svg)

---

## Highlights

| Category | Capabilities |
|----------|-------------|
| **Optimization** | Mean-Variance (max Sharpe, min vol, efficient frontier), HRP, HERC, Black-Litterman, TIC, Risk Budgeting, inverse variance — 14 methods total |
| **Backtesting** | Walk-forward analysis, configurable rebalance frequency, 6 transaction cost models, benchmark comparison, out-of-sample validation |
| **Prediction** | 25-method ensemble stock predictor — Merton Jump-Diffusion Monte Carlo, James-Stein shrinkage, Kelly criterion, bootstrap CI, vol-adaptive signal scaling |
| **Risk** | Factor analysis (Fama-French 3-factor), HMM regime detection, Monte Carlo simulation (parametric GBM + block bootstrap), stress testing |
| **Tax & Execution** | Tax-loss harvesting with replacement suggestions, multi-account asset location, market impact models, trade order generation |
| **Data** | Yahoo Finance (free, no key), Tiingo, Alpha Vantage, FRED macro data — with SQLite caching and automatic retry on network failures |
| **Import** | Fidelity, Schwab, Robinhood CSV + OFX/QFX files — auto-detected format |
| **AI** | Claude-powered copilot chat, portfolio report generation, natural language analysis |
| **Interface** | 33 dockable panels, Bloomberg-style tiling, save/restore layouts, CSV/JSON/Excel/PNG export |

---

## Installation

### Windows Installer (Recommended)

Download `Meridian-Setup.exe` from [Releases](https://github.com/connorbhickey/Meridian_v1.0.0/releases). Double-click to install — includes Start Menu shortcut, desktop icon, and Add/Remove Programs entry.

### From Source

```bash
# Clone and install
git clone https://github.com/connorbhickey/Meridian_v1.0.0.git
cd Meridian_v1.0.0
pip install -e ".[dev]"

# Launch
python -m portopt.app

# Validate environment
python -m portopt.app --check
```

**Requirements:** Python 3.12+ on Windows, macOS, or Linux.

---

## Quick Start

1. **Launch** — run `python -m portopt.app` or the installed desktop shortcut
2. **Load a sample** — File > Load Sample (Balanced 60/40, Tech Growth, or Dividend Income)
3. **Optimize** — Ctrl+O to open optimization panel, select method, click Run
4. **Backtest** — Ctrl+B to configure and run walk-forward backtest
5. **Predict** — AI > Stock Predictor (Ctrl+Shift+P) to run 25-method ensemble forecasts
6. **Explore** — dock and arrange any of the 33 panels to build your workspace

### Sample Portfolios

| Portfolio | Assets | Description |
|-----------|--------|-------------|
| Balanced 60/40 | VTI, VXUS, BND, BNDX, VNQ | Classic diversified allocation |
| Tech Growth | AAPL, MSFT, NVDA, GOOGL, AMZN, META, ... | Large-cap technology focus |
| Dividend Income | VYM, SCHD, JNJ, PG, KO, ... | High-yield dividend strategy |

---

## API Keys

Enter via **AI > API Key** in the menu bar. Stored securely in OS credential manager (Windows Credential Manager / macOS Keychain).

| Key | Purpose | Required? |
|-----|---------|-----------|
| **Anthropic** | AI Copilot chat + report generation | For AI features |
| **FRED** | Macro data (interest rates, CPI, VIX) | Optional |
| **Tiingo** | Backup price provider | Optional |
| **Alpha Vantage** | Backup price provider | Optional |

Yahoo Finance provides free price data with no API key. Tiingo and Alpha Vantage serve as fallback providers with automatic retry on network failures.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+I` | Import portfolio (CSV/OFX) |
| `Ctrl+O` | Run optimization |
| `Ctrl+B` | Run backtest |
| `Ctrl+L` | Open Strategy Lab |
| `F5` | Refresh positions |
| `Ctrl+Shift+E` | Export data |
| `Ctrl+R` | Generate AI report |
| `Ctrl+Shift+A` | Open AI Copilot |
| `Ctrl+Shift+P` | Stock Predictor |
| `Ctrl+F` | Fidelity auto-import |
| `F1` | Show all shortcuts |
| `Ctrl+Q` | Exit |

---

## Architecture

```
src/portopt/
  app.py                     # Entry point — QApplication + MainWindow
  constants.py               # Enums (OptMethod, CovEstimator, ...), Colors, Fonts
  config.py                  # QSettings (INI format)
  engine/                    # Pure computation — zero GUI imports
    optimization/            # MVO, HRP, HERC, Black-Litterman, TIC, Risk Budget
    risk.py                  # Covariance estimation (sample, Ledoit-Wolf, denoised, EW)
    returns.py               # Return estimation (historical, CAPM, exponential)
    metrics.py               # 24 portfolio metrics (Sharpe, Sortino, CVaR, max DD, ...)
    factors.py               # Fama-French 3-factor regression
    regime.py                # HMM regime detection + transition matrices
    risk_budgeting.py        # Risk budget optimization + Euler decomposition + ERC
    tax_harvest.py           # Tax-loss harvesting candidates
    strategy_compare.py      # Multi-method comparison + parameter sweeps
    execution.py             # Market impact models (sqrt/linear) + capacity analysis
    network/mst.py           # Minimum Spanning Tree network graph
    prediction/              # 25-method ensemble stock predictor (MJD, signals, J-S, Kelly)
  data/
    providers/               # YFinance, Tiingo, AlphaVantage, FRED, Fundamentals
    importers/               # Fidelity, Schwab, Robinhood CSV, OFX/QFX, Generic
    cache.py                 # SQLite price/asset cache with WAL mode
    manager.py               # DataManager — coordinates providers with retry + fallback
    quality.py               # Data quality (coverage, staleness, anomalies)
  backtest/                  # Engine, runner, walk-forward, costs, rebalancer
  gui/
    panels/                  # 33 panels (portfolio, frontier, correlation, prediction, ...)
    controllers/             # 7 controllers (optimization, backtest, data, prediction, ...)
    dialogs/                 # 12 dialogs (API keys, preferences, export, ...)
    theme.py                 # Dark "deep-space" stylesheet
    dock_manager.py          # Save/restore panel layouts
  utils/
    threading.py             # QThreadPool worker wrappers
    credentials.py           # OS keyring integration
    retry.py                 # Exponential backoff for network failures
```

**Design principles:**
- Engine layer has zero GUI imports — testable independently
- Controllers emit Qt signals; MainWindow wires them to panels
- All long operations run on QThreadPool (never blocks UI)
- Data providers use automatic retry with exponential backoff

---

## Testing

```bash
python -m pytest tests/ -x -q          # All 1131 tests
python -m pytest tests/engine/ -x -q   # Engine (optimization, metrics, risk)
python -m pytest tests/data/ -x -q     # Data layer (providers, cache, importers)
python -m pytest tests/backtest/ -x -q # Backtesting engine
python -m pytest tests/gui/ -x -q      # GUI integration + stress tests
```

**Coverage:** 1131 tests across engine, data, backtest, and GUI layers — including integration tests for full controller signal chains, stress tests with 500+ holdings, and 119 prediction engine tests.

---

## CI/CD

- **CI**: Lint (ruff) → Test (4-matrix: Ubuntu/Windows x Python 3.12/3.13) → Build (PyInstaller)
- **Release**: Push `v*` tag → PyInstaller build → NSIS installer → GitHub Release with changelog
- All GitHub Actions pinned to SHA for supply-chain security

```bash
# Trigger a release
git tag v2.1.2 && git push origin v2.1.2
```

---

## Building from Source

```bash
# Install build dependencies
pip install -e ".[build]"

# Build standalone executable
pyinstaller meridian.spec --noconfirm   # -> dist/Meridian/Meridian.exe

# Build Windows installer (requires NSIS)
makensis scripts/installer/meridian.nsi  # -> dist/Meridian-Setup.exe
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **UI** | PySide6, pyqtgraph |
| **Computation** | numpy, pandas, scipy, scikit-learn, cvxpy |
| **Data** | yfinance, requests, SQLite |
| **AI** | anthropic SDK, jinja2 |
| **Regime Detection** | hmmlearn |
| **Network Analysis** | networkx |
| **Security** | keyring (OS credential manager) |
| **Build** | PyInstaller, NSIS |

---

## License

Private repository. All rights reserved.
