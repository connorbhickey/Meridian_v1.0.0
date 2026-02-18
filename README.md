# Meridian

**Quantitative Portfolio Terminal** — professional-grade portfolio optimization, backtesting, and risk analysis.

Built with Python + PySide6. Runs on Windows, macOS, and Linux.

![CI](https://github.com/connorbhickey/Meridian_v1.0.0/actions/workflows/ci.yml/badge.svg)

## Features

- **14 Optimization Methods** — Mean-Variance (max Sharpe, min vol, efficient frontier), HRP, HERC, Black-Litterman, TIC, Risk Budgeting, and more
- **Walk-Forward Backtesting** — configurable rebalance frequency, transaction costs, and benchmark comparison
- **Factor Analysis** — Fama-French 3-factor model with exposure decomposition
- **Regime Detection** — HMM-based market regime identification with transition matrices
- **Risk Budgeting** — Euler decomposition, Equal Risk Contribution, custom risk budgets
- **Tax-Loss Harvesting** — candidate identification with replacement suggestions and savings estimates
- **Strategy Comparison** — multi-method side-by-side analysis with parameter sweeps and bootstrap tests
- **Execution Simulation** — market impact models (square-root, linear), capacity analysis
- **Monte Carlo Simulation** — parametric GBM and block bootstrap with confidence bands
- **AI Copilot** — Claude-powered assistant for portfolio analysis and report generation
- **Multi-Brokerage Import** — Fidelity, Schwab, Robinhood CSV + OFX/QFX file support
- **Real-Time Data** — Yahoo Finance (free), with Tiingo and Alpha Vantage as backups
- **Macro Data** — FRED integration for rates, CPI, VIX, and more
- **30 Dockable Panels** — Bloomberg-style tiling with save/restore layouts and presets
- **Export** — CSV, JSON, Excel, and PNG chart export

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Launch
python -m portopt.app

# Validate installation
python -m portopt.app --check

# Run tests
python -m pytest tests/ -x -q
```

## Sample Portfolios

Three sample portfolios are included to explore immediately:

- **Balanced 60/40** — VTI, VXUS, BND, BNDX, VNQ
- **Tech Growth** — AAPL, MSFT, NVDA, GOOGL, AMZN, META, and more
- **Dividend Income** — VYM, SCHD, JNJ, PG, KO, and more

Load via **File > Load Sample** or the welcome wizard on first launch.

## API Keys

Enter via **AI > API Key** in the app. Keys are stored in OS credential manager (encrypted).

| Key | Used By | Required? |
|-----|---------|-----------|
| Anthropic | AI Copilot chat + report generation | For AI features |
| FRED | Macro data (rates, CPI, VIX) | Optional |
| Tiingo | Backup price provider | Optional |
| Alpha Vantage | Backup price provider | Optional |

Yahoo Finance works without any API key for basic price data.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+I` | Import portfolio |
| `Ctrl+O` | Run optimization |
| `Ctrl+B` | Run backtest |
| `Ctrl+L` | Strategy Lab |
| `Ctrl+F` | Fidelity connection |
| `F5` | Refresh positions |
| `Ctrl+Shift+E` | Export |
| `Ctrl+R` | Generate AI report |
| `Ctrl+Shift+A` | Open Copilot |
| `F1` | Keyboard shortcuts |
| `Ctrl+Q` | Exit |

## Architecture

```
src/portopt/
  app.py                     # Entry point
  constants.py               # Enums, colors, fonts
  config.py                  # QSettings (INI format) configuration
  engine/                    # Pure computation — zero GUI knowledge
    optimization/            # MVO, HRP, HERC, Black-Litterman, TIC
    risk.py                  # Covariance estimation, nearest PD
    returns.py               # Return estimation (historical, CAPM, EW)
    metrics.py               # 24 portfolio metrics
    factors.py               # Fama-French 3-factor
    regime.py                # HMM regime detection
    risk_budgeting.py        # Risk budget optimization + Euler decomposition
    tax_harvest.py           # Tax-loss harvesting
    strategy_compare.py      # Multi-method comparison + bootstrap
    execution.py             # Market impact models + execution simulation
    account_optimizer.py     # Multi-account tax-aware asset location
    order_manager.py         # Trade order generation
    network/mst.py           # Minimum Spanning Tree
  data/                      # Data providers, importers, caching
    models.py                # Asset, Holding, Portfolio, OptimizationResult
    cache.py                 # SQLite price cache
    manager.py               # DataManager (coordinates providers)
    providers/               # YFinance, Tiingo, AlphaVantage, FRED, Fundamentals
    importers/               # Fidelity, Schwab, Robinhood, OFX, Generic CSV
    quality.py               # Data quality analysis
  backtest/                  # Backtesting engine + walk-forward analysis
  gui/                       # PySide6 panels (30), controllers (5), dialogs (12)
  samples/                   # Bundled sample portfolios
  utils/                     # Threading, credentials, helpers
```

- **Engine layer** has zero GUI imports — testable independently
- **Controllers** emit Qt signals; MainWindow wires them to panel updates
- **All long operations** run on QThreadPool (never blocks the UI)

## Tech Stack

- Python 3.12+, PySide6, pyqtgraph, numpy, pandas, scipy, scikit-learn, cvxpy
- SQLite for price caching
- Keyring for secure credential storage
- hmmlearn for regime detection
- anthropic SDK for AI Copilot
- jinja2 for report templating

## Testing

```bash
python -m pytest tests/ -x -q          # All 842+ tests
python -m pytest tests/engine/ -x -q   # Engine only
python -m pytest tests/data/ -x -q     # Data layer
python -m pytest tests/backtest/ -x -q # Backtesting
python -m pytest tests/gui/ -x -q      # GUI (requires pytest-qt)
```

## CI/CD

- **CI**: lint (ruff) → test (4 matrix: Ubuntu/Windows x Python 3.12/3.13) → build (PyInstaller)
- **Release**: Push a `v*` tag to trigger Windows build + GitHub Release with changelog
- Actions pinned to SHA for supply-chain security

```bash
git tag v1.0.0 && git push origin v1.0.0   # Trigger release
```

## Building

```bash
pip install -e ".[build]"
python scripts/build.py                # Produces dist/Meridian/Meridian.exe
pyinstaller meridian.spec --noconfirm  # Direct PyInstaller build
```

## License

Private repository. All rights reserved.
