# Meridian

**Quantitative Portfolio Terminal** — professional-grade portfolio optimization, backtesting, and risk analysis.

Built with Python + PySide6. Runs on Windows, macOS, and Linux.

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
- **31 Dockable Panels** — Bloomberg-style tiling with save/restore layouts and presets
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
  engine/         # Pure computation — zero GUI knowledge
  data/           # Data providers, importers, caching
  backtest/       # Backtesting engine
  gui/            # PySide6 panels, controllers, dialogs
  samples/        # Bundled sample portfolios
```

- **Engine layer** has zero GUI imports — testable independently
- **Controllers** emit Qt signals; MainWindow wires them to panel updates
- **All long operations** run on QThreadPool (never blocks the UI)

## Tech Stack

- Python 3.12+, PySide6, pyqtgraph, numpy, pandas, scipy, scikit-learn, cvxpy
- SQLite for price caching
- Keyring for secure credential storage

## Testing

```bash
python -m pytest tests/ -x -q          # All 842+ tests
python -m pytest tests/engine/ -x -q   # Engine only
python -m pytest tests/gui/ -x -q      # GUI (requires pytest-qt)
```

## Building

```bash
pip install -e ".[build]"
python scripts/build.py                # Produces dist/Meridian.exe
```

## License

Private repository. All rights reserved.
