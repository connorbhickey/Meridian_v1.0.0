# Meridian — Portfolio Optimization Terminal

## Quick Start

```bash
python -m portopt.app          # Launch the GUI
python -m portopt.app --check  # Validate installation
python -m pytest tests/ -x -q  # Run all tests (842+)
```

## Project Structure

```
src/portopt/                   # 98 Python source files
  app.py                       # Entry point — launches QApplication + MainWindow
  constants.py                 # Enums (OptMethod, CovEstimator, etc.), Colors, Fonts
  config.py                    # QSettings (INI format via get_settings())
  data/
    models.py                  # Core models: Asset, Holding, Portfolio, OptimizationResult
    cache.py                   # SQLite price cache (stores column names as LOWERCASE)
    manager.py                 # DataManager — coordinates providers + cache
    quality.py                 # Data quality analysis (coverage, staleness, anomalies)
    importers/
      fidelity_csv.py          # Fidelity CSV position export parser
      schwab_csv.py            # Schwab multi-account CSV parser
      robinhood_csv.py         # Robinhood CSV position export parser
      ofx_importer.py          # OFX/QFX SGML file parser (regex, not XML)
      generic_csv.py           # Generic CSV importer (fallback)
    providers/
      yfinance_provider.py     # Yahoo Finance (free, no API key)
      tiingo_provider.py       # Tiingo daily prices (API key required)
      alphavantage_provider.py # Alpha Vantage (API key required)
      fred_provider.py         # FRED macro data (rates, CPI, VIX)
      fundamental_provider.py  # Fundamental data (P/E, P/B, sector, market cap)
  engine/                      # Pure computation — ZERO GUI knowledge
    optimization/
      mean_variance.py         # MVO: max_sharpe, min_vol, efficient_risk/return, etc.
      hrp.py                   # Hierarchical Risk Parity
      herc.py                  # Hierarchical Equal Risk Contribution
      black_litterman.py       # Black-Litterman model
      tic.py                   # Theory-Implied Correlation
    risk.py                    # Covariance estimation, cov_to_corr, nearest_pd
    returns.py                 # Return estimation (historical, CAPM, exponential)
    metrics.py                 # 24 portfolio metrics
    factors.py                 # Fama-French 3-factor model + exposure analysis
    regime.py                  # HMM regime detection + rolling vol regimes
    risk_budgeting.py          # Risk budget optimization + Euler decomposition + ERC
    tax_harvest.py             # Tax-loss harvesting candidates + replacement suggestions
    strategy_compare.py        # Multi-method comparison + parameter sweeps + bootstrap
    execution.py               # Market impact models (sqrt/linear), execution sim
    account_optimizer.py       # Multi-account tax-aware asset location
    order_manager.py           # Trade order generation from optimization deltas
    constraints.py             # PortfolioConstraints dataclass
    network/mst.py             # Minimum Spanning Tree (takes DataFrame, NOT Graph)
  backtest/
    engine.py                  # BacktestEngine + BacktestConfig + BacktestOutput
    runner.py                  # Core execution loop (step through time, rebalance)
    walk_forward.py            # Walk-forward analysis (rolling/anchored windows)
    costs.py                   # BaseCostModel, ZeroCost, etc.
    rebalancer.py              # Rebalance scheduling + drift detection
    results.py                 # SingleRunResult, WalkForwardResult, etc.
  gui/
    main_window.py             # MainWindow — creates all panels, controllers, layouts
    theme.py                   # Dark "deep-space" theme stylesheet
    dock_manager.py            # Save/restore dock layouts (method: list_layouts())
    controllers/               # 5 controllers
      optimization_controller.py  # Wires GUI to engine, runs on QThreadPool
      backtest_controller.py      # Wires GUI to backtest engine
      data_controller.py          # Manages DataManager
      price_stream_controller.py  # Real-time QTimer-based price poller
      fidelity_controller.py      # Fidelity account integration
    panels/                    # 30 panels (+ base_panel.py + __init__.py)
      base_panel.py            # BasePanel(QDockWidget) — base for all panels
      portfolio_panel.py       # Positions table with P&L coloring
      optimization_panel.py    # Optimization config controls
      strategy_lab_panel.py    # Independent optimization/backtest sandbox
      backtest_panel.py        # Backtest config controls
      factor_panel.py          # Factor exposure chart + regression table
      regime_panel.py          # HMM regime timeline + transition matrix
      risk_budget_panel.py     # Risk contribution chart + budget editor
      tax_harvest_panel.py     # Harvest candidates table + savings chart
      data_quality_panel.py    # Coverage, staleness, anomaly detection
      sankey_panel.py          # Rebalance flow diagram (current->target)
      order_panel.py           # Trade order generation and review
      attribution_panel.py     # Return attribution
      copilot_panel.py         # AI assistant panel
      monte_carlo_panel.py     # Monte Carlo simulation
      stress_test_panel.py     # Stress testing scenarios
      frontier_panel.py        # Interactive efficient frontier
      rolling_panel.py         # Rolling analytics
      # + 12 more panels (metrics, risk, correlation, dendrogram, etc.)
    dialogs/                   # 12 dialogs
      api_key_dialog.py        # Manage Anthropic, FRED, Tiingo, AV keys
      preferences_dialog.py    # Application preferences
      export_dialog.py         # Multi-format export (CSV, JSON, Excel, PNG)
      report_dialog.py         # AI report generation
      bl_views_dialog.py       # Black-Litterman views editor
      constraint_dialog.py     # Portfolio constraints editor
      welcome_dialog.py        # First-launch wizard
      about_dialog.py          # About/version info
      # + 4 more dialogs
  utils/
    threading.py               # run_in_thread() — QThreadPool wrapper
    credentials.py             # OS keyring (Windows Credential Manager)
  samples/                     # 3 bundled sample portfolios (JSON)
scripts/
  build.py                     # PyInstaller build script
meridian.spec                  # PyInstaller spec file
.github/workflows/
  ci.yml                       # lint -> test -> build (pinned SHAs)
  release.yml                  # Semantic version tag -> GitHub Release
tests/                         # 38 test files, 842+ tests
  conftest.py                  # Shared fixtures
  engine/                      # Engine tests
  data/                        # Data layer tests
  backtest/                    # Backtest tests
  gui/                         # GUI tests (requires pytest-qt)
```

## Architecture Rules

- **Engine layer has zero GUI knowledge.** Never import PySide6 in `engine/` or `backtest/`.
- **Controllers emit Qt Signals** — MainWindow connects them to panel update methods.
- **All long operations run on QThreadPool** via `utils/threading.py:run_in_thread()`.
- **BasePanel** uses `self._layout` (QVBoxLayout) with alias `self.content_layout`.
- **Strategy Lab** has its own isolated controller instances — lab operations never affect the main portfolio.
- **QSettings**: Always use `config.get_settings()` (INI format). Never use `QSettings(APP_NAME, APP_NAME)` directly (avoids Windows registry).
- **Provider pattern**: Inherit from `BaseDataProvider`, implement `get_prices()`, `get_asset_info()`, `get_current_price()`.
- **DataManager fallback chain**: YFinance -> Tiingo -> AlphaVantage (based on API key availability).
- **CSV import chain**: Fidelity -> Schwab -> Robinhood -> Generic (tries each parser in order).

## Common Pitfalls

- **SQLite cache stores column names as lowercase** — always use case-insensitive column matching:
  ```python
  col_map = {c.lower(): c for c in df.columns}
  ```
- **`compute_mst()` takes a DataFrame**, not a networkx Graph.
- **EFFICIENT_RETURN needs `target_risk`; EFFICIENT_RISK needs `target_return`** — the naming is counter-intuitive.
- **DockManager method is `list_layouts()`**, not `get_saved_layouts()`.
- **Money market symbols** (SPAXX**, FCASH**, etc.) have `AssetType.MONEY_MARKET` — included in `Portfolio.holdings` and `total_value` but excluded from `Portfolio.symbols` (used for optimization/pricing).
- **Backtest initial portfolio value shifts on first day** due to rebalancing costs — use `rel=0.05` tolerance in tests.
- **AssetType enum uses `STOCK`** not `EQUITY` — check `data/models.py` for exact values.
- **OFX is SGML, not XML** — use regex extraction, not XML parser.
- **Schwab CSV multi-account** — sections separated by "Positions for account" header lines.
- **Holding.weight** — uses `_weight` field + property setter (was always 0.0 before fix).

## Environment Variables / API Keys

Stored in OS keyring via `utils/credentials.py`. Managed in-app via **AI > API Key** dialog.

| Credential Key | Purpose | Required |
|----------------|---------|----------|
| `anthropic_api_key` | AI Copilot + report generation | For AI features |
| `fred_api_key` | FRED macro data | Optional |
| `tiingo_api_key` | Tiingo price provider | Optional |
| `alpha_vantage_api_key` | Alpha Vantage price provider | Optional |

## CI/CD

- **Workflow**: `.github/workflows/ci.yml` — lint (ruff) -> test (4-matrix) -> build (PyInstaller)
- **Release**: `.github/workflows/release.yml` — push `v*` tag -> build + GitHub Release
- **Actions pinned to SHA** for supply-chain security
- **Required secrets**: None (all public actions, no deployment secrets needed)

## Testing

```bash
python -m pytest tests/ -x -q          # All tests
python -m pytest tests/engine/ -x -q   # Engine tests only
python -m pytest tests/data/ -x -q     # Data layer tests
python -m pytest tests/backtest/ -x -q # Backtest tests
python -m pytest tests/gui/ -x -q      # GUI tests (needs pytest-qt)
```

- 842+ tests across engine, data, backtest, and GUI layers.
- `pytest-qt` must be installed separately for GUI tests.
- `hmmlearn` must be installed separately for regime detection tests.
- Test fixtures are in `tests/conftest.py`.

## Building

```bash
pip install -e ".[build]"
pyinstaller meridian.spec --noconfirm  # -> dist/Meridian/Meridian.exe
python scripts/build.py                # Wrapper script
```

## Tech Stack

- Python 3.12+, PySide6 6.7+, pyqtgraph, numpy, pandas, scipy, scikit-learn, cvxpy
- SQLite for price caching, keyring for credentials
- hmmlearn for HMM regime detection
- anthropic SDK + jinja2 for AI Copilot and reports
- networkx for MST graph analysis
- yfinance for primary market data (free, no key)
