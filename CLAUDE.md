# Meridian — Portfolio Optimization Terminal

## Quick Start

```bash
python -m portopt.app          # Launch the GUI
python -m pytest tests/ -x -q  # Run all tests
```

## Project Structure

```
src/portopt/
  app.py                     # Entry point — launches QApplication + MainWindow
  constants.py               # Enums (OptMethod, CovEstimator, etc.), colors, fonts
  data/
    models.py                # Core data models: Asset, Holding, Portfolio, OptimizationResult
    cache.py                 # SQLite price cache (stores column names as LOWERCASE)
    manager.py               # DataManager — coordinates providers + cache
    importers/
      fidelity_csv.py        # Parses Fidelity CSV position exports
      generic_csv.py         # Generic CSV importer
    providers/
      yfinance_provider.py   # Yahoo Finance price fetcher
      alphavantage_provider.py
  engine/                    # Pure computation — ZERO GUI knowledge
    optimization/
      mean_variance.py       # MVO: max_sharpe, min_vol, efficient_risk/return, etc.
      hrp.py                 # Hierarchical Risk Parity
      herc.py                # Hierarchical Equal Risk Contribution
      black_litterman.py     # Black-Litterman model
      tic.py                 # Theory-Implied Correlation
    risk.py                  # Covariance estimation, cov_to_corr, corr_to_cov
    returns.py               # Return estimation (historical mean, CAPM, etc.)
    metrics.py               # Portfolio metrics computation
    factors.py               # Fama-French 3-factor model + exposure analysis
    regime.py                # HMM regime detection + rolling vol regimes
    risk_budgeting.py        # Risk budget optimization + Euler decomposition + ERC
    tax_harvest.py           # Tax-loss harvesting candidates + replacement suggestions
    network/mst.py           # Minimum Spanning Tree (takes DataFrame, NOT networkx Graph)
  backtest/
    engine.py                # BacktestEngine + BacktestConfig + BacktestOutput
    walk_forward.py          # Walk-forward analysis
  gui/
    main_window.py           # MainWindow — creates all panels, controllers, layouts
    theme.py                 # Dark "deep-space" theme stylesheet
    dock_manager.py          # Save/restore dock layouts (method: list_layouts())
    controllers/
      optimization_controller.py  # Wires GUI to engine, runs on QThreadPool
      backtest_controller.py      # Wires GUI to backtest engine
      data_controller.py          # Manages DataManager
      fidelity_controller.py      # Fidelity account integration
    panels/
      base_panel.py          # BasePanel(QDockWidget) — base for all panels
      portfolio_panel.py     # Positions table with P&L coloring
      strategy_lab_panel.py  # Independent optimization/backtest sandbox
      optimization_panel.py  # Optimization config controls
      backtest_panel.py      # Backtest config controls
      factor_panel.py        # Factor exposure chart + regression table
      regime_panel.py        # HMM regime timeline + transition matrix
      risk_budget_panel.py   # Risk contribution chart + budget editor
      tax_harvest_panel.py   # Harvest candidates table + savings chart
      ...                    # 14 more panels (metrics, risk, frontier, etc.)
    dialogs/
      preferences_dialog.py  # Application preferences (theme, data, optimization defaults)
      export_dialog.py       # Multi-format export (CSV, JSON, Excel, PNG)
      ...                    # 6 more dialogs (BL views, constraints, layout, etc.)
scripts/
  build.py                   # PyInstaller build script
meridian.spec                # PyInstaller spec file
.github/workflows/ci.yml     # GitHub Actions CI (pytest + lint)
```

## Architecture Rules

- **Engine layer has zero GUI knowledge.** Never import PySide6 in `engine/` or `backtest/`.
- **Controllers emit Qt Signals** — MainWindow connects them to panel update methods.
- **All long operations run on QThreadPool** via `utils/threading.py:run_in_thread()`.
- **BasePanel** uses `self._layout` (QVBoxLayout) with alias `self.content_layout`.
- **Strategy Lab** has its own isolated controller instances — lab operations never affect the main portfolio.

## Common Pitfalls

- **SQLite cache stores column names as lowercase** — always use case-insensitive column matching when reading from cache:
  ```python
  col_map = {c.lower(): c for c in df.columns}
  ```
- **`compute_mst()` takes a DataFrame**, not a networkx Graph.
- **EFFICIENT_RETURN needs `target_risk`; EFFICIENT_RISK needs `target_return`** — the naming is counter-intuitive.
- **DockManager method is `list_layouts()`**, not `get_saved_layouts()`.
- **Money market symbols** (SPAXX**, FCASH**, etc.) have `AssetType.MONEY_MARKET` — they are included in Portfolio.holdings and total_value but excluded from `Portfolio.symbols` (which is used for optimization/pricing).
- **Backtest initial portfolio value shifts on first day** due to rebalancing costs — use `rel=0.05` tolerance in tests.

## Testing

```bash
python -m pytest tests/ -x -q          # All tests
python -m pytest tests/engine/ -x -q   # Engine tests only
python -m pytest tests/gui/ -x -q      # GUI tests (needs pytest-qt)
```

- 372+ tests across engine, data, backtest, and GUI layers.
- `pytest-qt` must be installed separately (`pip install pytest-qt`).
- Test fixtures are in `tests/conftest.py`.

## Enhancement Plan

See `ENHANCEMENT_PLAN.md` for the full 8-phase roadmap. Completed phases:
- **Phase 1**: Polish & bug fixes — **COMPLETE**
- **Phase 3**: Engine enhancements (factor models, regime detection, risk budgeting, tax harvest) — **COMPLETE**
- **Phase 5**: AI agent integration (Copilot panel, report generation) — **COMPLETE**
- **Phase 7**: Infrastructure (export, preferences, PyInstaller, CI/CD) — **COMPLETE**

## Tech Stack

- Python 3.14+, PySide6 6.10+, pyqtgraph, numpy, pandas, scipy
- SQLite for price caching
- yfinance for market data
