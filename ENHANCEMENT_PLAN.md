# Meridian Enhancement Plan

## What Meridian Is Today

A **desktop quantitative portfolio terminal** built on PySide6 with:
- 13 optimization methods (MVO, HRP, HERC, BL, TIC, Risk Parity, etc.)
- Walk-forward backtesting with 6 cost models, Brinson attribution, cross-validation, PBO
- Fidelity brokerage integration (Playwright-based auto-import)
- 20 GUI panels with a dark "deep-space" theme
- SQLite price cache, yfinance + AlphaVantage data providers
- 208+ tests across all layers

---

## Phase 1: Polish & Bug Fixes (Low Risk, High Impact)

### 1.1 Fix `Holding.weight` always returning `0.0`
`models.py:Holding.weight` is a property that always returns `0.0` with a comment saying it's set externally. This is fragile — weight should be computed from `market_value / portfolio.total_value` when the holding is part of a portfolio context, or accept a setter.

### 1.2 Error handling gaps in controllers
- `optimization_controller.py` catches generic `Exception` and emits `error` signal, but the error message is just `str(e)` — no traceback. Add traceback logging for debugging.
- `data_controller.py` has no retry logic for transient network failures in yfinance. Add exponential backoff for `ConnectionError`/`Timeout`.

### 1.3 Console panel logging
The `ConsolePanel` exists but is barely wired. Route all controller `status_changed` and `error` signals into it so users can see a running log of what the app is doing, with timestamps and severity levels.

### 1.4 Price chart panel completion
`PriceChartPanel` and `WatchlistPanel` appear to be stub panels. Flesh them out:
- **PriceChart**: OHLCV candlestick chart for any symbol, with volume bars, moving averages, and Bollinger bands
- **Watchlist**: Editable symbol list with live price, change%, and sparkline columns

### 1.5 Keyboard shortcuts
No keyboard shortcuts exist. Add:
- `Ctrl+R` — Run optimization
- `Ctrl+B` — Run backtest
- `Ctrl+1..9` — Switch to panel by index
- `Ctrl+S` — Save layout
- `F5` — Refresh data

---

## Phase 2: Data Layer Enhancements (Medium Risk, High Impact)

### 2.1 Multi-source data providers
Currently yfinance is primary with AlphaVantage fallback. Add:
- **FRED provider** — interest rates, inflation, macro indicators (for factor models)
- **Tiingo provider** — more reliable than yfinance for some symbols
- **CSV/Excel import** — generic time series import for custom data (private equity, real estate, etc.)

### 2.2 Fundamental data integration
Add a `FundamentalProvider` that fetches:
- P/E, P/B, dividend yield, market cap, sector, industry
- Useful for: factor-based screens, sector constraint auto-population, fundamental-weighted portfolios

### 2.3 Real-time price streaming
Replace the 3-second QTimer label hack with a proper `QTimer`-based price poller (30s interval) that updates:
- Portfolio panel current prices and P&L
- Watchlist panel
- Status bar portfolio value

### 2.4 Data quality dashboard
Add a panel or dialog showing:
- Coverage: which symbols have data, date range, missing days
- Staleness: how old the cached data is
- Anomalies: outlier returns, zero-volume days, suspected stock splits

---

## Phase 3: Engine Enhancements (Medium Risk, High Value)

### 3.1 Factor models
Add a `FactorModel` engine module:
- **Fama-French 3-factor** (market, size, value) and **5-factor** (+ profitability, investment)
- **Custom factor construction** — allow users to define factors from any time series
- Factor exposure analysis for current portfolio
- Factor-constrained optimization (e.g., "neutral to value factor")

### 3.2 Regime detection
Add regime-aware optimization:
- **Hidden Markov Model** (2-3 states: bull/bear/crisis) using `hmmlearn`
- **Rolling volatility regime** — simple threshold-based
- Regime-conditional covariance and returns for optimization
- Backtest regime overlay on equity curve

### 3.3 Risk budgeting
Extend beyond Risk Parity to full **risk budgeting**:
- User-specified risk contribution targets per asset (e.g., "AAPL should contribute 10% of portfolio risk")
- Equal Risk Contribution (ERC) as a special case
- Risk contribution decomposition visualization

### 3.4 Monte Carlo simulation
Add Monte Carlo engine:
- **Parametric** — simulate from estimated mu/Sigma
- **Bootstrap** — resample historical returns with block bootstrap
- **Distribution of terminal wealth** — fan chart visualization
- **Probability of ruin/shortfall** — given a spending rate, what's the probability of running out

### 3.5 Tax-loss harvesting optimizer
For taxable accounts:
- Identify positions with unrealized losses
- Suggest tax-loss harvesting swaps (correlated replacements)
- Compute tax alpha from harvesting

---

## Phase 4: Backtest Enhancements (Medium Risk, High Value)

### 4.1 Benchmark comparison
Currently the backtest engine supports benchmarks but the GUI doesn't expose it well. Add:
- Benchmark selector (SPY, 60/40, equal-weight, custom)
- Side-by-side equity curves
- Relative performance chart (portfolio / benchmark)
- Rolling alpha/beta/information ratio charts

### 4.2 Strategy comparison framework
Allow running multiple strategies simultaneously:
- Different optimization methods with same universe
- Parameter sweeps (e.g., "test rebalance monthly vs quarterly")
- Statistical significance tests (bootstrap paired t-test of strategy returns)

### 4.3 Stress testing
Add historical stress test scenarios:
- **Named events**: 2008 GFC, COVID crash, 2022 rate hikes, dot-com bust
- **Custom scenarios**: user-defined return shocks per asset/sector
- **Reverse stress test**: "what return scenario would cause a 20% drawdown?"

### 4.4 Execution simulation improvements
- **Slippage model** — market impact based on volume
- **Capacity analysis** — at what AUM does the strategy degrade
- **Partial fill simulation** — not all trades execute at target price

---

## Phase 5: AI Agent Integration (High Impact, Higher Risk)

### 5.1 Portfolio AI Copilot
Integrate an LLM-powered assistant directly into the terminal:
- **Natural language optimization**: "Optimize my portfolio for maximum Sharpe with no more than 5% in any single stock"
- **Explain results**: "Why did HRP allocate 30% to bonds?" — The AI reads the correlation matrix, clustering, and explains the logic
- **What-if queries**: "What happens if I add TSLA to the portfolio?" — Runs optimization and summarizes delta

**Implementation**: Claude API via `anthropic` Python SDK. A `CopilotPanel` with chat interface. The agent has tool-use access to all engine functions.

### 5.2 Market Intelligence Agent
An agent that:
- Monitors news for portfolio-relevant events (earnings, M&A, macro)
- Flags regime changes or unusual correlation breakdowns
- Generates a daily briefing: "Your portfolio's tech concentration increased risk — NVDA and AMD correlation jumped from 0.6 to 0.85 this month"

### 5.3 Automated rebalance recommendations
AI-powered rebalance suggestions:
- "Based on current drift, regime, and tax implications, here's the recommended rebalance"
- Estimates trading cost vs. tracking error tradeoff
- Suggests tax-lot selection for taxable accounts

### 5.4 Report generation
One-click professional PDF/HTML report:
- Executive summary (AI-written)
- Performance attribution
- Risk analysis with charts
- Forward-looking scenarios
- All charts exported as high-res images

---

## Phase 6: UX & Visualization (Medium Risk, High Polish)

### 6.1 Interactive efficient frontier
Current frontier is static. Make it interactive:
- Click any point on the frontier to see the weights at that risk/return
- Drag a slider along the frontier
- Show current portfolio position on the frontier
- Show individual assets as scatter points

### 6.2 Sankey flow diagram
For rebalancing: visualize capital flows from current to target allocation as a Sankey diagram.

### 6.3 Rolling window analytics
Add rolling charts:
- Rolling Sharpe, Sortino, max drawdown
- Rolling correlation between any two assets
- Rolling factor exposures

### 6.4 Dashboard customization
- User-configurable metric cards (drag-and-drop which metrics appear)
- Custom color themes beyond deep-space
- Panel presets: "Optimization Focus", "Risk Monitor", "Backtest Lab", "Trading Desk"

### 6.5 Multi-monitor support
Detect multiple monitors and allow panels to be dragged to separate screens with proper DPI scaling.

---

## Phase 7: Infrastructure & Distribution (Low Risk, Operational)

### 7.1 Electron wrapper (started in package.json)
The `package.json` already exists. Complete the Electron shell:
- `electron/main.js` — spawn Python process, embed in BrowserWindow (or use native PySide6 as child process)
- Auto-updater via electron-builder
- System tray icon with quick-launch

### 7.2 PyInstaller/Nuitka packaging
Alternatively (and probably more practical for a PySide6 app):
- Single `.exe` via PyInstaller with `--onefile`
- Include all assets, fonts, icons
- NSIS installer (already configured in package.json)

### 7.3 Settings/preferences system
Formalize user preferences:
- Default optimization method, covariance estimator, lookback
- Default benchmark
- Data refresh interval
- API keys management (currently in `.env`)
- Persist in QSettings (already partially done)

### 7.4 Export capabilities
- Export any table to CSV/Excel
- Export any chart to PNG/SVG
- Export optimization results to JSON
- Export full session state for reproducibility

### 7.5 CI/CD
- GitHub Actions: run `pytest` on push
- Pre-commit hooks: ruff lint + format
- Auto-build Windows installer on tagged releases

---

## Phase 8: Multi-Account & Brokerage (High Value, High Complexity)

### 8.1 Multi-brokerage support
Extend beyond Fidelity:
- **Schwab** — CSV import (similar format to Fidelity)
- **Interactive Brokers** — via TWS API for live trading
- **Robinhood** — CSV import
- Generic OFX/QFX import for any brokerage

### 8.2 Account-level optimization
Currently optimization is portfolio-wide. Add:
- Per-account constraints (e.g., "IRA can hold anything, taxable should avoid high-dividend stocks")
- Tax-aware asset location (bonds in tax-deferred, growth in taxable)
- Household-level optimization across accounts

### 8.3 Trade execution
For Interactive Brokers users:
- Generate order list from rebalance
- Submit via IB API with limit orders
- Track fill status in trading blotter panel

---

## Priority Order

1. **Phase 1** (1-2 days) — Polish fixes that make the existing app more solid
2. **Phase 5.1 + 5.4** (2-3 days) — AI Copilot + Report generation are the highest-differentiating features
3. **Phase 3.4** (1 day) — Monte Carlo simulation is a standard expectation for portfolio tools
4. **Phase 4.1 + 4.3** (1-2 days) — Benchmark comparison and stress testing
5. **Phase 6.1 + 6.3** (1-2 days) — Interactive frontier and rolling analytics
6. **Phase 7.2 + 7.4** (1 day) — Packaging and export for distribution
