# Meridian — Agent Quick Reference

## What is this project?

Meridian is a desktop quantitative portfolio terminal built with Python and PySide6. It provides portfolio optimization, backtesting, risk analysis, and AI-assisted insights in a Bloomberg-style dockable panel interface.

## Repository Layout

- `src/portopt/` — 98 source files across engine, data, backtest, gui, and utils
- `tests/` — 38 test files, 842+ tests
- `.github/workflows/` — CI (lint/test/build) + Release (tag-triggered)
- `meridian.spec` + `scripts/build.py` — PyInstaller packaging

## Key Commands

| Task | Command |
|------|---------|
| Launch app | `python -m portopt.app` |
| Check install | `python -m portopt.app --check` |
| Run all tests | `python -m pytest tests/ -x -q` |
| Run lint | `python -m ruff check src/ --select E,W --ignore E501` |
| Build exe | `pyinstaller meridian.spec --noconfirm` |
| Create release | `git tag v1.x.x && git push origin v1.x.x` |

## Architecture Layers

```
GUI (PySide6)  →  Controllers (signals)  →  Engine (pure Python)
     ↓                    ↓                        ↓
  Panels (30)      QThreadPool workers       numpy/scipy/cvxpy
  Dialogs (12)     DataManager               No GUI imports ever
```

## Critical Rules

1. **Engine has zero GUI knowledge** — never import PySide6 in `engine/` or `backtest/`
2. **QSettings via `config.get_settings()`** — INI format, not Windows registry
3. **BasePanel base class** — use `self._layout` / `self.content_layout`
4. **SQLite cache columns are lowercase** — always do case-insensitive matching
5. **All long ops on QThreadPool** — never block the main thread

## API Keys (OS Keyring)

| Key | Service |
|-----|---------|
| `anthropic_api_key` | AI Copilot (Claude) |
| `fred_api_key` | FRED macro data |
| `tiingo_api_key` | Tiingo prices |
| `alpha_vantage_api_key` | Alpha Vantage prices |

Managed via `utils/credentials.py` and the in-app API Key dialog.

## Data Flow

1. **Import**: CSV/OFX -> importers -> Portfolio model (Fidelity > Schwab > Robinhood > Generic)
2. **Prices**: DataManager -> YFinance > Tiingo > AlphaVantage (fallback chain)
3. **Cache**: SQLite via `data/cache.py` (lowercase column names)
4. **Engine**: Receives DataFrames, returns OptimizationResult / metrics dicts
5. **Display**: Controllers emit signals -> MainWindow routes to panels

## Testing Notes

- `pytest-qt` required for GUI tests (install separately)
- `hmmlearn` required for regime detection tests
- Backtest value tolerance: use `rel=0.05` for first-day shift
- `AssetType.STOCK` not `EQUITY`

## CI/CD

- **CI**: lint (ruff) -> test (Ubuntu+Windows, Python 3.12+3.13) -> build (PyInstaller Windows)
- **Release**: Push `v*` tag -> build + zip + GitHub Release with changelog
- All GitHub Actions pinned to SHA
