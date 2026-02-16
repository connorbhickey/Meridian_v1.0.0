"""Controller for backtesting workflows — wires GUI to backtest engine."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal

from portopt.backtest.engine import BacktestConfig, BacktestEngine, BacktestOutput
from portopt.backtest.walk_forward import WalkForwardConfig
from portopt.constants import CostModel, CovEstimator, RebalanceFreq, ReturnEstimator
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.metrics import compute_all_metrics
from portopt.utils.threading import run_in_thread

logger = logging.getLogger(__name__)

# Map panel combo values to engine constants
FREQ_MAP = {
    "Daily": RebalanceFreq.DAILY,
    "Weekly": RebalanceFreq.WEEKLY,
    "Monthly": RebalanceFreq.MONTHLY,
    "Quarterly": RebalanceFreq.QUARTERLY,
    "Yearly": RebalanceFreq.YEARLY,
}

COST_MAP = {
    "Zero": "zero",
    "Fixed": "fixed",
    "Proportional": "proportional",
    "Tiered": "tiered",
    "Spread": "spread",
}


class BacktestController(QObject):
    """Manages backtesting workflows with background threading."""

    # Signals
    backtest_complete = Signal(object)           # BacktestOutput
    equity_curve_ready = Signal(object, object)  # dates_epoch, values
    drawdown_ready = Signal(object, object)      # dates_epoch, drawdowns
    metrics_ready = Signal(dict)                 # metrics dict
    trades_ready = Signal(list)                  # list of trade dicts
    attribution_ready = Signal(dict)             # attribution data
    error = Signal(str)
    status_changed = Signal(str)
    progress = Signal(str)

    def __init__(self, data_controller, parent=None):
        super().__init__(parent)
        self.data_controller = data_controller
        self._worker = None
        self._prices: pd.DataFrame | None = None
        self._last_output: BacktestOutput | None = None

    def set_prices(self, prices: pd.DataFrame):
        """Set price data for backtesting."""
        self._prices = prices

    @property
    def last_output(self) -> BacktestOutput | None:
        return self._last_output

    # ── Run Backtest ──────────────────────────────────────────────────

    def run_backtest(self, config: dict):
        """Run backtest with the given configuration from the panel.

        config keys:
            method: str (optimization method name)
            rebalance_freq: str (e.g. "Monthly")
            drift_threshold: float
            cost_model: str (e.g. "Zero")
            cost_rate: float
            walk_forward_enabled: bool
            train_window: int (days)
            test_window: int (days)
            anchored: bool
            initial_value: float
            lookback: int (days)
        """
        if self._prices is None or self._prices.empty:
            self.error.emit("No price data available. Load prices first.")
            return

        self.status_changed.emit("Running backtest...")
        self.progress.emit("Configuring backtest engine...")

        self._worker = run_in_thread(
            self._do_backtest, config,
            on_result=self._on_backtest_done,
            on_error=self._on_error,
        )

    def _do_backtest(self, config: dict) -> BacktestOutput:
        """Run backtest in background thread."""
        method = config.get("method", "max_sharpe")
        freq_str = config.get("rebalance_freq", "Monthly")
        drift = config.get("drift_threshold", 0.0)
        cost_model_str = config.get("cost_model", "Zero")
        cost_rate = config.get("cost_rate", 0.0)
        wf_enabled = config.get("walk_forward_enabled", False)
        train_win = config.get("train_window", 252)
        test_win = config.get("test_window", 63)
        anchored = config.get("anchored", False)
        initial_value = config.get("initial_value", 1_000_000.0)
        lookback = config.get("lookback", 756)

        rebalance_freq = FREQ_MAP.get(freq_str, RebalanceFreq.MONTHLY)
        cost_type = COST_MAP.get(cost_model_str, "zero")
        cost_params = {}
        if cost_type == "proportional":
            cost_params["rate"] = cost_rate
        elif cost_type == "fixed":
            cost_params["cost_per_trade"] = cost_rate
        elif cost_type == "spread":
            cost_params["spread_bps"] = cost_rate * 10000

        wf_config = None
        if wf_enabled:
            wf_config = WalkForwardConfig(
                train_window=train_win,
                test_window=test_win,
                anchored=anchored,
            )

        bt_config = BacktestConfig(
            method=method,
            rebalance_freq=rebalance_freq,
            cost_model_type=cost_type,
            cost_params=cost_params,
            initial_value=initial_value,
            lookback=lookback if lookback > 0 else None,
            drift_threshold=drift if drift > 0 else None,
            walk_forward=wf_config,
        )

        engine = BacktestEngine(self._prices, bt_config)
        output = engine.run()
        return output

    def _on_backtest_done(self, output: BacktestOutput):
        """Handle backtest results on main thread."""
        self._last_output = output
        self.backtest_complete.emit(output)

        # Extract equity curve
        if output.walk_forward_result is not None:
            wf = output.walk_forward_result
            if len(wf.aggregate_values) > 0:
                dates_epoch = self._dates_to_epoch(wf.aggregate_dates)
                self.equity_curve_ready.emit(dates_epoch, wf.aggregate_values)

                # Drawdown
                dd = self._compute_drawdown(wf.aggregate_values)
                self.drawdown_ready.emit(dates_epoch, dd)

                self.status_changed.emit(
                    f"Walk-forward complete: {wf.n_windows} windows | "
                    f"Return={wf.total_return:.2%}"
                )

                # Collect trades from all windows
                all_trades = []
                for w in wf.windows:
                    if w.test_result:
                        all_trades.extend(self._trades_to_dicts(w.test_result.trades))
                if all_trades:
                    self.trades_ready.emit(all_trades)
        elif output.result is not None:
            result = output.result
            if len(result.portfolio_values) > 0:
                dates_epoch = self._dates_to_epoch(result.dates)
                self.equity_curve_ready.emit(dates_epoch, result.portfolio_values)

                dd = self._compute_drawdown(result.portfolio_values)
                self.drawdown_ready.emit(dates_epoch, dd)

                self.status_changed.emit(
                    f"Backtest complete: {result.n_rebalances} rebalances | "
                    f"Return={result.total_return:.2%} | Costs=${result.total_costs:,.2f}"
                )

                trades = self._trades_to_dicts(result.trades)
                if trades:
                    self.trades_ready.emit(trades)

        # Metrics
        if output.metrics:
            self.metrics_ready.emit(output.metrics)

        # Attribution
        if output.contribution is not None:
            try:
                contrib_dict = {
                    "type": "contribution",
                    "data": output.contribution.to_dict() if hasattr(output.contribution, "to_dict") else {},
                }
                self.attribution_ready.emit(contrib_dict)
            except Exception:
                pass

    def _on_error(self, msg: str):
        logger.error("Backtest error: %s", msg)
        self.status_changed.emit("Backtest failed")
        self.error.emit(f"Backtest error: {msg}")

    # ── Fetch + Backtest ──────────────────────────────────────────────

    def fetch_and_backtest(self, symbols: list[str], config: dict,
                           lookback_days: int = 1260):
        """Fetch prices then run backtest."""
        self.status_changed.emit(f"Fetching prices for {len(symbols)} symbols...")
        start = date.today() - timedelta(days=lookback_days)
        self._pending_config = config

        self._worker = run_in_thread(
            self.data_controller.data_manager.get_multiple_prices,
            symbols, start,
            on_result=self._on_prices_for_bt,
            on_error=self._on_error,
        )

    def _on_prices_for_bt(self, prices_dict: dict):
        """Prices fetched — build DataFrame and run backtest."""
        frames = {}
        for sym, df in prices_dict.items():
            if not df.empty and "Close" in df.columns:
                frames[sym] = df["Close"]
            elif not df.empty and "Adj Close" in df.columns:
                frames[sym] = df["Adj Close"]

        if not frames:
            self.error.emit("No price data retrieved")
            return

        self._prices = pd.DataFrame(frames).dropna()
        self.status_changed.emit(f"Prices loaded: {len(self._prices.columns)} symbols")

        config = getattr(self, "_pending_config", {})
        self.run_backtest(config)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dates_to_epoch(dates: list) -> np.ndarray:
        """Convert date list to epoch seconds for pyqtgraph."""
        epochs = []
        for d in dates:
            if hasattr(d, "timestamp"):
                epochs.append(d.timestamp())
            elif isinstance(d, date):
                from datetime import datetime
                epochs.append(datetime.combine(d, datetime.min.time()).timestamp())
            else:
                epochs.append(float(d))
        return np.array(epochs)

    @staticmethod
    def _compute_drawdown(values) -> np.ndarray:
        """Compute drawdown series from portfolio values."""
        vals = np.asarray(values, dtype=float)
        if len(vals) == 0:
            return np.array([])
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak
        return dd

    @staticmethod
    def _trades_to_dicts(trades) -> list[dict]:
        """Convert Trade objects to dicts for the blotter panel."""
        result = []
        for t in trades:
            result.append({
                "date": str(t.date),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": abs(t.quantity),
                "price": t.price,
                "cost": t.cost,
                "weight_after": t.weight_after,
            })
        return result
