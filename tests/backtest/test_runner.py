"""Tests for backtest runner and walk-forward analysis."""

import numpy as np
import pandas as pd
import pytest

from portopt.backtest.costs import ProportionalCost, ZeroCost
from portopt.backtest.engine import BacktestConfig, BacktestEngine, BacktestOutput
from portopt.backtest.rebalancer import RebalanceSchedule
from portopt.backtest.results import SingleRunResult, WalkForwardResult
from portopt.backtest.runner import make_optimizer_fn, run_backtest
from portopt.backtest.walk_forward import WalkForwardConfig, generate_windows
from portopt.constants import RebalanceFreq


# ── Runner ────────────────────────────────────────────────────────────

class TestRunBacktest:
    def test_basic_run(self, prices_5):
        optimizer_fn = make_optimizer_fn(method="max_sharpe")
        result = run_backtest(
            prices=prices_5,
            optimizer_fn=optimizer_fn,
            initial_value=1_000_000.0,
        )
        assert isinstance(result, SingleRunResult)
        assert len(result.portfolio_values) > 0
        assert result.portfolio_values[0] == pytest.approx(1_000_000.0, rel=0.05)

    def test_with_cost_model(self, prices_5):
        optimizer_fn = make_optimizer_fn(method="min_volatility")
        cost_model = ProportionalCost(rate=0.001)
        result = run_backtest(
            prices=prices_5,
            optimizer_fn=optimizer_fn,
            cost_model=cost_model,
            initial_value=1_000_000.0,
        )
        assert result.total_costs > 0

    def test_monthly_rebalance(self, prices_5):
        optimizer_fn = make_optimizer_fn(method="max_sharpe")
        schedule = RebalanceSchedule(frequency=RebalanceFreq.MONTHLY)
        result = run_backtest(
            prices=prices_5,
            optimizer_fn=optimizer_fn,
            schedule=schedule,
        )
        # Monthly rebalance over ~2 years should have ~24 rebalances
        assert 10 < result.n_rebalances < 30


class TestMakeOptimizerFn:
    def test_returns_callable(self):
        fn = make_optimizer_fn(method="max_sharpe")
        assert callable(fn)

    def test_fn_produces_weights(self, prices_5):
        fn = make_optimizer_fn(method="inverse_variance")
        cov = prices_5.pct_change().dropna().cov() * 252
        weights = fn(cov, prices_5.pct_change().dropna())
        assert isinstance(weights, dict)
        assert len(weights) == 5
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01


# ── Walk-Forward ──────────────────────────────────────────────────────

class TestGenerateWindows:
    def test_rolling_windows(self, prices_5):
        config = WalkForwardConfig(train_window=126, test_window=63)
        windows = generate_windows(prices_5.index, config)
        assert len(windows) > 0

    def test_anchored_grows(self, prices_5):
        config = WalkForwardConfig(train_window=126, test_window=63, anchored=True)
        windows = generate_windows(prices_5.index, config)
        assert len(windows) > 0


# ── BacktestEngine ────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_single_run(self, prices_5):
        config = BacktestConfig(method="max_sharpe")
        engine = BacktestEngine(prices_5, config)
        output = engine.run()

        assert isinstance(output, BacktestOutput)
        assert output.result is not None
        assert len(output.metrics) > 0

    def test_walk_forward_run(self, prices_5):
        config = BacktestConfig(
            method="inverse_variance",
            walk_forward=WalkForwardConfig(
                train_window=126,
                test_window=63,
            ),
        )
        engine = BacktestEngine(prices_5, config)
        output = engine.run()

        assert isinstance(output, BacktestOutput)
        assert output.walk_forward_result is not None

    def test_with_costs(self, prices_5):
        config = BacktestConfig(
            method="max_sharpe",
            cost_model_type="proportional",
            cost_params={"rate": 0.001},
        )
        engine = BacktestEngine(prices_5, config)
        output = engine.run()
        assert output.result is not None
