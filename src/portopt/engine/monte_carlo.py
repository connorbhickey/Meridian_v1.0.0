"""Monte Carlo simulation engine: parametric GBM and block bootstrap.

Generates thousands of possible future portfolio paths to produce
confidence bands on wealth trajectories and distributions of risk metrics.
Zero GUI imports — pure computation module.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from portopt.constants import MCSimMethod
from portopt.data.models import MonteCarloConfig, MonteCarloResult
from portopt.engine.metrics import compute_all_metrics
from portopt.engine.risk import nearest_positive_definite

logger = logging.getLogger(__name__)

# Subset of metrics to track across simulations (keeps memory reasonable)
_TRACKED_METRICS = (
    "sharpe_ratio",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "cvar_95",
)


def run_monte_carlo(
    weights: dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    config: MonteCarloConfig,
    historical_returns: pd.DataFrame | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation.

    Args:
        weights: Portfolio weights {symbol: weight}, must align with mu/cov index.
        mu: Annualized expected returns (from estimate_returns).
        cov: Annualized covariance matrix (from estimate_covariance).
        config: Simulation configuration.
        historical_returns: Daily simple returns DataFrame (required for BOOTSTRAP).
        progress_cb: Optional callback for progress messages.

    Returns:
        MonteCarloResult with percentile curves, metrics distributions, shortfall.
    """
    # Align weights to mu index order
    symbols = list(mu.index)
    w = np.array([weights.get(s, 0.0) for s in symbols])
    w = w / w.sum()  # normalize

    horizon = config.horizon_days
    n_sims = config.n_sims

    if config.method == MCSimMethod.PARAMETRIC:
        if progress_cb:
            progress_cb("Preparing parametric simulation...")
        all_paths = _run_parametric(w, mu.values, cov.values, config, progress_cb)
    elif config.method == MCSimMethod.BOOTSTRAP:
        if historical_returns is None:
            raise ValueError(
                "historical_returns is required for BOOTSTRAP simulation method"
            )
        if progress_cb:
            progress_cb("Preparing bootstrap simulation...")
        # Align historical returns to same symbol order
        hist = historical_returns[symbols].values
        all_paths = _run_bootstrap(w, hist, config, progress_cb)
    else:
        raise ValueError(f"Unknown simulation method: {config.method}")

    # Compute percentile curves: shape (horizon+1, n_percentiles)
    if progress_cb:
        progress_cb("Computing percentile bands...")
    pcts = np.array(config.percentiles)
    equity_percentiles = np.percentile(all_paths, pcts, axis=0).T

    # Build forward dates
    dates = _build_forward_dates(horizon)

    # Metrics distributions
    if progress_cb:
        progress_cb("Computing metrics distributions...")
    metrics_dist = _compute_metrics_distribution(all_paths, config, progress_cb)

    # Shortfall analysis
    terminal_values = all_paths[:, -1]
    threshold = config.initial_value * (1 - config.spending_rate)
    shortfall_prob = _compute_shortfall(terminal_values, threshold)

    method_name = "parametric" if config.method == MCSimMethod.PARAMETRIC else "bootstrap"

    return MonteCarloResult(
        equity_percentiles=equity_percentiles,
        percentile_labels=config.percentiles,
        dates=dates,
        metrics_distributions=metrics_dist,
        shortfall_probability=shortfall_prob,
        shortfall_threshold=threshold,
        n_sims=n_sims,
        method=method_name,
        config=config,
        metadata={
            "symbols": symbols,
            "mean_terminal": float(np.mean(terminal_values)),
            "median_terminal": float(np.median(terminal_values)),
        },
    )


def _run_parametric(
    w: np.ndarray,
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    config: MonteCarloConfig,
    progress_cb: Callable[[str], None] | None,
) -> np.ndarray:
    """GBM path generation. Returns shape (n_sims, horizon+1).

    Uses same Ito-corrected GBM as conftest._make_prices:
        log_ret = (mu - 0.5*sigma^2)*dt + L @ z
    """
    freq = config.frequency
    n_sims = config.n_sims
    horizon = config.horizon_days
    n_assets = len(w)

    # Convert to daily
    mu_daily = mu_annual / freq
    cov_daily = cov_annual / freq

    # Ensure PSD for Cholesky
    cov_daily_pd = nearest_positive_definite(cov_daily)
    L = np.linalg.cholesky(cov_daily_pd)
    diag_var = np.diag(cov_daily_pd)

    # Drift term per asset (Ito correction)
    drift = mu_daily - 0.5 * diag_var  # shape (n_assets,)

    paths = np.empty((n_sims, horizon + 1))
    paths[:, 0] = config.initial_value

    rng = np.random.default_rng(np.random.get_state()[1][0])

    for t in range(1, horizon + 1):
        z = rng.standard_normal((n_assets, n_sims))
        # Per-asset log returns: shape (n_assets, n_sims)
        log_rets = drift[:, None] + L @ z
        # Portfolio log return: shape (n_sims,)
        port_log_ret = w @ log_rets
        paths[:, t] = paths[:, t - 1] * np.exp(port_log_ret)

        if progress_cb and t % 50 == 0:
            progress_cb(f"Simulating day {t}/{horizon}...")

    return paths


def _run_bootstrap(
    w: np.ndarray,
    hist_returns: np.ndarray,
    config: MonteCarloConfig,
    progress_cb: Callable[[str], None] | None,
) -> np.ndarray:
    """Block bootstrap path generation. Returns shape (n_sims, horizon+1).

    Resamples blocks of historical portfolio returns, preserving
    autocorrelation within each block.
    """
    n_sims = config.n_sims
    horizon = config.horizon_days
    block_size = config.block_size
    T = hist_returns.shape[0]

    # Compute portfolio daily returns from historical asset returns
    port_rets = hist_returns @ w  # shape (T,)

    rng = np.random.default_rng(np.random.get_state()[1][0])

    # Number of blocks needed to cover the horizon
    n_blocks = int(np.ceil(horizon / block_size)) + 1

    # Sample block start indices: shape (n_sims, n_blocks)
    max_start = max(T - block_size, 1)
    starts = rng.integers(0, max_start, size=(n_sims, n_blocks))

    # Expand to full index matrix: shape (n_sims, n_blocks * block_size)
    offsets = np.arange(block_size)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_sims, -1)
    indices = np.clip(indices, 0, T - 1)

    # Trim to horizon length and index into portfolio returns
    resampled = port_rets[indices[:, :horizon]]  # shape (n_sims, horizon)

    # Build wealth paths
    paths = np.empty((n_sims, horizon + 1))
    paths[:, 0] = config.initial_value
    paths[:, 1:] = config.initial_value * np.cumprod(1 + resampled, axis=1)

    if progress_cb:
        progress_cb(f"Bootstrap complete: {n_sims} paths x {horizon} days")

    return paths


def _compute_metrics_distribution(
    all_paths: np.ndarray,
    config: MonteCarloConfig,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-simulation metrics and return sorted distributions."""
    n_sims = all_paths.shape[0]
    collectors: dict[str, list[float]] = {m: [] for m in _TRACKED_METRICS}

    for i in range(n_sims):
        # Daily returns from wealth path
        daily_rets = all_paths[i, 1:] / all_paths[i, :-1] - 1
        metrics = compute_all_metrics(
            daily_rets,
            risk_free_rate=config.risk_free_rate,
            frequency=config.frequency,
        )
        for m in _TRACKED_METRICS:
            collectors[m].append(metrics.get(m, 0.0))

        if progress_cb and (i + 1) % 200 == 0:
            progress_cb(f"Computing metrics: {i + 1}/{n_sims}...")

    return {m: np.sort(np.array(v)) for m, v in collectors.items()}


def _compute_shortfall(
    terminal_values: np.ndarray,
    threshold: float,
) -> float:
    """P(terminal_value < threshold)."""
    return float(np.mean(terminal_values < threshold))


def _build_forward_dates(horizon: int) -> list[date]:
    """Build a list of business-day-approximate forward dates."""
    dates = pd.bdate_range(start=date.today(), periods=horizon + 1)
    return [d.date() for d in dates]
