"""Merton Jump-Diffusion Monte Carlo simulator.

dS/S = (μ − λk)dt + σ·dW + J·dN
  W uses Student-t(ν) for fat tails
  J ~ LogN(jumpMu, jumpSig²)
  N ~ Poisson(λ)

Reference: Merton 1976, "Option pricing when underlying stock returns
are discontinuous", Journal of Financial Economics 3(1-2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from portopt.engine.prediction.prng import percentile


@dataclass
class MJDParams:
    """Parameters for Merton Jump-Diffusion simulation."""
    nu: float = 5.0           # Student-t degrees of freedom
    lambda_: float = 2.0      # Poisson jump intensity (per year)
    jump_mu: float = -0.02    # Mean jump size (log)
    jump_sig: float = 0.08    # Jump volatility
    earnings_jump: float = 0.0   # Earnings event vol (0 = no event)
    earnings_day: int = -1       # Day of earnings within horizon


@dataclass
class MCResult:
    """Monte Carlo simulation output."""
    est: float = 0.0     # Median (P50)
    mean: float = 0.0
    p5: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0


def mjd_simulate(
    s0: float,
    mu: float,
    sig: float,
    trading_days: int,
    n_sims: int,
    seed: int,
    params: MJDParams | None = None,
) -> np.ndarray:
    """Run Merton Jump-Diffusion Monte Carlo simulation (vectorized).

    Args:
        s0: Current price
        mu: Annual drift (log-return)
        sig: Annual volatility
        trading_days: Horizon in trading days
        n_sims: Number of simulation paths
        seed: RNG seed for reproducibility
        params: MJD parameters (defaults if None)

    Returns:
        1-D array of terminal prices (length = n_sims)
    """
    if params is None:
        params = MJDParams()

    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)

    # Compensator: k = E[e^J - 1]
    k = math.exp(params.jump_mu + 0.5 * params.jump_sig ** 2) - 1.0
    drift = (mu - 0.5 * sig * sig - params.lambda_ * k) * dt
    diff = sig * math.sqrt(dt)

    # Student-t(nu) innovations, normalized to unit variance
    # numpy standard_t has variance = nu/(nu-2), so scale by sqrt((nu-2)/nu)
    raw_t = rng.standard_t(params.nu, size=(n_sims, trading_days))
    if params.nu > 2:
        raw_t *= math.sqrt((params.nu - 2.0) / params.nu)

    # Daily log-returns: drift + diffusion
    log_returns = drift + diff * raw_t

    # Poisson jumps (vectorized)
    jump_mask = rng.random(size=(n_sims, trading_days)) < params.lambda_ * dt
    jump_normals = rng.standard_normal(size=(n_sims, trading_days))
    log_returns += jump_mask * (params.jump_mu + params.jump_sig * jump_normals)

    # Earnings event shock on a specific day
    if 0 <= params.earnings_day < trading_days and params.earnings_jump > 0:
        log_returns[:, params.earnings_day] += (
            rng.standard_normal(n_sims) * params.earnings_jump
        )

    # Terminal prices
    return s0 * np.exp(np.sum(log_returns, axis=1))


def mc_percentiles(terminal_prices: np.ndarray) -> MCResult:
    """Extract standard percentiles from terminal price distribution."""
    arr = terminal_prices
    mean_val = float(np.mean(arr))
    return MCResult(
        est=round(percentile(arr, 50), 2),
        mean=round(mean_val, 2),
        p5=round(percentile(arr, 5), 2),
        p10=round(percentile(arr, 10), 2),
        p25=round(percentile(arr, 25), 2),
        p50=round(percentile(arr, 50), 2),
        p75=round(percentile(arr, 75), 2),
        p90=round(percentile(arr, 90), 2),
        p95=round(percentile(arr, 95), 2),
    )


def build_histogram(arr: np.ndarray, bins: int = 40) -> list[dict]:
    """Build histogram from terminal prices (P2–P98 range).

    Returns list of {c: center, d: density_%} dicts.
    """
    lo = percentile(arr, 2)
    hi = percentile(arr, 98)
    step = (hi - lo) / bins
    if step <= 0:
        return [{"c": round(lo, 2), "d": 100.0}]

    n = len(arr)
    hist = []
    for i in range(bins):
        a0 = lo + i * step
        b0 = a0 + step
        cnt = int(np.sum((arr >= a0) & (arr < b0)))
        hist.append({
            "c": round((a0 + b0) / 2, 2),
            "d": round(cnt / n * 100, 2),
        })
    return hist


def prob_above(terminal_prices: np.ndarray, threshold: float) -> float:
    """Compute P(terminal > threshold) as a percentage."""
    return round(float(np.sum(terminal_prices > threshold) / len(terminal_prices) * 100), 1)
