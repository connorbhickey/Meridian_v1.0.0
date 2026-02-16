"""Performance metrics: 20+ measures for return, risk, risk-adjusted, and distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_all_metrics(
    returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    risk_free_rate: float = 0.04,
    frequency: int = 252,
) -> dict[str, float]:
    """Compute all performance metrics for a return series.

    Args:
        returns: Daily return series.
        benchmark_returns: Optional benchmark daily returns.
        risk_free_rate: Annual risk-free rate.
        frequency: Trading days per year.

    Returns:
        Dictionary of metric name -> value.
    """
    r = np.asarray(returns, dtype=float)
    daily_rf = risk_free_rate / frequency

    metrics = {}

    # ── Return metrics ────────────────────────────────────
    metrics["total_return"] = total_return(r)
    metrics["annualized_return"] = annualized_return(r, frequency)
    metrics["cagr"] = cagr(r, frequency)

    # ── Risk metrics ──────────────────────────────────────
    metrics["annualized_volatility"] = annualized_volatility(r, frequency)
    metrics["downside_deviation"] = downside_deviation(r, daily_rf, frequency)
    metrics["max_drawdown"] = max_drawdown(r)
    metrics["avg_drawdown"] = avg_drawdown(r)
    metrics["max_drawdown_duration"] = max_drawdown_duration(r)
    metrics["var_95"] = value_at_risk(r, 0.05)
    metrics["var_99"] = value_at_risk(r, 0.01)
    metrics["cvar_95"] = conditional_var(r, 0.05)
    metrics["cvar_99"] = conditional_var(r, 0.01)
    metrics["ulcer_index"] = ulcer_index(r)

    # ── Risk-adjusted metrics ─────────────────────────────
    metrics["sharpe_ratio"] = sharpe_ratio(r, daily_rf, frequency)
    metrics["sortino_ratio"] = sortino_ratio(r, daily_rf, frequency)
    metrics["calmar_ratio"] = calmar_ratio(r, frequency)
    metrics["omega_ratio"] = omega_ratio(r, daily_rf)
    metrics["tail_ratio"] = tail_ratio(r)

    # ── Distribution metrics ──────────────────────────────
    metrics["skewness"] = float(stats.skew(r))
    metrics["kurtosis"] = float(stats.kurtosis(r))
    metrics["best_day"] = float(np.max(r))
    metrics["worst_day"] = float(np.min(r))
    metrics["positive_days_pct"] = float(np.mean(r > 0) * 100)
    metrics["profit_factor"] = profit_factor(r)

    # ── Tracking metrics (if benchmark provided) ──────────
    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float)
        min_len = min(len(r), len(b))
        r_t, b_t = r[:min_len], b[:min_len]
        metrics["alpha"], metrics["beta"] = alpha_beta(r_t, b_t, daily_rf, frequency)
        metrics["tracking_error"] = tracking_error(r_t, b_t, frequency)
        metrics["information_ratio"] = information_ratio(r_t, b_t, frequency)
        metrics["treynor_ratio"] = treynor_ratio(r_t, b_t, daily_rf, frequency)
        metrics["up_capture"] = up_capture(r_t, b_t)
        metrics["down_capture"] = down_capture(r_t, b_t)

    return metrics


# ══════════════════════════════════════════════════════════════════════
# Individual metric functions
# ══════════════════════════════════════════════════════════════════════


def total_return(returns: np.ndarray) -> float:
    """Cumulative total return."""
    return float(np.prod(1 + returns) - 1)


def annualized_return(returns: np.ndarray, frequency: int = 252) -> float:
    """Annualized mean return."""
    return float(np.mean(returns) * frequency)


def cagr(returns: np.ndarray, frequency: int = 252) -> float:
    """Compound Annual Growth Rate."""
    # Use log-sum-exp to avoid overflow for long return sequences
    total = np.exp(np.sum(np.log1p(returns)))
    n_years = len(returns) / frequency
    if n_years <= 0 or total <= 0 or not np.isfinite(total):
        return 0.0
    result = total ** (1 / n_years) - 1
    return float(result) if np.isfinite(result) else 0.0


def annualized_volatility(returns: np.ndarray, frequency: int = 252) -> float:
    """Annualized standard deviation of returns."""
    return float(np.std(returns, ddof=1) * np.sqrt(frequency))


def downside_deviation(returns: np.ndarray, mar: float = 0.0, frequency: int = 252) -> float:
    """Downside deviation (semi-deviation below MAR)."""
    downside = np.minimum(returns - mar, 0)
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(frequency))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from peak to trough."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))


def avg_drawdown(returns: np.ndarray) -> float:
    """Average drawdown."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    # Find individual drawdown periods
    in_dd = drawdowns < 0
    if not np.any(in_dd):
        return 0.0
    return float(np.mean(drawdowns[in_dd]))


def max_drawdown_duration(returns: np.ndarray) -> int:
    """Maximum drawdown duration in periods."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    in_drawdown = cumulative < running_max
    max_dur = 0
    current_dur = 0
    for dd in in_drawdown:
        if dd:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical Value at Risk (left-tail percentile)."""
    return float(np.percentile(returns, alpha * 100))


def conditional_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall)."""
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def ulcer_index(returns: np.ndarray) -> float:
    """Ulcer Index — RMS of percentage drawdowns."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    dd_pct = (cumulative / running_max - 1) * 100
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def sharpe_ratio(returns: np.ndarray, daily_rf: float = 0.0, frequency: int = 252) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - daily_rf
    if np.std(excess, ddof=1) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(frequency))


def sortino_ratio(returns: np.ndarray, daily_rf: float = 0.0, frequency: int = 252) -> float:
    """Annualized Sortino ratio."""
    excess = returns - daily_rf
    downside = np.minimum(excess, 0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(frequency))


def calmar_ratio(returns: np.ndarray, frequency: int = 252) -> float:
    """Calmar ratio: CAGR / |Max Drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return cagr(returns, frequency) / mdd


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains / losses above/below threshold."""
    excess = returns - threshold
    gains = np.sum(excess[excess > 0])
    losses = abs(np.sum(excess[excess <= 0]))
    if losses == 0:
        return 0.0
    return float(gains / losses)


def tail_ratio(returns: np.ndarray) -> float:
    """Tail ratio: 95th percentile / |5th percentile|."""
    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))
    if p5 == 0:
        return 0.0
    return float(p95 / p5)


def profit_factor(returns: np.ndarray) -> float:
    """Profit factor: gross profits / gross losses."""
    gains = np.sum(returns[returns > 0])
    losses = abs(np.sum(returns[returns < 0]))
    if losses == 0:
        return 0.0
    return float(gains / losses)


# ── Tracking / benchmark metrics ──────────────────────────────────────


def alpha_beta(
    returns: np.ndarray,
    benchmark: np.ndarray,
    daily_rf: float = 0.0,
    frequency: int = 252,
) -> tuple[float, float]:
    """Jensen's alpha and beta via OLS regression."""
    excess_r = returns - daily_rf
    excess_b = benchmark - daily_rf
    if np.std(excess_b) == 0:
        return 0.0, 0.0
    beta = np.cov(excess_r, excess_b)[0, 1] / np.var(excess_b, ddof=1)
    alpha = (np.mean(excess_r) - beta * np.mean(excess_b)) * frequency
    return float(alpha), float(beta)


def tracking_error(returns: np.ndarray, benchmark: np.ndarray, frequency: int = 252) -> float:
    """Annualized tracking error (std of active returns)."""
    active = returns - benchmark
    return float(np.std(active, ddof=1) * np.sqrt(frequency))


def information_ratio(returns: np.ndarray, benchmark: np.ndarray, frequency: int = 252) -> float:
    """Information ratio: active return / tracking error."""
    te = tracking_error(returns, benchmark, frequency)
    if te == 0:
        return 0.0
    active_return = (np.mean(returns) - np.mean(benchmark)) * frequency
    return float(active_return / te)


def treynor_ratio(
    returns: np.ndarray,
    benchmark: np.ndarray,
    daily_rf: float = 0.0,
    frequency: int = 252,
) -> float:
    """Treynor ratio: excess return / beta."""
    _, beta = alpha_beta(returns, benchmark, daily_rf, frequency)
    if beta == 0:
        return 0.0
    excess_return = (np.mean(returns) - daily_rf) * frequency
    return float(excess_return / beta)


def up_capture(returns: np.ndarray, benchmark: np.ndarray) -> float:
    """Up-capture ratio: portfolio return in up-markets / benchmark return in up-markets."""
    up = benchmark > 0
    if not np.any(up):
        return 0.0
    port_up = np.mean(returns[up])
    bench_up = np.mean(benchmark[up])
    if bench_up == 0:
        return 0.0
    return float(port_up / bench_up * 100)


def down_capture(returns: np.ndarray, benchmark: np.ndarray) -> float:
    """Down-capture ratio: portfolio return in down-markets / benchmark return in down-markets."""
    down = benchmark < 0
    if not np.any(down):
        return 0.0
    port_down = np.mean(returns[down])
    bench_down = np.mean(benchmark[down])
    if bench_down == 0:
        return 0.0
    return float(port_down / bench_down * 100)


# ── Drawdown series ───────────────────────────────────────────────────


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Compute the drawdown time series."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    return cumulative / running_max - 1
