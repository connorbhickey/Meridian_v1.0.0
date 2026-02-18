"""Tool schemas and implementations for the AI Copilot.

Pure computation — zero GUI imports.  Each tool wraps existing engine
functions and returns JSON-serialisable dicts that Claude can reason over.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from portopt.constants import OptMethod
from portopt.engine.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)

# ── Tool Schema Definitions (Anthropic tool-use format) ───────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_portfolio_metrics",
        "description": (
            "Compute 20+ performance metrics for the current portfolio, "
            "including Sharpe ratio, volatility, max drawdown, VaR, Sortino, "
            "CAGR, and more.  Returns a dictionary of metric names to values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_correlation_summary",
        "description": (
            "Return the top N most-correlated and least-correlated asset "
            "pairs in the portfolio.  Useful for diversification analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of pairs to return for each category (default 5).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_weight_summary",
        "description": (
            "Return the current portfolio weights sorted by allocation size, "
            "with total count and concentration metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_stress_test_summary",
        "description": (
            "Run historical stress scenarios (2008 GFC, COVID Crash, etc.) "
            "on the current portfolio and return expected impact for each."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_monte_carlo_summary",
        "description": (
            "Run a Monte Carlo simulation on the current portfolio and "
            "return percentile wealth projections and risk metrics.  "
            "Returns median, 5th, and 95th percentile outcomes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_sims": {
                    "type": "integer",
                    "description": "Number of simulations (default 1000).",
                },
                "horizon_days": {
                    "type": "integer",
                    "description": "Projection horizon in trading days (default 252).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_quick_optimization",
        "description": (
            "Run a portfolio optimization with the specified method.  "
            "Returns optimal weights, expected return, volatility, and Sharpe ratio.  "
            "Methods: MAX_SHARPE, MIN_VOLATILITY, MAX_DIVERSIFICATION, "
            "INVERSE_VARIANCE, MAX_DECORRELATION."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": (
                        "Optimization method.  One of: MAX_SHARPE, MIN_VOLATILITY, "
                        "MAX_DIVERSIFICATION, INVERSE_VARIANCE, MAX_DECORRELATION."
                    ),
                },
                "max_weight": {
                    "type": "number",
                    "description": "Maximum weight per asset (0-1, default 1.0).",
                },
            },
            "required": [],
        },
    },
]


# ── Tool Implementations ──────────────────────────────────────────────

def _safe_float(val: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python floats for JSON."""
    if isinstance(val, (np.floating, np.integer)):
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 6)
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 6)
    return val


def execute_tool(
    tool_name: str,
    tool_input: dict,
    context: dict,
) -> dict:
    """Dispatch a tool call and return JSON-serialisable result.

    Parameters
    ----------
    tool_name : name from TOOL_SCHEMAS
    tool_input : parsed input arguments from Claude
    context : {
        "prices": pd.DataFrame | None,
        "weights": dict[str, float] | None,
        "mu": pd.Series | None,
        "cov": pd.DataFrame | None,
        "result": OptimizationResult | None,
    }
    """
    dispatch = {
        "get_portfolio_metrics": _get_portfolio_metrics,
        "get_correlation_summary": _get_correlation_summary,
        "get_weight_summary": _get_weight_summary,
        "get_stress_test_summary": _get_stress_test_summary,
        "get_monte_carlo_summary": _get_monte_carlo_summary,
        "run_quick_optimization": _run_quick_optimization,
    }
    fn = dispatch.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(tool_input, context)
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
        return {"error": f"Tool execution failed: {e}"}


# ── Individual Tool Functions ─────────────────────────────────────────

def _get_portfolio_metrics(params: dict, ctx: dict) -> dict:
    """Compute portfolio metrics from current weights and prices."""
    from portopt.engine.metrics import compute_all_metrics

    prices = ctx.get("prices")
    weights = ctx.get("weights")
    if prices is None or weights is None:
        return {"error": "No portfolio data available. Run an optimization first."}

    # Build portfolio return series from weights and asset returns
    returns_df = prices.pct_change().dropna()
    symbols = [s for s in weights if s in returns_df.columns]
    if not symbols:
        return {"error": "No matching symbols found between weights and price data."}

    w = np.array([weights[s] for s in symbols])
    w = w / w.sum()
    portfolio_returns = returns_df[symbols].values @ w

    metrics = compute_all_metrics(pd.Series(portfolio_returns))
    return {k: _safe_float(v) for k, v in metrics.items()}


def _get_correlation_summary(params: dict, ctx: dict) -> dict:
    """Return top/bottom correlated pairs."""
    prices = ctx.get("prices")
    if prices is None:
        return {"error": "No price data available."}

    top_n = params.get("top_n", 5)
    returns_df = prices.pct_change().dropna()
    corr = returns_df.corr()

    # Extract upper triangle pairs
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    most = [{"pair": f"{a}-{b}", "correlation": round(c, 4)} for a, b, c in pairs[:top_n]]
    least = [{"pair": f"{a}-{b}", "correlation": round(c, 4)} for a, b, c in pairs[-top_n:]]

    return {
        "most_correlated": most,
        "least_correlated": least,
        "num_assets": len(cols),
    }


def _get_weight_summary(params: dict, ctx: dict) -> dict:
    """Return current portfolio weights summary."""
    weights = ctx.get("weights")
    if not weights:
        return {"error": "No portfolio weights available."}

    sorted_w = sorted(weights.items(), key=lambda x: -x[1])
    top_5 = [{"symbol": s, "weight": round(w, 4)} for s, w in sorted_w[:5]]
    bottom_5 = [{"symbol": s, "weight": round(w, 4)} for s, w in sorted_w[-5:] if w > 0]

    # Concentration metrics
    w_arr = np.array(list(weights.values()))
    hhi = float(np.sum(w_arr ** 2))
    effective_n = 1.0 / hhi if hhi > 0 else 0

    return {
        "total_assets": len(weights),
        "top_5_holdings": top_5,
        "bottom_5_holdings": bottom_5,
        "max_weight": round(float(w_arr.max()), 4),
        "min_weight": round(float(w_arr[w_arr > 0].min()), 4) if (w_arr > 0).any() else 0,
        "herfindahl_index": round(hhi, 4),
        "effective_num_assets": round(effective_n, 1),
    }


def _get_stress_test_summary(params: dict, ctx: dict) -> dict:
    """Run historical stress tests."""
    from portopt.engine.stress import run_all_stress_tests

    weights = ctx.get("weights")
    cov = ctx.get("cov")
    if not weights:
        return {"error": "No portfolio weights available."}

    results = run_all_stress_tests(weights, cov=cov)
    scenarios = []
    for r in results:
        scenarios.append({
            "scenario": r.scenario.name,
            "description": r.scenario.description,
            "portfolio_impact": round(r.portfolio_impact * 100, 2),
            "stressed_value": round(r.stressed_value, 0),
            "worst_asset": max(r.asset_impacts.items(), key=lambda x: abs(x[1]))[0],
        })

    return {
        "scenarios": scenarios,
        "worst_case": scenarios[0]["scenario"] if scenarios else None,
        "worst_case_impact_pct": scenarios[0]["portfolio_impact"] if scenarios else None,
    }


def _get_monte_carlo_summary(params: dict, ctx: dict) -> dict:
    """Run a Monte Carlo simulation."""
    from portopt.data.models import MonteCarloConfig
    from portopt.engine.monte_carlo import run_monte_carlo

    weights = ctx.get("weights")
    mu = ctx.get("mu")
    cov = ctx.get("cov")
    prices = ctx.get("prices")

    if weights is None or mu is None or cov is None:
        return {"error": "Need weights, expected returns, and covariance. Run optimization first."}

    n_sims = params.get("n_sims", 1000)
    horizon = params.get("horizon_days", 252)

    config = MonteCarloConfig(n_sims=n_sims, horizon_days=horizon)
    historical_returns = prices.pct_change().dropna() if prices is not None else None

    result = run_monte_carlo(
        weights, mu, cov, config,
        historical_returns=historical_returns,
    )

    return {
        "n_sims": result.n_sims,
        "horizon_days": horizon,
        "initial_value": 100_000,
        "median_final_value": round(float(result.percentile_curves[2, -1] * 100_000), 0),
        "p5_final_value": round(float(result.percentile_curves[0, -1] * 100_000), 0),
        "p95_final_value": round(float(result.percentile_curves[4, -1] * 100_000), 0),
        "shortfall_probability": round(float(result.shortfall_probability) * 100, 1),
        "metric_distributions": {
            k: {
                "mean": round(float(np.mean(v)), 4),
                "median": round(float(np.median(v)), 4),
                "std": round(float(np.std(v)), 4),
            }
            for k, v in result.metric_distributions.items()
        },
    }


def _run_quick_optimization(params: dict, ctx: dict) -> dict:
    """Run a quick optimization with specified method."""
    from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer

    mu = ctx.get("mu")
    cov = ctx.get("cov")
    if mu is None or cov is None:
        return {"error": "Need expected returns and covariance. Run optimization first."}

    method_name = params.get("method", "MAX_SHARPE").upper()
    method_map = {
        "MAX_SHARPE": OptMethod.MAX_SHARPE,
        "MIN_VOLATILITY": OptMethod.MIN_VOLATILITY,
        "MAX_DIVERSIFICATION": OptMethod.MAX_DIVERSIFICATION,
        "INVERSE_VARIANCE": OptMethod.INVERSE_VARIANCE,
        "MAX_DECORRELATION": OptMethod.MAX_DECORRELATION,
    }
    method = method_map.get(method_name)
    if method is None:
        return {"error": f"Unknown method: {method_name}. Use: {', '.join(method_map.keys())}"}

    max_weight = params.get("max_weight", 1.0)
    constraints = PortfolioConstraints(
        long_only=True,
        max_weight=max_weight,
    )

    optimizer = MeanVarianceOptimizer(mu, cov, constraints=constraints, method=method)
    result = optimizer.optimize()

    sorted_w = sorted(result.weights.items(), key=lambda x: -x[1])
    return {
        "method": method_name,
        "expected_return": round(result.expected_return, 4),
        "volatility": round(result.volatility, 4),
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "weights": {s: round(w, 4) for s, w in sorted_w if w > 0.001},
        "num_assets_with_allocation": sum(1 for _, w in sorted_w if w > 0.001),
    }
