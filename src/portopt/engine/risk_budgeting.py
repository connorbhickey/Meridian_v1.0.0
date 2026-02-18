"""Risk budgeting optimization — Euler decomposition, risk parity, and custom risk budgets.

Implements three core capabilities:
1. Risk contribution analysis via Euler decomposition
2. Custom risk budget optimization (target risk allocations per asset)
3. Equal Risk Contribution (ERC / risk parity) as a special case

All optimizations use scipy.optimize.minimize with SLSQP, which handles the
non-convex risk budgeting objective more reliably than convex solvers.

This is a pure computation module — zero GUI imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portopt.data.models import OptimizationResult

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class RiskBudgetConfig:
    """Configuration for risk budget optimization.

    Attributes:
        risk_budgets: Mapping of symbol to target risk contribution fraction.
                      Values must sum to 1.0.
        risk_measure: Risk measure to use — "volatility" or "cvar".
    """
    risk_budgets: dict[str, float]
    risk_measure: str = "volatility"

    def __post_init__(self):
        total = sum(self.risk_budgets.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Risk budgets must sum to 1.0, got {total:.6f}"
            )
        if any(v < 0 for v in self.risk_budgets.values()):
            raise ValueError("Risk budgets must be non-negative")
        if self.risk_measure not in ("volatility", "cvar"):
            raise ValueError(
                f"Unsupported risk_measure: {self.risk_measure!r}. "
                "Use 'volatility' or 'cvar'."
            )


@dataclass
class RiskContribution:
    """Risk contribution of a single asset in a portfolio.

    Attributes:
        symbol: Asset ticker symbol.
        weight: Portfolio weight of the asset.
        marginal_risk: Marginal risk contribution (d sigma_p / d w_i).
        risk_contribution: Absolute risk contribution (w_i * MRC_i).
        risk_contribution_pct: Risk contribution as a fraction of total
                               portfolio risk (RC_i / sigma_p).
    """
    symbol: str
    weight: float
    marginal_risk: float
    risk_contribution: float
    risk_contribution_pct: float


# ── Risk Contribution Analysis ───────────────────────────────────────


def compute_risk_contributions(
    weights: dict[str, float],
    cov: pd.DataFrame,
) -> list[RiskContribution]:
    """Compute the Euler decomposition of portfolio risk.

    Uses the volatility risk measure: sigma_p = sqrt(w' Sigma w).
    The Euler decomposition guarantees that the sum of individual risk
    contributions equals the total portfolio risk.

    Euler decomposition:
        MRC_i  = (Sigma @ w)_i / sigma_p
        RC_i   = w_i * MRC_i
        RC%_i  = RC_i / sigma_p

    Args:
        weights: Mapping of symbol to portfolio weight.
        cov: Covariance matrix DataFrame (symbols x symbols).

    Returns:
        List of RiskContribution sorted by symbol.
    """
    # Align symbols from the covariance matrix
    symbols = list(cov.columns)
    n = len(symbols)

    # Build weight vector aligned to covariance symbols
    w = np.array([weights.get(s, 0.0) for s in symbols])
    Sigma = cov.values

    # Portfolio volatility
    port_var = float(w @ Sigma @ w)
    if port_var <= 0:
        # Degenerate case: zero or negative variance
        return [
            RiskContribution(
                symbol=s,
                weight=float(w[i]),
                marginal_risk=0.0,
                risk_contribution=0.0,
                risk_contribution_pct=0.0,
            )
            for i, s in enumerate(symbols)
        ]

    sigma_p = np.sqrt(port_var)

    # Sigma @ w vector
    sigma_w = Sigma @ w

    results = []
    for i, s in enumerate(symbols):
        mrc_i = float(sigma_w[i] / sigma_p)
        rc_i = float(w[i] * mrc_i)
        rc_pct = float(rc_i / sigma_p) if sigma_p > 0 else 0.0

        results.append(RiskContribution(
            symbol=s,
            weight=float(w[i]),
            marginal_risk=mrc_i,
            risk_contribution=rc_i,
            risk_contribution_pct=rc_pct,
        ))

    results.sort(key=lambda rc: rc.symbol)
    return results


# ── Risk Budget Optimization ─────────────────────────────────────────


def optimize_risk_budget(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_budgets: dict[str, float],
    constraints: object = None,
) -> OptimizationResult:
    """Optimize portfolio weights to match target risk budget allocations.

    Uses the log-barrier formulation which is equivalent to risk budgeting
    but yields a convex-like optimization landscape:

        minimize  w' Sigma w / 2  -  sum_i  b_i * log(w_i)

    subject to  sum(w_i) = 1,  w_i >= epsilon.

    At the optimum, the risk contributions satisfy RC_i / sigma_p = b_i.

    Args:
        mu: Expected returns Series indexed by symbol.
        cov: Covariance matrix DataFrame (symbols x symbols).
        risk_budgets: Target risk contribution fractions {symbol: b_i},
                      must sum to 1.0.
        constraints: Reserved for future use (PortfolioConstraints).

    Returns:
        OptimizationResult with method="Risk Budget".

    Raises:
        ValueError: If risk budgets are invalid or symbols do not match.
    """
    symbols = list(mu.index)
    n = len(symbols)

    # Validate risk budgets
    budget_total = sum(risk_budgets.values())
    if abs(budget_total - 1.0) > 1e-6:
        raise ValueError(
            f"Risk budgets must sum to 1.0, got {budget_total:.6f}"
        )

    # Build budget vector aligned to symbols
    b = np.array([risk_budgets.get(s, 0.0) for s in symbols])
    if np.any(b < 0):
        raise ValueError("Risk budgets must be non-negative")
    if abs(b.sum() - 1.0) > 1e-6:
        raise ValueError(
            f"Risk budgets for available symbols must sum to 1.0, got {b.sum():.6f}"
        )

    # Single asset edge case
    if n == 1:
        w_dict = {symbols[0]: 1.0}
        exp_ret = float(mu.values[0])
        vol = float(np.sqrt(cov.values[0, 0]))
        sharpe = exp_ret / vol if vol > 0 else 0.0
        return OptimizationResult(
            method="Risk Budget",
            weights=w_dict,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata={"risk_budgets": risk_budgets},
        )

    Sigma = cov.values
    mu_arr = mu.values

    # Direct squared-error objective on risk contribution fractions:
    #   minimize  sum_i (RC_i / sigma_p^2 - b_i)^2
    # where RC_i = w_i * (Sigma w)_i  and  sigma_p^2 = w' Sigma w.
    # Dividing by sigma_p^2 (variance) rather than sigma_p (volatility)
    # is standard for the Spinu (2013) formulation and yields cleaner
    # gradients. The log-barrier formulation is used as a second stage
    # refinement if the direct objective does not converge well.
    def objective(w):
        sigma_w = Sigma @ w
        port_var = w @ sigma_w
        if port_var < 1e-30:
            return 1e10
        rc_frac = (w * sigma_w) / port_var  # RC_i / sigma_p^2
        return float(np.sum((rc_frac - b) ** 2))

    # Log-barrier objective (fallback / refinement):
    #   minimize  w'Sigma w / 2  -  sum_i  b_i * log(w_i)
    # At the optimum of this function, RC_i / sigma_p^2 = b_i exactly.
    def objective_log_barrier(w):
        port_var = w @ Sigma @ w
        w_safe = np.maximum(w, 1e-16)
        log_term = b @ np.log(w_safe)
        return 0.5 * port_var - log_term

    def gradient_log_barrier(w):
        grad_var = Sigma @ w
        w_safe = np.maximum(w, 1e-16)
        grad_log = b / w_safe
        return grad_var - grad_log

    # Bounds: small positive lower bound
    eps = 1e-10
    bounds = [(eps, 1.0)] * n

    # Constraint: weights sum to 1
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Multi-start: try several initial points and keep the best
    starts = []
    # Start 1: proportional to risk budgets
    w_b = np.maximum(b.copy(), 1e-6)
    w_b /= w_b.sum()
    starts.append(w_b)
    # Start 2: equal weight
    starts.append(np.ones(n) / n)
    # Start 3: inverse-volatility
    diag_var = np.maximum(np.diag(Sigma), 1e-16)
    inv_vol = 1.0 / np.sqrt(diag_var)
    starts.append(inv_vol / inv_vol.sum())

    best_result = None
    best_obj = np.inf

    for w0 in starts:
        # Stage 1: log-barrier (smooth, reliable convergence)
        res = minimize(
            objective_log_barrier,
            w0,
            jac=gradient_log_barrier,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_constraint],
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        # Stage 2: refine with direct objective
        res2 = minimize(
            objective,
            res.x,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_constraint],
            options={"maxiter": 1000, "ftol": 1e-15},
        )
        obj_val = objective(res2.x)
        if obj_val < best_obj:
            best_obj = obj_val
            best_result = res2

    result = best_result

    if not result.success:
        logger.warning(
            "Risk budget optimization did not converge: %s. "
            "Using best iterate.",
            result.message,
        )

    w_opt = result.x
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum()

    # Build result
    w_dict = dict(zip(symbols, w_opt.tolist()))
    exp_ret = float(w_opt @ mu_arr)
    vol = float(np.sqrt(w_opt @ Sigma @ w_opt))
    sharpe = exp_ret / vol if vol > 0 else 0.0

    # Compute actual risk contributions for metadata
    rc_list = compute_risk_contributions(w_dict, cov)
    rc_actual = {rc.symbol: rc.risk_contribution_pct for rc in rc_list}

    return OptimizationResult(
        method="Risk Budget",
        weights=w_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        metadata={
            "risk_budgets": risk_budgets,
            "actual_risk_contributions": rc_actual,
            "optimizer_message": result.message,
            "optimizer_converged": result.success,
        },
    )


# ── Equal Risk Contribution (Risk Parity) ────────────────────────────


def equal_risk_contribution(
    mu: pd.Series,
    cov: pd.DataFrame,
    constraints: object = None,
) -> OptimizationResult:
    """Compute Equal Risk Contribution (ERC / risk parity) portfolio.

    All assets contribute equally to total portfolio risk:
        b_i = 1/N  for all i.

    Uses a pairwise squared-difference objective which directly targets
    equal risk contributions:

        minimize  sum_{i,j} (w_i * (Sigma w)_i  -  w_j * (Sigma w)_j)^2

    subject to  sum(w_i) = 1,  w_i >= 0.

    Args:
        mu: Expected returns Series indexed by symbol.
        cov: Covariance matrix DataFrame (symbols x symbols).
        constraints: Reserved for future use (PortfolioConstraints).

    Returns:
        OptimizationResult with method="Equal Risk Contribution".
    """
    symbols = list(mu.index)
    n = len(symbols)

    # Single asset edge case
    if n == 1:
        w_dict = {symbols[0]: 1.0}
        exp_ret = float(mu.values[0])
        vol = float(np.sqrt(cov.values[0, 0]))
        sharpe = exp_ret / vol if vol > 0 else 0.0
        return OptimizationResult(
            method="Equal Risk Contribution",
            weights=w_dict,
            expected_return=exp_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata={"risk_budgets": {symbols[0]: 1.0}},
        )

    Sigma = cov.values
    mu_arr = mu.values

    # Objective: minimize sum of squared pairwise differences of risk
    # contributions.  RC_i = w_i * (Sigma @ w)_i
    def objective(w):
        sigma_w = Sigma @ w
        rc = w * sigma_w  # element-wise: RC_i = w_i * (Sigma w)_i
        # Sum of all pairwise squared differences
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += (rc[i] - rc[j]) ** 2
        return total

    def gradient(w):
        """Analytical gradient of the ERC objective."""
        sigma_w = Sigma @ w
        rc = w * sigma_w

        grad = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = rc[i] - rc[j]
                # d(RC_i)/d(w_k)
                # RC_i = w_i * (Sigma w)_i
                # d(RC_i)/d(w_i) = (Sigma w)_i + w_i * Sigma_{i,i}
                # d(RC_i)/d(w_k) = w_i * Sigma_{i,k}  for k != i
                #
                # d(RC_j)/d(w_k) similarly

                # Gradient contribution for each w_k
                for k in range(n):
                    if k == i:
                        dRC_i = sigma_w[i] + w[i] * Sigma[i, i]
                    else:
                        dRC_i = w[i] * Sigma[i, k]

                    if k == j:
                        dRC_j = sigma_w[j] + w[j] * Sigma[j, j]
                    else:
                        dRC_j = w[j] * Sigma[j, k]

                    grad[k] += 2 * diff * (dRC_i - dRC_j)

        # We're double-counting pairs (i,j) and (j,i) in the loops above
        # since the outer objective only sums i < j. But expanding both
        # directions and halving is equivalent. The factor cancels because
        # each pair (i,j) with i<j appears once in objective but twice in
        # the gradient loop. Divide by 2 to correct.
        return grad / 2.0

    # Initial guess: inverse-volatility heuristic (good starting point for ERC)
    diag_var = np.diag(Sigma)
    diag_var = np.maximum(diag_var, 1e-16)
    inv_vol = 1.0 / np.sqrt(diag_var)
    w0 = inv_vol / inv_vol.sum()

    # Bounds: allow very small but positive weights
    eps = 1e-10
    bounds = [(eps, 1.0)] * n

    # Constraint: weights sum to 1
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"maxiter": 1000, "ftol": 1e-15},
    )

    if not result.success:
        # Try again with log-barrier formulation as fallback
        logger.warning(
            "ERC pairwise objective did not converge: %s. "
            "Falling back to log-barrier formulation.",
            result.message,
        )
        budgets = {s: 1.0 / n for s in symbols}
        return optimize_risk_budget(mu, cov, budgets, constraints)

    w_opt = result.x
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum()

    # Build result
    w_dict = dict(zip(symbols, w_opt.tolist()))
    exp_ret = float(w_opt @ mu_arr)
    vol = float(np.sqrt(w_opt @ Sigma @ w_opt))
    sharpe = exp_ret / vol if vol > 0 else 0.0

    # Compute actual risk contributions for metadata
    rc_list = compute_risk_contributions(w_dict, cov)
    rc_actual = {rc.symbol: rc.risk_contribution_pct for rc in rc_list}

    return OptimizationResult(
        method="Equal Risk Contribution",
        weights=w_dict,
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        metadata={
            "risk_budgets": {s: 1.0 / n for s in symbols},
            "actual_risk_contributions": rc_actual,
            "optimizer_message": result.message,
            "optimizer_converged": result.success,
        },
    )
