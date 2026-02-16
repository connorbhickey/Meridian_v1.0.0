"""Mean-Variance Optimization — 8 solutions + custom objective + frontier sweep.

Solutions implemented:
1. Inverse Variance (analytical)
2. Minimum Volatility
3. Maximum Sharpe Ratio
4. Efficient Risk (minimize vol for target return)
5. Efficient Return (maximize return for target risk)
6. Maximum Quadratic Utility
7. Maximum Diversification Ratio
8. Maximum Decorrelation

All convex solutions use cvxpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp

from portopt.constants import OptMethod
from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.optimization.base import BaseOptimizer


class MeanVarianceOptimizer(BaseOptimizer):
    """Mean-Variance Optimization with multiple solution methods."""

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: PortfolioConstraints | None = None,
        method: OptMethod = OptMethod.MAX_SHARPE,
    ):
        super().__init__(expected_returns, covariance, constraints)
        self.method = method

    def optimize(self) -> OptimizationResult:
        """Run the selected MVO method."""
        if self._single_asset:
            return self._single_asset_result(self.method.name.lower())

        dispatch = {
            OptMethod.INVERSE_VARIANCE: self._inverse_variance,
            OptMethod.MIN_VOLATILITY: self._min_volatility,
            OptMethod.MAX_SHARPE: self._max_sharpe,
            OptMethod.EFFICIENT_RISK: self._efficient_risk,
            OptMethod.EFFICIENT_RETURN: self._efficient_return,
            OptMethod.MAX_QUADRATIC_UTILITY: self._max_quadratic_utility,
            OptMethod.MAX_DIVERSIFICATION: self._max_diversification,
            OptMethod.MAX_DECORRELATION: self._max_decorrelation,
        }
        fn = dispatch.get(self.method)
        if fn is None:
            raise ValueError(f"Unsupported MVO method: {self.method}")
        return fn()

    def efficient_frontier(self, n_points: int = 50) -> list[OptimizationResult]:
        """Compute the efficient frontier by sweeping target returns.

        Returns a list of OptimizationResult along the frontier from
        minimum volatility to maximum return.
        """
        # Find return range
        min_vol_result = self._min_volatility()
        mu_min = min_vol_result.expected_return

        # Max return = put everything in best asset (subject to constraints)
        mu_max = float(self.expected_returns.max())

        targets = np.linspace(mu_min, mu_max, n_points)
        frontier = []
        for target in targets:
            try:
                w = self._solve_efficient_risk(target)
                if w is not None:
                    frontier.append(self._build_result(w, f"Frontier(target={target:.4f})"))
            except Exception:
                continue
        return frontier

    # ── 1. Inverse Variance (analytical, no optimization) ─────────────

    def _inverse_variance(self) -> OptimizationResult:
        """Weights proportional to 1/variance."""
        variances = np.diag(self.covariance.values)
        inv_var = 1.0 / variances
        w = inv_var / inv_var.sum()
        return self._build_result(w, "Inverse Variance")

    # ── 2. Minimum Volatility ─────────────────────────────────────────

    def _min_volatility(self) -> OptimizationResult:
        """Minimize portfolio variance: min w'Σw."""
        w = cp.Variable(self.n_assets)
        Sigma = self.covariance.values

        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = self._cvxpy_constraints(w)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Min Volatility optimization failed: {prob.status}")

        return self._build_result(w.value, "Min Volatility")

    # ── 3. Maximum Sharpe Ratio ───────────────────────────────────────

    def _max_sharpe(self, risk_free_rate: float = 0.04) -> OptimizationResult:
        """Maximize Sharpe ratio using the Cornuejols-Tutuncu transformation.

        Transform: let y = w/k where k = (μ-rf)'w, then minimize y'Σy
        subject to (μ-rf)'y = 1.
        """
        mu = self.expected_returns.values
        Sigma = self.covariance.values
        excess = mu - risk_free_rate

        # If all excess returns are negative, fall back to min vol
        if np.all(excess <= 0):
            return self._min_volatility()

        y = cp.Variable(self.n_assets)
        k = cp.Variable(nonneg=True)

        objective = cp.Minimize(cp.quad_form(y, Sigma))
        constraints = [
            excess @ y == 1,
            cp.sum(y) == k,
        ]
        # Bounds
        for i, sym in enumerate(self.symbols):
            lo, hi = self.constraints.get_bounds(sym)
            constraints.append(y[i] >= lo * k)
            constraints.append(y[i] <= hi * k)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Max Sharpe optimization failed: {prob.status}")

        w = y.value / k.value if k.value > 1e-10 else y.value
        return self._build_result(w, "Max Sharpe")

    # ── 4. Efficient Risk ─────────────────────────────────────────────

    def _efficient_risk(self) -> OptimizationResult:
        """Minimize volatility for a target return."""
        target = self.constraints.target_return
        if target is None:
            raise ValueError("target_return required for Efficient Risk")
        w = self._solve_efficient_risk(target)
        if w is None:
            raise RuntimeError("Efficient Risk optimization failed")
        return self._build_result(w, f"Efficient Risk (target={target:.4f})")

    def _solve_efficient_risk(self, target_return: float) -> np.ndarray | None:
        """Solve: min w'Σw s.t. μ'w >= target."""
        w = cp.Variable(self.n_assets)
        Sigma = self.covariance.values
        mu = self.expected_returns.values

        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = self._cvxpy_constraints(w)
        constraints.append(mu @ w >= target_return)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status in ("optimal", "optimal_inaccurate"):
            return w.value
        return None

    # ── 5. Efficient Return ───────────────────────────────────────────

    def _efficient_return(self) -> OptimizationResult:
        """Maximize return for a target risk (volatility)."""
        target = self.constraints.target_risk
        if target is None:
            raise ValueError("target_risk required for Efficient Return")

        w = cp.Variable(self.n_assets)
        Sigma = self.covariance.values
        mu = self.expected_returns.values

        objective = cp.Maximize(mu @ w)
        constraints = self._cvxpy_constraints(w)
        constraints.append(cp.quad_form(w, Sigma) <= target ** 2)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Efficient Return optimization failed: {prob.status}")

        return self._build_result(w.value, f"Efficient Return (target_vol={target:.4f})")

    # ── 6. Maximum Quadratic Utility ──────────────────────────────────

    def _max_quadratic_utility(self) -> OptimizationResult:
        """Maximize: μ'w - (λ/2) * w'Σw."""
        lam = self.constraints.risk_aversion

        w = cp.Variable(self.n_assets)
        Sigma = self.covariance.values
        mu = self.expected_returns.values

        objective = cp.Maximize(mu @ w - (lam / 2) * cp.quad_form(w, Sigma))
        constraints = self._cvxpy_constraints(w)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Max Quadratic Utility optimization failed: {prob.status}")

        return self._build_result(w.value, f"Max Quadratic Utility (lambda={lam})")

    # ── 7. Maximum Diversification Ratio ──────────────────────────────

    def _max_diversification(self) -> OptimizationResult:
        """Maximize the diversification ratio: w'σ / sqrt(w'Σw).

        Uses the same Cornuejols-Tutuncu trick as max Sharpe.
        """
        Sigma = self.covariance.values
        sigma = np.sqrt(np.diag(Sigma))  # individual volatilities

        y = cp.Variable(self.n_assets)
        k = cp.Variable(nonneg=True)

        objective = cp.Minimize(cp.quad_form(y, Sigma))
        constraints = [
            sigma @ y == 1,
            cp.sum(y) == k,
        ]
        for i, sym in enumerate(self.symbols):
            lo, hi = self.constraints.get_bounds(sym)
            constraints.append(y[i] >= lo * k)
            constraints.append(y[i] <= hi * k)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Max Diversification optimization failed: {prob.status}")

        w = y.value / k.value if k.value > 1e-10 else y.value
        return self._build_result(w, "Max Diversification")

    # ── 8. Maximum Decorrelation ──────────────────────────────────────

    def _max_decorrelation(self) -> OptimizationResult:
        """Minimize portfolio correlation: min w'Cw where C is correlation matrix.

        Equivalent to minimum volatility on the correlation matrix.
        """
        Sigma = self.covariance.values
        sigma = np.sqrt(np.diag(Sigma))
        Corr = Sigma / np.outer(sigma, sigma)
        np.fill_diagonal(Corr, 1.0)

        w = cp.Variable(self.n_assets)

        objective = cp.Minimize(cp.quad_form(w, Corr))
        constraints = self._cvxpy_constraints(w)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Max Decorrelation optimization failed: {prob.status}")

        return self._build_result(w.value, "Max Decorrelation")

    # ── cvxpy constraint builder ──────────────────────────────────────

    def _cvxpy_constraints(self, w: cp.Variable) -> list:
        """Build standard cvxpy constraints from PortfolioConstraints."""
        constraints = []
        c = self.constraints

        # Sum constraint
        if c.market_neutral:
            constraints.append(cp.sum(w) == 0)
        else:
            constraints.append(cp.sum(w) == c.leverage)

        # Per-asset bounds
        for i, sym in enumerate(self.symbols):
            lo, hi = c.get_bounds(sym)
            constraints.append(w[i] >= lo)
            constraints.append(w[i] <= hi)

        # Turnover constraint
        if c.max_turnover is not None and c.current_weights:
            current = np.array([c.current_weights.get(s, 0.0) for s in self.symbols])
            constraints.append(cp.norm(w - current, 1) <= c.max_turnover)

        # Sector constraints
        if c.sector_constraints and c.sector_map:
            sectors: dict[str, list[int]] = {}
            for i, sym in enumerate(self.symbols):
                sector = c.sector_map.get(sym, "Unknown")
                sectors.setdefault(sector, []).append(i)
            for sector, (s_min, s_max) in c.sector_constraints.items():
                if sector in sectors:
                    idx = sectors[sector]
                    constraints.append(cp.sum(w[idx]) >= s_min)
                    constraints.append(cp.sum(w[idx]) <= s_max)

        return constraints
