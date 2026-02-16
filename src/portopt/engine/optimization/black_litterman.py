"""Black-Litterman model.

Implements:
- Market-implied equilibrium returns (reverse optimization)
- Absolute and relative views with confidence
- Posterior returns via the Master Formula
- Idzorek confidence-based Omega calibration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints
from portopt.engine.optimization.base import BaseOptimizer


@dataclass
class BLView:
    """A single Black-Litterman view.

    Absolute view: "AAPL will return 10%" -> assets=["AAPL"], weights=[1.0], view_return=0.10
    Relative view: "AAPL will outperform MSFT by 2%" ->
        assets=["AAPL", "MSFT"], weights=[1.0, -1.0], view_return=0.02
    """
    assets: list[str]
    weights: list[float]
    view_return: float
    confidence: float = 1.0  # 0-1, used with Idzorek method


class BlackLittermanModel:
    """Black-Litterman posterior return estimation.

    Args:
        covariance: Annualized covariance matrix (N x N).
        market_caps: Market capitalizations by symbol (for equilibrium weights).
        risk_aversion: Market risk aversion parameter (delta).
        tau: Uncertainty scaling factor (typically 0.025 - 0.05).
        risk_free_rate: Annual risk-free rate.
    """

    def __init__(
        self,
        covariance: pd.DataFrame,
        market_caps: pd.Series | None = None,
        market_weights: pd.Series | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.04,
    ):
        self.covariance = covariance
        self.symbols = list(covariance.index)
        self.n_assets = len(self.symbols)
        self.Sigma = covariance.values
        self.delta = risk_aversion
        self.tau = tau
        self.rf = risk_free_rate

        # Market weights
        if market_weights is not None:
            self.w_mkt = market_weights.values
        elif market_caps is not None:
            total = market_caps.sum()
            self.w_mkt = (market_caps / total).values
        else:
            self.w_mkt = np.ones(self.n_assets) / self.n_assets

        # Equilibrium returns: π = δΣw_mkt
        self.pi = self.delta * self.Sigma @ self.w_mkt

    def equilibrium_returns(self) -> pd.Series:
        """Return market-implied equilibrium returns."""
        return pd.Series(self.pi, index=self.symbols)

    def posterior(self, views: list[BLView]) -> tuple[pd.Series, pd.DataFrame]:
        """Compute posterior returns and covariance using the Master Formula.

        Returns:
            (posterior_returns, posterior_covariance)
        """
        if not views:
            return self.equilibrium_returns(), self.covariance

        P, Q, Omega = self._build_view_matrices(views)

        tau_Sigma = self.tau * self.Sigma

        # Master Formula:
        # μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
        tau_Sigma_inv = np.linalg.inv(tau_Sigma)
        Omega_inv = np.linalg.inv(Omega)

        M = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        posterior_mu = M @ (tau_Sigma_inv @ self.pi + P.T @ Omega_inv @ Q)

        # Posterior covariance: Σ_BL = Σ + [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
        posterior_Sigma = self.Sigma + M

        mu = pd.Series(posterior_mu, index=self.symbols)
        cov = pd.DataFrame(posterior_Sigma, index=self.symbols, columns=self.symbols)
        return mu, cov

    def _build_view_matrices(self, views: list[BLView]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (pick matrix), Q (view returns), Omega (view uncertainty).

        P: K x N matrix (K views, N assets)
        Q: K vector of view returns
        Omega: K x K diagonal uncertainty matrix
        """
        K = len(views)
        N = self.n_assets
        P = np.zeros((K, N))
        Q = np.zeros(K)

        for k, view in enumerate(views):
            Q[k] = view.view_return
            for asset, weight in zip(view.assets, view.weights):
                if asset in self.symbols:
                    idx = self.symbols.index(asset)
                    P[k, idx] = weight

        # Omega: proportional to P * tau * Sigma * P' (He-Litterman)
        # Scaled by confidence
        base_omega = np.diag(np.diag(P @ (self.tau * self.Sigma) @ P.T))
        for k, view in enumerate(views):
            if view.confidence < 1.0:
                # Idzorek: scale uncertainty inversely with confidence
                # confidence=1 -> omega=0 (certain), confidence=0 -> omega=inf
                conf = max(view.confidence, 0.01)
                base_omega[k, k] /= conf
            else:
                # Full confidence — keep He-Litterman default
                pass

        return P, Q, base_omega


class BlackLittermanOptimizer(BaseOptimizer):
    """Portfolio optimizer using Black-Litterman posterior returns + MVO."""

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        views: list[BLView],
        market_caps: pd.Series | None = None,
        market_weights: pd.Series | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.04,
        constraints: PortfolioConstraints | None = None,
    ):
        super().__init__(expected_returns, covariance, constraints)
        self.views = views
        self.bl = BlackLittermanModel(
            covariance=covariance,
            market_caps=market_caps,
            market_weights=market_weights,
            risk_aversion=risk_aversion,
            tau=tau,
            risk_free_rate=risk_free_rate,
        )

    def optimize(self) -> OptimizationResult:
        """Compute BL posterior, then run max Sharpe on posterior."""
        from portopt.engine.optimization.mean_variance import MeanVarianceOptimizer
        from portopt.constants import OptMethod

        posterior_mu, posterior_cov = self.bl.posterior(self.views)

        mvo = MeanVarianceOptimizer(
            expected_returns=posterior_mu,
            covariance=posterior_cov,
            constraints=self.constraints,
            method=OptMethod.MAX_SHARPE,
        )
        result = mvo.optimize()
        result.method = "Black-Litterman"
        result.metadata["views"] = [
            {"assets": v.assets, "weights": v.weights, "return": v.view_return, "confidence": v.confidence}
            for v in self.views
        ]
        result.metadata["equilibrium_returns"] = self.bl.equilibrium_returns().to_dict()
        result.metadata["posterior_returns"] = posterior_mu.to_dict()
        return result
