"""Abstract base class for all portfolio optimizers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Abstract base for portfolio optimization algorithms.

    Subclasses implement `optimize()` which returns an OptimizationResult.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: PortfolioConstraints | None = None,
    ):
        self.expected_returns = expected_returns
        self.covariance = covariance
        self.symbols = list(expected_returns.index)
        self.n_assets = len(self.symbols)
        self.constraints = constraints or PortfolioConstraints()

        # Validate alignment
        assert list(covariance.index) == self.symbols, "Returns and covariance must have matching symbols"

        # Guard: empty portfolio
        if self.n_assets == 0:
            raise ValueError("Cannot optimize with zero assets")

        # Guard: NaN in expected returns
        nan_mu = np.where(np.isnan(expected_returns.values))[0]
        if len(nan_mu) > 0:
            bad_syms = [self.symbols[i] for i in nan_mu]
            raise ValueError(f"NaN in expected returns for: {', '.join(bad_syms)}")

        # Guard: NaN in covariance matrix
        if np.any(np.isnan(covariance.values)):
            raise ValueError("NaN values found in covariance matrix")

        # Guard: ensure positive semi-definite covariance
        self._single_asset = (self.n_assets == 1)
        if not self._single_asset:
            from portopt.engine.risk import is_positive_definite, nearest_positive_definite
            if not is_positive_definite(covariance.values):
                logger.warning("Covariance matrix is not PSD — applying nearest PSD correction")
                psd_vals = nearest_positive_definite(covariance.values)
                self.covariance = pd.DataFrame(psd_vals, index=covariance.index, columns=covariance.columns)

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Run the optimization and return results."""
        ...

    def _build_result(self, weights: np.ndarray, method: str, **metadata) -> OptimizationResult:
        """Build an OptimizationResult from weight array."""
        w = np.array(weights).flatten()
        weight_dict = dict(zip(self.symbols, w.tolist()))

        mu = float(w @ self.expected_returns.values)
        vol = float(np.sqrt(w @ self.covariance.values @ w))
        sharpe = mu / vol if vol > 0 else 0.0

        return OptimizationResult(
            method=method,
            weights=weight_dict,
            expected_return=mu,
            volatility=vol,
            sharpe_ratio=sharpe,
            metadata=metadata,
        )

    def _single_asset_result(self, method: str) -> OptimizationResult:
        """Return 100% weight to the single asset."""
        return self._build_result(np.array([1.0]), method)

    def _portfolio_return(self, w: np.ndarray) -> float:
        return float(w @ self.expected_returns.values)

    def _portfolio_volatility(self, w: np.ndarray) -> float:
        return float(np.sqrt(w @ self.covariance.values @ w))
