"""Abstract base class for all portfolio optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from portopt.data.models import OptimizationResult
from portopt.engine.constraints import PortfolioConstraints


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

    def _portfolio_return(self, w: np.ndarray) -> float:
        return float(w @ self.expected_returns.values)

    def _portfolio_volatility(self, w: np.ndarray) -> float:
        return float(np.sqrt(w @ self.covariance.values @ w))
