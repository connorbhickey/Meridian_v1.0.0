"""Base class for Online Portfolio Selection strategies.

Online strategies process price relatives sequentially and update
portfolio weights one period at a time. They don't require a
covariance matrix or expected returns upfront.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OnlineResult:
    """Result of running an online portfolio strategy."""
    method: str
    weights_history: list[np.ndarray] = field(default_factory=list)
    portfolio_values: list[float] = field(default_factory=list)
    final_wealth: float = 1.0
    symbols: list[str] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return self.final_wealth - 1.0

    @property
    def final_weights(self) -> dict[str, float]:
        if self.weights_history and self.symbols:
            return dict(zip(self.symbols, self.weights_history[-1].tolist()))
        return {}


class OnlineStrategy(ABC):
    """Abstract base for online portfolio selection strategies.

    Strategies process price relatives x_t = p_t / p_{t-1} sequentially.
    At each step, they output a weight vector b_t for the next period.
    """

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    @abstractmethod
    def initialize(self) -> np.ndarray:
        """Return the initial portfolio weights (before any data)."""
        ...

    @abstractmethod
    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        """Update weights based on new price relatives.

        Args:
            price_relative: x_t = p_t / p_{t-1} for each asset.
            current_weights: Current portfolio weights b_t.

        Returns:
            New portfolio weights b_{t+1}.
        """
        ...

    def run(self, price_relatives: pd.DataFrame) -> OnlineResult:
        """Run the strategy on a full matrix of price relatives.

        Args:
            price_relatives: DataFrame of price relatives (T x N), each row = x_t.

        Returns:
            OnlineResult with weight history and portfolio values.
        """
        symbols = list(price_relatives.columns)
        X = price_relatives.values
        T, N = X.shape

        weights = self.initialize()
        wealth = 1.0
        weights_history = [weights.copy()]
        portfolio_values = [wealth]

        for t in range(T):
            x_t = X[t]
            # Portfolio return for this period
            period_return = np.dot(weights, x_t)
            wealth *= period_return

            # Update weights for next period
            weights = self.update(x_t, weights)
            weights = self._project_simplex(weights)

            weights_history.append(weights.copy())
            portfolio_values.append(wealth)

        return OnlineResult(
            method=self.__class__.__name__,
            weights_history=weights_history,
            portfolio_values=portfolio_values,
            final_wealth=wealth,
            symbols=symbols,
        )

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """Project a vector onto the probability simplex (weights >= 0, sum = 1).

        Uses the algorithm from Duchi et al. (2008).
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0)


def price_relatives_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price DataFrame to price relatives: x_t = p_t / p_{t-1}."""
    return (prices / prices.shift(1)).dropna()
