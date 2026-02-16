"""Benchmark online portfolio strategies.

- BAH: Buy and Hold (equal-weighted)
- BestStock: Best individual stock in hindsight (oracle)
- CRP: Constant Rebalanced Portfolio
- BCRP: Best Constant Rebalanced Portfolio (oracle)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import product

from portopt.engine.optimization.online.base import OnlineStrategy, OnlineResult


class BAH(OnlineStrategy):
    """Buy and Hold — equal-weight initial, then let weights drift."""

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        # Let weights drift with price changes
        new_weights = current_weights * price_relative
        return new_weights / new_weights.sum()


class BestStock(OnlineStrategy):
    """Best Stock in hindsight (oracle benchmark).

    Can only be computed after seeing all data.
    """

    def __init__(self, n_assets: int, price_relatives: pd.DataFrame | None = None):
        super().__init__(n_assets)
        self._best_idx = 0
        if price_relatives is not None:
            cumulative = price_relatives.values.prod(axis=0)
            self._best_idx = int(np.argmax(cumulative))

    def initialize(self) -> np.ndarray:
        w = np.zeros(self.n_assets)
        w[self._best_idx] = 1.0
        return w

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        return current_weights

    @classmethod
    def compute(cls, price_relatives: pd.DataFrame) -> OnlineResult:
        """Compute best stock result directly."""
        X = price_relatives.values
        cumulative = X.prod(axis=0)
        best_idx = int(np.argmax(cumulative))
        wealth = float(cumulative[best_idx])

        w = np.zeros(X.shape[1])
        w[best_idx] = 1.0

        return OnlineResult(
            method="BestStock",
            weights_history=[w.copy()] * (X.shape[0] + 1),
            portfolio_values=[1.0] + [float(X[:t+1, best_idx].prod()) for t in range(X.shape[0])],
            final_wealth=wealth,
            symbols=list(price_relatives.columns),
        )


class CRP(OnlineStrategy):
    """Constant Rebalanced Portfolio — rebalance to fixed weights each period.

    Default: equal weights (uniform CRP).
    """

    def __init__(self, n_assets: int, target_weights: np.ndarray | None = None):
        super().__init__(n_assets)
        self._target = target_weights if target_weights is not None else np.ones(n_assets) / n_assets

    def initialize(self) -> np.ndarray:
        return self._target.copy()

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        return self._target.copy()


class BCRP(OnlineStrategy):
    """Best Constant Rebalanced Portfolio (oracle).

    Finds the CRP weights that maximize terminal wealth in hindsight.
    Uses grid search for small N, gradient for larger.
    """

    def __init__(self, n_assets: int):
        super().__init__(n_assets)
        self._best_weights: np.ndarray | None = None

    def initialize(self) -> np.ndarray:
        if self._best_weights is not None:
            return self._best_weights.copy()
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        if self._best_weights is not None:
            return self._best_weights.copy()
        return current_weights

    @classmethod
    def compute(cls, price_relatives: pd.DataFrame, n_grid: int = 10) -> OnlineResult:
        """Find BCRP via grid search."""
        X = price_relatives.values
        T, N = X.shape

        best_wealth = 0.0
        best_w = np.ones(N) / N

        if N <= 5:
            # Grid search for small N
            steps = np.linspace(0, 1, n_grid)
            for combo in product(steps, repeat=N):
                w = np.array(combo)
                if abs(w.sum() - 1.0) > 0.01:
                    continue
                w = w / w.sum()
                wealth = float(np.prod(X @ w))
                if wealth > best_wealth:
                    best_wealth = wealth
                    best_w = w.copy()
        else:
            # Gradient ascent for larger N
            w = np.ones(N) / N
            lr = 0.01
            for _ in range(500):
                log_returns = np.log(X @ w + 1e-10)
                grad = np.zeros(N)
                for t in range(T):
                    denom = X[t] @ w
                    if denom > 0:
                        grad += X[t] / denom
                grad /= T
                w = w * np.exp(lr * grad)
                w = w / w.sum()

            best_w = w
            best_wealth = float(np.prod(X @ w))

        return OnlineResult(
            method="BCRP",
            weights_history=[best_w.copy()] * (T + 1),
            portfolio_values=[1.0] + [float(np.prod(X[:t+1] @ best_w)) for t in range(T)],
            final_wealth=best_wealth,
            symbols=list(price_relatives.columns),
        )
