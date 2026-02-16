"""Mean-reversion online portfolio strategies.

- PAMR: Passive Aggressive Mean Reversion (Li et al., 2012)
- CWMR: Confidence Weighted Mean Reversion (Li et al., 2013)
- OLMAR: On-Line Moving Average Reversion (Li & Hoi, 2015)
- RMR: Robust Median Reversion (Huang et al., 2016)
"""

from __future__ import annotations

import numpy as np

from portopt.engine.optimization.online.base import OnlineStrategy


class PAMR(OnlineStrategy):
    """Passive Aggressive Mean Reversion.

    Uses a passive-aggressive online learning approach:
    - If the portfolio loses money, aggressively update toward mean reversion.
    - The update magnitude is proportional to the loss.

    Variant: PAMR-2 with C sensitivity parameter.
    """

    def __init__(self, n_assets: int, epsilon: float = 0.5, C: float = 500.0):
        """
        Args:
            epsilon: Mean reversion threshold (loss sensitivity).
            C: Aggressiveness parameter (higher = more aggressive updates).
        """
        super().__init__(n_assets)
        self.epsilon = epsilon
        self.C = C

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        x_mean = np.mean(price_relative)
        portfolio_return = np.dot(current_weights, price_relative)

        # Loss
        loss = max(0, portfolio_return - self.epsilon)

        # Denominator: ||x_t - x_bar||^2
        x_dev = price_relative - x_mean
        denom = np.dot(x_dev, x_dev)

        if denom == 0:
            return current_weights

        # PAMR-2: τ = min(C, loss / ||x - x_bar||^2)
        tau = min(self.C, loss / denom)

        # Update: b_{t+1} = b_t - τ * (x_t - x_bar * 1)
        new_weights = current_weights - tau * x_dev

        return self._project_simplex(new_weights)


class CWMR(OnlineStrategy):
    """Confidence Weighted Mean Reversion.

    Maintains a Gaussian distribution over weight vectors.
    Updates both mean and variance of the distribution.
    """

    def __init__(self, n_assets: int, phi: float = 0.5, epsilon: float = 0.5):
        """
        Args:
            phi: Confidence parameter.
            epsilon: Mean reversion threshold.
        """
        super().__init__(n_assets)
        self.phi = phi
        self.epsilon = epsilon
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def initialize(self) -> np.ndarray:
        self._mu = np.ones(self.n_assets) / self.n_assets
        self._sigma = np.eye(self.n_assets) / self.n_assets
        return self._mu.copy()

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        x_mean = np.mean(price_relative)
        x_dev = price_relative - x_mean

        # Confidence margin
        M = np.dot(x_dev, self._mu)
        V = np.dot(x_dev, self._sigma @ x_dev)

        # Check if update needed
        if M - self.epsilon >= self.phi * np.sqrt(V):
            return current_weights

        # Compute lambda
        W = max(V + 2 * self.phi * np.sqrt(V) * (self.epsilon - M), 0)
        if V == 0:
            return current_weights

        lam = max(0, (-(2 * M - 2 * self.epsilon) + np.sqrt(
            max(0, (2 * M - 2 * self.epsilon) ** 2 - 4 * W)
        )) / (2 * V + 1e-10))

        # Update mean
        self._mu = self._mu - lam * self._sigma @ x_dev

        # Update covariance
        sx = self._sigma @ x_dev
        self._sigma = self._sigma - (lam / (1 + lam * V + 1e-10)) * np.outer(sx, sx)

        return self._project_simplex(self._mu.copy())


class OLMAR(OnlineStrategy):
    """On-Line Moving Average Reversion.

    Predicts next price relatives using moving average and
    shifts portfolio toward underperforming assets.

    Two variants:
    - OLMAR-1: Simple moving average
    - OLMAR-2: Exponential moving average
    """

    def __init__(self, n_assets: int, window: int = 5, epsilon: float = 10.0):
        """
        Args:
            window: Moving average lookback window.
            epsilon: Reversion aggressiveness parameter.
        """
        super().__init__(n_assets)
        self.window = window
        self.epsilon = epsilon
        self._price_history: list[np.ndarray] = []

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._price_history.append(price_relative.copy())

        if len(self._price_history) < self.window:
            return current_weights

        # Moving average prediction: x̂_{t+1} = MA / x_t
        recent = np.array(self._price_history[-self.window:])
        ma = recent.mean(axis=0)
        x_predict = ma / price_relative  # predicted price relative

        # Mean of predicted
        x_mean = np.mean(x_predict)

        # Deviation from mean
        x_dev = x_predict - x_mean
        denom = np.dot(x_dev, x_dev)

        # Portfolio return with predicted
        portfolio_predict = np.dot(current_weights, x_predict)

        if denom == 0:
            return current_weights

        # Lambda: max(0, (epsilon - b·x̂) / ||x̂ - x̂_bar||^2)
        lam = max(0, (self.epsilon - portfolio_predict) / denom)

        # Update
        new_weights = current_weights + lam * x_dev

        return self._project_simplex(new_weights)


class RMR(OnlineStrategy):
    """Robust Median Reversion.

    Similar to OLMAR but uses L1-median (geometric median) instead
    of arithmetic mean for robustness to outliers.
    """

    def __init__(self, n_assets: int, window: int = 5, epsilon: float = 5.0):
        super().__init__(n_assets)
        self.window = window
        self.epsilon = epsilon
        self._price_history: list[np.ndarray] = []

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._price_history.append(price_relative.copy())

        if len(self._price_history) < self.window:
            return current_weights

        recent = np.array(self._price_history[-self.window:])

        # L1-median (Weiszfeld's algorithm)
        median = self._l1_median(recent)
        x_predict = median / price_relative

        x_mean = np.mean(x_predict)
        x_dev = x_predict - x_mean
        denom = np.dot(x_dev, x_dev)

        portfolio_predict = np.dot(current_weights, x_predict)

        if denom == 0:
            return current_weights

        lam = max(0, (self.epsilon - portfolio_predict) / denom)
        new_weights = current_weights + lam * x_dev

        return self._project_simplex(new_weights)

    @staticmethod
    def _l1_median(data: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """Compute the geometric (L1) median using Weiszfeld's algorithm."""
        y = np.mean(data, axis=0).copy()
        for _ in range(max_iter):
            dists = np.linalg.norm(data - y, axis=1)
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / dists
            y_new = np.average(data, axis=0, weights=weights)
            if np.linalg.norm(y_new - y) < 1e-8:
                break
            y = y_new
        return y
