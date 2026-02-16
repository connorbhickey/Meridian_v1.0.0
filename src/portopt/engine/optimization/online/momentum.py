"""Momentum-based online portfolio strategies.

- EG: Exponential Gradient (Helmbold et al., 1998)
- FTL: Follow the Leader
- FTRL: Follow the Regularized Leader
"""

from __future__ import annotations

import numpy as np

from portopt.engine.optimization.online.base import OnlineStrategy


class EG(OnlineStrategy):
    """Exponential Gradient — multiplicative update toward recent winners.

    b_{t+1,i} = b_{t,i} * exp(η * x_{t,i} / (b_t · x_t)) / Z

    where Z is a normalization constant and η is the learning rate.
    """

    def __init__(self, n_assets: int, eta: float = 0.05):
        super().__init__(n_assets)
        self.eta = eta

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        portfolio_return = np.dot(current_weights, price_relative)
        if portfolio_return <= 0:
            return current_weights

        log_update = self.eta * price_relative / portfolio_return
        new_weights = current_weights * np.exp(log_update)

        # Normalize to simplex
        total = new_weights.sum()
        if total > 0:
            new_weights /= total
        return new_weights


class FTL(OnlineStrategy):
    """Follow the Leader — invest in the asset with best cumulative return.

    At each step, put all weight on the asset with the highest
    cumulative product of price relatives so far.
    """

    def __init__(self, n_assets: int):
        super().__init__(n_assets)
        self._cumulative = np.ones(n_assets)

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._cumulative *= price_relative
        best = np.argmax(self._cumulative)
        w = np.zeros(self.n_assets)
        w[best] = 1.0
        return w


class FTRL(OnlineStrategy):
    """Follow the Regularized Leader — FTL with entropic regularization.

    Softmax over cumulative log-returns with temperature parameter.

    b_{t+1,i} ∝ exp(η * Σ_{s=1}^{t} log(x_{s,i}))
    """

    def __init__(self, n_assets: int, eta: float = 0.1):
        super().__init__(n_assets)
        self.eta = eta
        self._cum_log_return = np.zeros(n_assets)

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._cum_log_return += np.log(np.maximum(price_relative, 1e-10))

        # Softmax
        scores = self.eta * self._cum_log_return
        scores -= scores.max()  # numerical stability
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum()
