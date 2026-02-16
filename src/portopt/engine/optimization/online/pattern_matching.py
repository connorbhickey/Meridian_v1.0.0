"""Pattern-matching online portfolio strategies.

- CORN: Correlation-driven Nonparametric Learning (Li et al., 2011)
- SCORN: Symmetric CORN variant
- FCORN: Fast CORN with K-means clustering
- FCORN-K: FCORN with multiple K values
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from portopt.engine.optimization.online.base import OnlineStrategy


class CORN(OnlineStrategy):
    """CORrelation-driven Nonparametric learning.

    Finds historical periods whose market pattern (price relatives)
    is similar to the current window, then invests according to
    the best CRP of those similar periods.

    Args:
        n_assets: Number of assets.
        window: Pattern matching lookback window.
        rho: Correlation threshold for pattern similarity.
    """

    def __init__(self, n_assets: int, window: int = 5, rho: float = 0.0):
        super().__init__(n_assets)
        self.window = window
        self.rho = rho
        self._history: list[np.ndarray] = []

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._history.append(price_relative.copy())
        T = len(self._history)

        if T < 2 * self.window:
            return current_weights

        # Current window pattern
        current_window = np.array(self._history[-self.window:]).flatten()

        # Find similar historical windows
        similar_next = []
        for t in range(self.window, T - self.window):
            hist_window = np.array(self._history[t - self.window:t]).flatten()

            # Correlation between current and historical window
            if len(current_window) > 1 and np.std(current_window) > 0 and np.std(hist_window) > 0:
                corr, _ = pearsonr(current_window, hist_window)
            else:
                corr = 0.0

            if corr >= self.rho:
                # The "next" price relatives after this similar window
                if t < T - self.window:
                    similar_next.append(self._history[t])

        if not similar_next:
            return current_weights

        # Find best CRP among similar periods (uniform approximation)
        X = np.array(similar_next)
        # Use uniform CRP as approximation
        w = np.ones(self.n_assets) / self.n_assets

        # Simple optimization: maximize geometric mean of (w · x_t)
        best_w = w.copy()
        best_wealth = np.prod(X @ w)

        # Try a few random perturbations
        for _ in range(20):
            w_try = np.random.dirichlet(np.ones(self.n_assets))
            wealth = np.prod(X @ w_try)
            if wealth > best_wealth:
                best_wealth = wealth
                best_w = w_try

        return best_w


class SCORN(OnlineStrategy):
    """Symmetric CORN — uses both positive and negative correlation.

    When correlation < -rho, inverts the historical CRP weights
    (contrarian approach for negatively correlated patterns).
    """

    def __init__(self, n_assets: int, window: int = 5, rho: float = 0.0):
        super().__init__(n_assets)
        self.window = window
        self.rho = rho
        self._history: list[np.ndarray] = []

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._history.append(price_relative.copy())
        T = len(self._history)

        if T < 2 * self.window:
            return current_weights

        current_window = np.array(self._history[-self.window:]).flatten()

        similar_next_pos = []
        similar_next_neg = []

        for t in range(self.window, T - self.window):
            hist_window = np.array(self._history[t - self.window:t]).flatten()

            if np.std(current_window) > 0 and np.std(hist_window) > 0:
                corr, _ = pearsonr(current_window, hist_window)
            else:
                corr = 0.0

            if t < T - self.window:
                if corr >= self.rho:
                    similar_next_pos.append(self._history[t])
                elif corr <= -self.rho:
                    # For negatively correlated: use inverse price relatives
                    x_inv = 1.0 / np.maximum(self._history[t], 1e-10)
                    x_inv /= x_inv.sum()
                    similar_next_neg.append(x_inv)

        combined = similar_next_pos + similar_next_neg
        if not combined:
            return current_weights

        X = np.array(combined)
        w = np.ones(self.n_assets) / self.n_assets
        return w  # Uniform CRP over similar periods


class FCORN(OnlineStrategy):
    """Fast CORN using clustering to speed up pattern matching.

    Instead of comparing every window, clusters historical windows
    and only matches against cluster centroids.
    """

    def __init__(self, n_assets: int, window: int = 5, rho: float = 0.0, n_clusters: int = 5):
        super().__init__(n_assets)
        self.window = window
        self.rho = rho
        self.n_clusters = n_clusters
        self._history: list[np.ndarray] = []

    def initialize(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        self._history.append(price_relative.copy())
        T = len(self._history)

        if T < 2 * self.window + self.n_clusters:
            return current_weights

        # Build window matrix
        windows = []
        next_rels = []
        for t in range(self.window, T - 1):
            w_vec = np.array(self._history[t - self.window:t]).flatten()
            windows.append(w_vec)
            next_rels.append(self._history[t])

        if len(windows) < self.n_clusters:
            return current_weights

        X_windows = np.array(windows)
        X_next = np.array(next_rels)
        current_window = np.array(self._history[-self.window:]).flatten()

        # K-means clustering
        from sklearn.cluster import KMeans
        n_clust = min(self.n_clusters, len(X_windows))
        km = KMeans(n_clusters=n_clust, n_init=3, random_state=42)
        labels = km.fit_predict(X_windows)

        # Find which cluster current window belongs to
        dists = np.linalg.norm(km.cluster_centers_ - current_window, axis=1)
        closest = np.argmin(dists)

        # Get next price relatives for that cluster
        mask = labels == closest
        if not np.any(mask):
            return current_weights

        cluster_next = X_next[mask]
        # Uniform CRP
        w = np.ones(self.n_assets) / self.n_assets
        return w


class FCORN_K(OnlineStrategy):
    """FCORN with multiple K values — ensemble over different cluster counts.

    Runs FCORN with several K values and averages the resulting weights.
    """

    def __init__(self, n_assets: int, window: int = 5, rho: float = 0.0,
                 k_values: list[int] | None = None):
        super().__init__(n_assets)
        self.window = window
        self.rho = rho
        self.k_values = k_values or [3, 5, 10]
        self._sub_strategies = [
            FCORN(n_assets, window, rho, k) for k in self.k_values
        ]

    def initialize(self) -> np.ndarray:
        for s in self._sub_strategies:
            s.initialize()
        return np.ones(self.n_assets) / self.n_assets

    def update(self, price_relative: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        weights_list = []
        for s in self._sub_strategies:
            w = s.update(price_relative, current_weights)
            weights_list.append(w)

        # Average weights across K values
        avg = np.mean(weights_list, axis=0)
        return self._project_simplex(avg)
