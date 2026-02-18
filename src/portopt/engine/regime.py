"""Market regime detection using Hidden Markov Models and rolling volatility.

Provides HMM-based regime classification (Bull/Bear/Crisis) and rolling
volatility regime indicators.  Pure computation — zero GUI dependencies.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RegimeInfo:
    """Descriptive statistics for a single market regime."""

    name: str  # "Bull", "Bear", "Crisis"
    mean_return: float  # Annualized
    volatility: float  # Annualized
    stationary_prob: float  # Long-run probability
    color: str  # Hex color for visualization


@dataclass
class RegimeResult:
    """Full output of HMM regime detection."""

    regimes: list[RegimeInfo]  # Sorted by volatility (low -> high)
    regime_sequence: np.ndarray  # Integer regime labels per date
    regime_probabilities: np.ndarray  # (T, n_regimes) posterior probs
    dates: list  # Corresponding dates
    transition_matrix: np.ndarray  # (n_regimes, n_regimes)
    current_regime: int  # Most recent regime index
    current_regime_name: str
    bic: float  # Model selection score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGIME_COLORS: dict[str, str] = {
    "Bull": "#00ff88",
    "Normal": "#f0b429",
    "Bear": "#f0b429",
    "Crisis": "#ff4757",
}

_MIN_OBSERVATIONS = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_regimes(
    returns: pd.Series | np.ndarray,
    n_regimes: int = 3,
    n_iter: int = 100,
) -> RegimeResult:
    """Fit a Gaussian HMM to a return series and classify market regimes.

    Args:
        returns: Daily return series.  If a ``pd.Series`` with a
            ``DatetimeIndex``, dates are preserved in the result.
        n_regimes: Number of hidden states (2, 3, or 4).
        n_iter: Maximum EM iterations for fitting.

    Returns:
        A :class:`RegimeResult` containing regime labels, posterior
        probabilities, transition matrix, and per-regime statistics.

    Raises:
        ValueError: If fewer than 30 observations are provided.
    """
    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    if isinstance(returns, pd.Series):
        dates = returns.index.tolist()
        X = returns.values.reshape(-1, 1)
    else:
        dates = list(range(len(returns)))
        X = np.asarray(returns).reshape(-1, 1)

    T = X.shape[0]
    if T < _MIN_OBSERVATIONS:
        raise ValueError(
            f"Need at least {_MIN_OBSERVATIONS} observations to fit an HMM, "
            f"got {T}."
        )

    # ------------------------------------------------------------------
    # Fit HMM (suppress convergence warnings from hmmlearn)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*did not converge.*")

        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
        )
        model.fit(X)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    regime_seq_raw = model.predict(X)
    proba_raw = model.predict_proba(X)

    # Per-component daily statistics (from model parameters)
    daily_means = model.means_.flatten()  # (n_regimes,)
    daily_stds = np.sqrt(model.covars_.flatten())  # (n_regimes,)

    # ------------------------------------------------------------------
    # Sort regimes by volatility (ascending) and relabel
    # ------------------------------------------------------------------
    sort_order = np.argsort(daily_stds)  # low vol first
    label_map = {old: new for new, old in enumerate(sort_order)}

    sorted_means = daily_means[sort_order]
    sorted_stds = daily_stds[sort_order]

    # Relabel sequence and reorder probability columns
    regime_sequence = np.array([label_map[r] for r in regime_seq_raw])
    regime_probabilities = proba_raw[:, sort_order]

    # Reorder transition matrix rows and columns
    transition_matrix = model.transmat_[np.ix_(sort_order, sort_order)]

    # ------------------------------------------------------------------
    # Regime names, colors, and annualized statistics
    # ------------------------------------------------------------------
    names = _label_regimes(sorted_means, sorted_stds, n_regimes)
    stationary_probs = _stationary_distribution(transition_matrix)

    regimes: list[RegimeInfo] = []
    for i in range(n_regimes):
        regimes.append(
            RegimeInfo(
                name=names[i],
                mean_return=float(sorted_means[i] * 252),
                volatility=float(sorted_stds[i] * np.sqrt(252)),
                stationary_prob=float(stationary_probs[i]),
                color=_REGIME_COLORS.get(names[i], "#aaaaaa"),
            )
        )

    # ------------------------------------------------------------------
    # BIC: -2 * logL + k * ln(T)
    # k = n*(n-1) transition probs + n means + n variances
    # ------------------------------------------------------------------
    log_likelihood = model.score(X) * T
    k = n_regimes * (n_regimes - 1) + n_regimes + n_regimes
    bic = -2 * log_likelihood + k * np.log(T)

    current_regime = int(regime_sequence[-1])

    logger.info(
        "HMM fit complete: %d regimes, BIC=%.1f, current=%s",
        n_regimes,
        bic,
        regimes[current_regime].name,
    )

    return RegimeResult(
        regimes=regimes,
        regime_sequence=regime_sequence,
        regime_probabilities=regime_probabilities,
        dates=dates,
        transition_matrix=transition_matrix,
        current_regime=current_regime,
        current_regime_name=regimes[current_regime].name,
        bic=bic,
    )


def rolling_regime(returns: pd.Series, window: int = 63) -> pd.Series:
    """Classify regimes via rolling realized volatility.

    Args:
        returns: Daily return series with a DatetimeIndex.
        window: Look-back window in trading days (default 63 ~ 1 quarter).

    Returns:
        A ``pd.Series`` of regime labels (``"Low Vol"``, ``"Normal"``,
        ``"High Vol"``) aligned with the input index.  The first
        *window - 1* entries are ``NaN``.
    """
    ann_vol = returns.rolling(window).std() * np.sqrt(252)
    q25 = ann_vol.quantile(0.25)
    q75 = ann_vol.quantile(0.75)

    labels = pd.Series(np.nan, index=returns.index, dtype=object)
    labels[ann_vol < q25] = "Low Vol"
    labels[ann_vol > q75] = "High Vol"
    labels[(ann_vol >= q25) & (ann_vol <= q75)] = "Normal"

    # Preserve NaN for the initial warm-up period
    labels.iloc[: window - 1] = np.nan
    return labels


def regime_conditional_parameters(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    regime_result: RegimeResult,
) -> dict[int, dict]:
    """Compute per-regime expected returns and covariance matrices.

    Args:
        returns: Daily returns DataFrame (index=date, columns=symbols).
        prices: Price DataFrame (unused but kept for API symmetry with
            other engine functions that accept prices).
        regime_result: Output of :func:`detect_regimes`.

    Returns:
        ``{regime_id: {"mu": pd.Series, "cov": pd.DataFrame}}`` where
        *mu* is annualized mean return per asset and *cov* is the
        annualized covariance matrix.
    """
    result: dict[int, dict] = {}
    regime_seq = regime_result.regime_sequence
    dates = regime_result.dates

    # Build a date -> regime mapping
    date_regime = pd.Series(regime_seq, index=dates)

    for regime_id in range(len(regime_result.regimes)):
        mask = date_regime == regime_id
        regime_dates = date_regime.index[mask]

        # Filter returns to the dates belonging to this regime
        common = returns.index.intersection(regime_dates)
        if len(common) < 2:
            logger.warning(
                "Regime %d (%s) has fewer than 2 overlapping dates; skipping.",
                regime_id,
                regime_result.regimes[regime_id].name,
            )
            continue

        r = returns.loc[common]
        mu = r.mean() * 252
        cov = r.cov() * 252

        result[regime_id] = {"mu": mu, "cov": cov}

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_regimes(
    means: np.ndarray,
    stds: np.ndarray,
    n_regimes: int,
) -> list[str]:
    """Assign human-readable names to regimes sorted by volatility.

    Regimes are assumed to be pre-sorted in ascending volatility order.

    Args:
        means: Array of daily mean returns per regime (sorted by vol).
        stds: Array of daily standard deviations per regime (sorted).
        n_regimes: Number of regimes.

    Returns:
        List of regime name strings.
    """
    if n_regimes == 2:
        return ["Bull", "Bear"]
    if n_regimes == 3:
        return ["Bull", "Bear", "Crisis"]
    if n_regimes == 4:
        return ["Bull", "Normal", "Bear", "Crisis"]
    # Fallback for arbitrary counts
    return [f"Regime {i}" for i in range(n_regimes)]


def _stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a Markov transition matrix.

    Solves pi @ P = pi subject to sum(pi) = 1 by finding the left
    eigenvector associated with eigenvalue 1.

    Args:
        transition_matrix: Row-stochastic (n x n) matrix.

    Returns:
        1-D array of stationary probabilities.
    """
    n = transition_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # Find eigenvector closest to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()  # Normalize to a probability distribution

    # Ensure non-negative (numerical noise can cause tiny negatives)
    pi = np.clip(pi, 0, None)
    pi = pi / pi.sum()
    return pi
