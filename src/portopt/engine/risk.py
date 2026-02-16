"""Covariance matrix estimators: sample, Ledoit-Wolf, de-noised, exponential."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from portopt.constants import CovEstimator
from portopt.engine.returns import log_returns


def estimate_covariance(
    prices: pd.DataFrame,
    method: CovEstimator = CovEstimator.SAMPLE,
    **kwargs,
) -> pd.DataFrame:
    """Estimate the covariance matrix of asset returns.

    Args:
        prices: DataFrame of close prices (index=date, columns=symbols).
        method: Which estimator to use.
        **kwargs: Method-specific parameters.

    Returns:
        Annualized covariance matrix DataFrame (symbols x symbols).
    """
    estimators = {
        CovEstimator.SAMPLE: _sample_covariance,
        CovEstimator.LEDOIT_WOLF: _ledoit_wolf,
        CovEstimator.DENOISED: _denoised,
        CovEstimator.EXPONENTIAL: _exponential_covariance,
    }
    fn = estimators[method]
    return fn(prices, **kwargs)


# ── Sample covariance ─────────────────────────────────────────────────


def _sample_covariance(
    prices: pd.DataFrame,
    frequency: int = 252,
    **_kwargs,
) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    rets = log_returns(prices)
    return rets.cov() * frequency


# ── Ledoit-Wolf shrinkage ─────────────────────────────────────────────


def _ledoit_wolf(
    prices: pd.DataFrame,
    frequency: int = 252,
    **_kwargs,
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage estimator (sklearn)."""
    from sklearn.covariance import LedoitWolf

    rets = log_returns(prices)
    lw = LedoitWolf().fit(rets.values)
    cov = pd.DataFrame(lw.covariance_, index=rets.columns, columns=rets.columns)
    return cov * frequency


# ── De-noised covariance (Marchenko-Pastur) ──────────────────────────


def _denoised(
    prices: pd.DataFrame,
    frequency: int = 252,
    **_kwargs,
) -> pd.DataFrame:
    """De-noised covariance using the Marchenko-Pastur theorem.

    Follows the approach from Marcos Lopez de Prado (2020):
    1. Compute correlation matrix from returns
    2. Compute eigenvalues/eigenvectors
    3. Find the Marchenko-Pastur max eigenvalue (noise threshold)
    4. Shrink eigenvalues below threshold toward their average
    5. Reconstruct the correlation matrix
    6. Convert back to covariance
    """
    rets = log_returns(prices)
    T, N = rets.shape
    q = T / N  # observations-to-variables ratio

    # Sample covariance and correlation
    sample_cov = rets.cov()
    std = np.sqrt(np.diag(sample_cov.values))
    corr = rets.corr().values

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur upper bound: lambda_+ = sigma^2 * (1 + 1/q + 2*sqrt(1/q))
    # For identity target, sigma^2 = 1
    lambda_plus = (1 + np.sqrt(1 / q)) ** 2

    # Separate signal from noise eigenvalues
    n_noise = np.sum(eigenvalues <= lambda_plus)
    if n_noise > 0 and n_noise < N:
        noise_mean = np.mean(eigenvalues[eigenvalues <= lambda_plus])
        eigenvalues[eigenvalues <= lambda_plus] = noise_mean

    # Reconstruct correlation
    corr_denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Ensure unit diagonal
    d = np.sqrt(np.diag(corr_denoised))
    corr_denoised = corr_denoised / np.outer(d, d)
    np.fill_diagonal(corr_denoised, 1.0)

    # Convert back to covariance
    cov_denoised = np.outer(std, std) * corr_denoised
    result = pd.DataFrame(cov_denoised, index=sample_cov.index, columns=sample_cov.columns)
    return result * frequency


# ── Exponentially-weighted covariance ─────────────────────────────────


def _exponential_covariance(
    prices: pd.DataFrame,
    span: int = 60,
    frequency: int = 252,
    **_kwargs,
) -> pd.DataFrame:
    """Exponentially-weighted covariance matrix."""
    rets = log_returns(prices)
    ewm_cov = rets.ewm(span=span).cov()
    # Extract the last cross-section
    last_date = rets.index[-1]
    cov = ewm_cov.loc[last_date]
    return cov * frequency


# ── Utility functions ─────────────────────────────────────────────────


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def corr_to_cov(corr: pd.DataFrame, std: pd.Series) -> pd.DataFrame:
    """Convert correlation matrix to covariance using given standard deviations."""
    cov = corr.values * np.outer(std.values, std.values)
    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Find the nearest positive-definite matrix (Higham, 2002)."""
    B = (matrix + matrix.T) / 2
    _, s, V = linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if is_positive_definite(A3):
        return A3
    spacing = np.spacing(linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(A3):
        min_eig = np.min(np.real(linalg.eigvals(A3)))
        A3 += I * (-min_eig * k ** 2 + spacing)
        k += 1
    return A3
