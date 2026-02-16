"""Theory-Implied Correlation (TIC).

Constructs a correlation matrix implied by a hierarchical tree structure,
then de-noises using the Marchenko-Pastur bound. The TIC matrix preserves
the cluster structure found in the data while removing noise.

Based on Marcos Lopez de Prado (2020) approach:
1. Fit a hierarchical clustering tree to the empirical correlation matrix
2. Derive a correlation matrix implied by the tree (linkage distances → correlations)
3. De-noise using Marchenko-Pastur eigenvalue clipping
4. Optionally blend with empirical correlation (shrinkage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet, to_tree
from scipy.spatial.distance import squareform

from portopt.constants import LinkageMethod
from portopt.engine.risk import cov_to_corr, corr_to_cov, nearest_positive_definite


def theory_implied_correlation(
    covariance: pd.DataFrame,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    denoise: bool = True,
    shrinkage_alpha: float = 0.0,
) -> pd.DataFrame:
    """Compute the Theory-Implied Correlation matrix.

    Args:
        covariance: Empirical covariance matrix.
        linkage_method: Linkage method for hierarchical clustering.
        denoise: If True, apply Marchenko-Pastur de-noising.
        shrinkage_alpha: Blend factor between TIC (0) and empirical (1).

    Returns:
        TIC-adjusted correlation matrix as DataFrame.
    """
    symbols = list(covariance.index)
    corr = cov_to_corr(covariance).values
    n = len(symbols)

    # Step 1: Compute distance from correlation
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)

    # Step 2: Hierarchical clustering
    dist_condensed = squareform(dist)
    link = linkage(dist_condensed, method=linkage_method.value)

    # Step 3: Cophenetic correlation matrix (tree-implied distances)
    coph_dist = cophenet(link)  # returns condensed distance array
    coph_matrix = squareform(coph_dist)

    # Step 4: Convert cophenetic distances back to correlations
    # d = sqrt(0.5 * (1 - rho)) => rho = 1 - 2*d^2
    tic_corr = 1 - 2 * coph_matrix ** 2
    np.fill_diagonal(tic_corr, 1.0)

    # Clip to [-1, 1]
    tic_corr = np.clip(tic_corr, -1.0, 1.0)

    # Step 5: De-noise using Marchenko-Pastur (optional)
    if denoise:
        tic_corr = _denoise_corr(tic_corr)

    # Step 6: Shrinkage blend (optional)
    if 0 < shrinkage_alpha <= 1:
        tic_corr = (1 - shrinkage_alpha) * tic_corr + shrinkage_alpha * corr

    # Ensure positive semi-definite
    if not _is_psd(tic_corr):
        tic_corr = nearest_positive_definite(tic_corr)
        # Re-normalize diagonal to 1
        d = np.sqrt(np.diag(tic_corr))
        tic_corr = tic_corr / np.outer(d, d)
        np.fill_diagonal(tic_corr, 1.0)

    return pd.DataFrame(tic_corr, index=symbols, columns=symbols)


def theory_implied_covariance(
    covariance: pd.DataFrame,
    linkage_method: LinkageMethod = LinkageMethod.SINGLE,
    denoise: bool = True,
    shrinkage_alpha: float = 0.0,
) -> pd.DataFrame:
    """Compute TIC-adjusted covariance matrix.

    Applies TIC to the correlation structure while preserving
    the original asset volatilities.
    """
    std = pd.Series(np.sqrt(np.diag(covariance.values)), index=covariance.index)
    tic_corr = theory_implied_correlation(
        covariance, linkage_method, denoise, shrinkage_alpha
    )
    return corr_to_cov(tic_corr, std)


def _denoise_corr(corr: np.ndarray) -> np.ndarray:
    """De-noise a correlation matrix using Marchenko-Pastur eigenvalue clipping."""
    n = corr.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur upper bound (assuming q = T/N is large)
    # Use a heuristic: clip eigenvalues below the mean of small eigenvalues
    median_eig = np.median(eigenvalues)
    noise_mask = eigenvalues < median_eig

    if np.any(noise_mask) and not np.all(noise_mask):
        noise_mean = np.mean(eigenvalues[noise_mask])
        eigenvalues[noise_mask] = noise_mean

    # Reconstruct
    corr_dn = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    d = np.sqrt(np.diag(corr_dn))
    corr_dn = corr_dn / np.outer(d, d)
    np.fill_diagonal(corr_dn, 1.0)
    return corr_dn


def _is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if matrix is positive semi-definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigenvalues >= -tol))
