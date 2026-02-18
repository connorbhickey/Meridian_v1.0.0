"""Fama-French factor models and factor exposure analysis.

Constructs proxy Fama-French three-factor models (MKT-RF, SMB, HML) from
price data and computes per-asset and portfolio-level factor exposures
via OLS regression.  Zero GUI imports — pure computation module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

from portopt.engine.returns import simple_returns

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════


@dataclass
class FactorExposure:
    """Single factor loading for one asset (or the portfolio)."""

    factor_name: str
    beta: float  # Factor loading
    t_stat: float  # Statistical significance
    p_value: float
    r_squared: float  # Factor's explanatory power


@dataclass
class FactorModelResult:
    """Complete output of a factor analysis run."""

    asset_exposures: dict[str, list[FactorExposure]]  # {symbol: [exposures]}
    portfolio_exposures: list[FactorExposure]  # Weighted portfolio betas
    factor_returns: pd.DataFrame  # Factor return series used
    residual_returns: pd.DataFrame  # Alpha (unexplained) returns
    r_squared: dict[str, float]  # Per-asset R²


# ══════════════════════════════════════════════════════════════════════
# Factor construction
# ══════════════════════════════════════════════════════════════════════


def build_fama_french_factors(
    prices: pd.DataFrame,
    market_prices: pd.Series | None = None,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """Construct proxy Fama-French 3 factors from price data.

    Factors:
        MKT-RF: Market excess return (market minus daily risk-free rate).
        SMB:    Small Minus Big — high-volatility tercile minus low-volatility
                tercile daily returns (proxy for size using rolling vol).
        HML:    High Minus Low — low-momentum tercile minus high-momentum
                tercile daily returns (proxy for value using 12-month momentum).

    Args:
        prices: DataFrame of close prices (index=date, columns=symbols).
        market_prices: Optional market index price series.  If *None*, the
            equal-weighted average of all asset returns is used.
        risk_free_rate: Annualized risk-free rate (default 4 %).

    Returns:
        DataFrame with columns ``['MKT-RF', 'SMB', 'HML']`` indexed by date.
    """
    rets = simple_returns(prices)
    daily_rf = risk_free_rate / 252

    # ── MKT-RF ────────────────────────────────────────────────────────
    if market_prices is not None:
        mkt_ret = simple_returns(market_prices.to_frame()).iloc[:, 0]
        # Align to asset return dates
        common_idx = rets.index.intersection(mkt_ret.index)
        mkt_ret = mkt_ret.loc[common_idx]
        rets_aligned = rets.loc[common_idx]
    else:
        mkt_ret = rets.mean(axis=1)
        rets_aligned = rets

    mkt_rf = mkt_ret - daily_rf

    # Need at least 252 observations for rolling sorts
    min_obs = 252
    if len(rets_aligned) < min_obs:
        logger.warning(
            "Only %d observations available (need %d for rolling sorts); "
            "factor construction will use available data with reduced lookback",
            len(rets_aligned),
            min_obs,
        )
        # Use whatever we have, with a reduced window
        vol_window = max(len(rets_aligned) // 2, 20)
        mom_window = max(len(rets_aligned) // 2, 20)
    else:
        vol_window = 252
        mom_window = 252

    n_assets = rets_aligned.shape[1]
    # Need at least 3 assets to form terciles
    if n_assets < 3:
        logger.warning(
            "Only %d assets — cannot form tercile sorts; "
            "returning MKT-RF only (SMB and HML set to 0)",
            n_assets,
        )
        factor_df = pd.DataFrame(
            {"MKT-RF": mkt_rf, "SMB": 0.0, "HML": 0.0},
            index=mkt_rf.index,
        )
        return factor_df

    # ── SMB (volatility proxy for size) ───────────────────────────────
    rolling_vol = rets_aligned.rolling(window=vol_window, min_periods=vol_window // 2).std()

    # ── HML (momentum proxy for value) ────────────────────────────────
    rolling_mom = rets_aligned.rolling(window=mom_window, min_periods=mom_window // 2).sum()

    smb_series = pd.Series(np.nan, index=rets_aligned.index, dtype=float)
    hml_series = pd.Series(np.nan, index=rets_aligned.index, dtype=float)

    symbols = rets_aligned.columns.tolist()

    for dt in rets_aligned.index:
        day_ret = rets_aligned.loc[dt]

        # --- SMB sort by volatility ---
        vol_row = rolling_vol.loc[dt]
        valid_vol = vol_row.dropna()
        if len(valid_vol) < 3:
            continue

        vol_median = valid_vol.median()
        high_vol_syms = valid_vol[valid_vol >= vol_median].index.tolist()
        low_vol_syms = valid_vol[valid_vol < vol_median].index.tolist()

        if high_vol_syms and low_vol_syms:
            smb_series.loc[dt] = (
                day_ret[high_vol_syms].mean() - day_ret[low_vol_syms].mean()
            )

        # --- HML sort by momentum ---
        mom_row = rolling_mom.loc[dt]
        valid_mom = mom_row.dropna()
        if len(valid_mom) < 3:
            continue

        mom_median = valid_mom.median()
        low_mom_syms = valid_mom[valid_mom < mom_median].index.tolist()
        high_mom_syms = valid_mom[valid_mom >= mom_median].index.tolist()

        # HML: low-momentum minus high-momentum (value proxy)
        if low_mom_syms and high_mom_syms:
            hml_series.loc[dt] = (
                day_ret[low_mom_syms].mean() - day_ret[high_mom_syms].mean()
            )

    factor_df = pd.DataFrame(
        {"MKT-RF": mkt_rf, "SMB": smb_series, "HML": hml_series},
        index=mkt_rf.index,
    )
    factor_df.dropna(inplace=True)

    return factor_df


# ══════════════════════════════════════════════════════════════════════
# Factor exposure estimation
# ══════════════════════════════════════════════════════════════════════


def compute_factor_exposures(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> dict[str, list[FactorExposure]]:
    """Regress each asset's returns on factor returns via OLS.

    For each asset column in *returns*:
        R_i = alpha + beta_1 * F_1 + beta_2 * F_2 + ... + epsilon

    Args:
        returns: Daily asset returns (index=date, columns=symbols).
        factor_returns: Daily factor returns (index=date,
            columns=factor names).

    Returns:
        Dict mapping symbol to a list of :class:`FactorExposure` objects
        (one per factor).
    """
    # Align dates (inner join)
    common_idx = returns.index.intersection(factor_returns.index)
    if len(common_idx) == 0:
        logger.warning("No overlapping dates between returns and factor_returns")
        return {}

    rets = returns.loc[common_idx]
    factors = factor_returns.loc[common_idx]

    X = factors.values  # (T, K)
    factor_names = factors.columns.tolist()
    T, K = X.shape

    exposures: dict[str, list[FactorExposure]] = {}

    for symbol in rets.columns:
        y = rets[symbol].values  # (T,)

        # Skip assets with insufficient or non-finite data
        valid_mask = np.isfinite(y)
        if valid_mask.sum() < K + 2:
            logger.warning(
                "Skipping %s: only %d valid observations (need at least %d)",
                symbol,
                valid_mask.sum(),
                K + 2,
            )
            continue

        y_clean = y[valid_mask]
        X_clean = X[valid_mask]

        model = LinearRegression(fit_intercept=True)
        model.fit(X_clean, y_clean)

        y_pred = model.predict(X_clean)
        residuals = y_clean - y_pred
        n = len(y_clean)
        p = K + 1  # number of parameters (K betas + intercept)
        dof = n - p

        if dof <= 0:
            logger.warning(
                "Skipping %s: degrees of freedom <= 0 (n=%d, p=%d)",
                symbol,
                n,
                p,
            )
            continue

        # MSE and standard errors of coefficients
        mse = np.sum(residuals ** 2) / dof

        # Design matrix with intercept column for variance computation
        X_design = np.column_stack([np.ones(n), X_clean])  # (n, p)
        try:
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
        except np.linalg.LinAlgError:
            logger.warning(
                "Skipping %s: singular X'X matrix — factors may be collinear",
                symbol,
            )
            continue

        se_all = np.sqrt(np.abs(mse * np.diag(XtX_inv)))
        # se_all[0] is SE for intercept; se_all[1:] for factor betas
        se_betas = se_all[1:]

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        asset_exposures: list[FactorExposure] = []
        for i, fname in enumerate(factor_names):
            beta = float(model.coef_[i])
            se = float(se_betas[i]) if se_betas[i] > 0 else 1e-10
            t_stat = beta / se
            # Two-tailed p-value
            p_value = float(2.0 * sp_stats.t.sf(abs(t_stat), dof))

            asset_exposures.append(
                FactorExposure(
                    factor_name=fname,
                    beta=beta,
                    t_stat=t_stat,
                    p_value=p_value,
                    r_squared=r_squared,
                )
            )

        exposures[symbol] = asset_exposures

    return exposures


# ══════════════════════════════════════════════════════════════════════
# Portfolio-level exposures
# ══════════════════════════════════════════════════════════════════════


def compute_portfolio_factor_exposures(
    weights: dict[str, float],
    asset_exposures: dict[str, list[FactorExposure]],
) -> list[FactorExposure]:
    """Compute weighted-average factor exposures for a portfolio.

    Args:
        weights: Portfolio weights {symbol: weight}.
        asset_exposures: Per-asset factor exposures (from
            :func:`compute_factor_exposures`).

    Returns:
        List of :class:`FactorExposure` — one per factor — representing
        the portfolio-level loadings.
    """
    if not asset_exposures:
        return []

    # Determine factor names from the first asset
    first_key = next(iter(asset_exposures))
    factor_names = [e.factor_name for e in asset_exposures[first_key]]

    # Collect weight-adjusted values
    total_weight = 0.0
    weighted_beta = {f: 0.0 for f in factor_names}
    weighted_tstat = {f: 0.0 for f in factor_names}
    weighted_r2 = {f: 0.0 for f in factor_names}

    for symbol, exps in asset_exposures.items():
        w = weights.get(symbol, 0.0)
        total_weight += w
        for exp in exps:
            weighted_beta[exp.factor_name] += w * exp.beta
            weighted_tstat[exp.factor_name] += w * exp.t_stat
            weighted_r2[exp.factor_name] += w * exp.r_squared

    # Normalize by total weight to handle non-unit-sum weights
    if total_weight > 0:
        for f in factor_names:
            weighted_beta[f] /= total_weight
            weighted_tstat[f] /= total_weight
            weighted_r2[f] /= total_weight

    portfolio_exposures: list[FactorExposure] = []
    for f in factor_names:
        # Recompute p-value from the weight-averaged t-stat
        # Use a conservative dof estimate (100)
        dof_est = 100
        p_value = float(2.0 * sp_stats.t.sf(abs(weighted_tstat[f]), dof_est))

        portfolio_exposures.append(
            FactorExposure(
                factor_name=f,
                beta=weighted_beta[f],
                t_stat=weighted_tstat[f],
                p_value=p_value,
                r_squared=weighted_r2[f],
            )
        )

    return portfolio_exposures


# ══════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════


def run_factor_analysis(
    prices: pd.DataFrame,
    weights: dict[str, float],
    factor_returns: pd.DataFrame | None = None,
) -> FactorModelResult:
    """Run the full Fama-French factor analysis pipeline.

    Steps:
        1. Compute simple returns from prices.
        2. Build proxy FF3 factors (if *factor_returns* not provided).
        3. Compute per-asset factor exposures via OLS.
        4. Compute portfolio-level weighted exposures.
        5. Compute residual (alpha) returns.

    Args:
        prices: DataFrame of close prices (index=date, columns=symbols).
        weights: Portfolio weights {symbol: weight}.
        factor_returns: Optional pre-built factor returns DataFrame.
            If *None*, factors are constructed from the price data via
            :func:`build_fama_french_factors`.

    Returns:
        :class:`FactorModelResult` with all analysis outputs.
    """
    # Step 1 — compute returns
    rets = simple_returns(prices)

    # Step 2 — build factors if not provided
    if factor_returns is None:
        logger.info("Building proxy Fama-French factors from price data")
        factor_returns = build_fama_french_factors(prices)

    # Step 3 — compute per-asset exposures
    asset_exposures = compute_factor_exposures(rets, factor_returns)

    # Step 4 — compute portfolio exposures
    portfolio_exposures = compute_portfolio_factor_exposures(
        weights, asset_exposures
    )

    # Step 5 — compute residual returns (actual - predicted)
    common_idx = rets.index.intersection(factor_returns.index)
    rets_aligned = rets.loc[common_idx]
    factors_aligned = factor_returns.loc[common_idx]

    X = factors_aligned.values
    residual_data: dict[str, np.ndarray] = {}

    for symbol in rets_aligned.columns:
        if symbol not in asset_exposures:
            continue

        y = rets_aligned[symbol].values
        model = LinearRegression(fit_intercept=True)

        valid_mask = np.isfinite(y)
        if valid_mask.sum() < X.shape[1] + 2:
            continue

        model.fit(X[valid_mask], y[valid_mask])
        y_pred = np.full_like(y, np.nan, dtype=float)
        y_pred[valid_mask] = model.predict(X[valid_mask])
        residual_data[symbol] = y - y_pred

    residual_returns = pd.DataFrame(
        residual_data, index=common_idx
    )

    # Per-asset R²
    r_squared: dict[str, float] = {}
    for symbol, exps in asset_exposures.items():
        if exps:
            r_squared[symbol] = exps[0].r_squared

    return FactorModelResult(
        asset_exposures=asset_exposures,
        portfolio_exposures=portfolio_exposures,
        factor_returns=factor_returns,
        residual_returns=residual_returns,
        r_squared=r_squared,
    )
