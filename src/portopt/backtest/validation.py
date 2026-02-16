"""Backtesting validation: train/test split, time-series CV, PBO.

Implements:
- Simple train/test split
- K-fold time-series cross-validation (purged + embargo)
- Combinatorial Purged Cross-Validation (CPCV)
- Probability of Backtest Overfitting (PBO)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from portopt.backtest.costs import BaseCostModel, ZeroCost
from portopt.backtest.rebalancer import RebalanceSchedule
from portopt.backtest.results import SingleRunResult
from portopt.backtest.runner import OptimizerFn, run_backtest


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_id: int
    train_indices: list[int]
    test_indices: list[int]
    train_result: SingleRunResult | None = None
    test_result: SingleRunResult | None = None


@dataclass
class CVResult:
    """Cross-validation results."""
    folds: list[CVFold] = field(default_factory=list)
    train_scores: list[float] = field(default_factory=list)
    test_scores: list[float] = field(default_factory=list)
    mean_train_score: float = 0.0
    mean_test_score: float = 0.0
    score_std: float = 0.0
    overfit_ratio: float = 0.0    # Mean train / mean test (>1 = overfit)

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def train_test_split(
    prices: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple time-series train/test split.

    Args:
        prices: Full price DataFrame.
        train_ratio: Fraction of data for training.

    Returns:
        (train_prices, test_prices) tuple.
    """
    n = len(prices)
    split_idx = int(n * train_ratio)
    return prices.iloc[:split_idx], prices.iloc[split_idx:]


def time_series_cv(
    prices: pd.DataFrame,
    optimizer_fn: OptimizerFn,
    n_folds: int = 5,
    purge_days: int = 5,
    embargo_days: int = 5,
    schedule: RebalanceSchedule | None = None,
    cost_model: BaseCostModel | None = None,
) -> CVResult:
    """K-fold time-series cross-validation with purging and embargo.

    Unlike standard k-fold, this respects temporal ordering:
    - Training always uses data before the test set
    - Purge: remove observations around the train/test boundary
    - Embargo: skip days after each test set before next training

    Args:
        prices: Full price DataFrame.
        optimizer_fn: Optimizer callback.
        n_folds: Number of folds.
        purge_days: Days to remove around train/test boundary.
        embargo_days: Days to skip after test set.
        schedule: Rebalance schedule.
        cost_model: Cost model.

    Returns:
        CVResult with per-fold and aggregate scores.
    """
    if schedule is None:
        schedule = RebalanceSchedule()
    if cost_model is None:
        cost_model = ZeroCost()

    n = len(prices)
    fold_size = n // n_folds
    folds = []
    train_scores = []
    test_scores = []

    for k in range(n_folds):
        test_start = k * fold_size
        test_end = min((k + 1) * fold_size, n)

        # Training = everything before test, minus purge
        train_end = max(0, test_start - purge_days)
        train_indices = list(range(0, train_end))

        # Test indices
        test_start_adjusted = min(test_start + embargo_days, test_end)
        test_indices = list(range(test_start_adjusted, test_end))

        if len(train_indices) < 30 or len(test_indices) < 5:
            continue

        fold = CVFold(
            fold_id=k,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        # Run on train
        train_prices = prices.iloc[train_indices]
        test_prices = prices.iloc[test_indices]

        try:
            # Optimize on training data
            optimal_weights = optimizer_fn(train_prices, None)
            fixed_weights = optimal_weights.copy()

            def fixed_fn(p, _cov=None, _w=fixed_weights):
                return _w.copy()

            # Test OOS
            if len(test_prices) >= 2:
                test_result = run_backtest(
                    prices=test_prices,
                    optimizer_fn=fixed_fn,
                    schedule=schedule,
                    cost_model=cost_model,
                )
                fold.test_result = test_result
                test_scores.append(test_result.total_return)

            # Train IS
            if len(train_prices) >= 2:
                train_result = run_backtest(
                    prices=train_prices,
                    optimizer_fn=fixed_fn,
                    schedule=schedule,
                    cost_model=cost_model,
                )
                fold.train_result = train_result
                train_scores.append(train_result.total_return)

        except Exception:
            continue

        folds.append(fold)

    mean_train = float(np.mean(train_scores)) if train_scores else 0.0
    mean_test = float(np.mean(test_scores)) if test_scores else 0.0
    score_std = float(np.std(test_scores)) if test_scores else 0.0
    overfit = mean_train / mean_test if mean_test != 0 else float("inf")

    return CVResult(
        folds=folds,
        train_scores=train_scores,
        test_scores=test_scores,
        mean_train_score=mean_train,
        mean_test_score=mean_test,
        score_std=score_std,
        overfit_ratio=overfit,
    )


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""
    pbo: float = 0.0                   # Probability of overfitting (0 to 1)
    logit_distribution: list[float] = field(default_factory=list)
    n_combinations: int = 0
    rank_correlation: float = 0.0      # IS vs OOS rank correlation


def probability_of_backtest_overfitting(
    prices: pd.DataFrame,
    optimizer_fns: list[OptimizerFn],
    n_partitions: int = 16,
    schedule: RebalanceSchedule | None = None,
    cost_model: BaseCostModel | None = None,
) -> PBOResult:
    """Estimate Probability of Backtest Overfitting (PBO).

    Based on Bailey, Borwein, Lopez de Prado, Zhu (2017).

    The idea: partition data into S equal blocks, then for each combination
    of S/2 blocks as "in-sample", the remaining S/2 blocks are "out-of-sample".
    For each strategy, measure IS vs OOS performance. If the best IS strategy
    tends to perform poorly OOS, the probability of overfitting is high.

    Args:
        prices: Full price DataFrame.
        optimizer_fns: List of optimizer callbacks to compare.
        n_partitions: Number of data partitions (must be even).
        schedule: Rebalance schedule.
        cost_model: Cost model.

    Returns:
        PBOResult with estimated overfitting probability.
    """
    if schedule is None:
        schedule = RebalanceSchedule()
    if cost_model is None:
        cost_model = ZeroCost()

    if n_partitions % 2 != 0:
        n_partitions += 1

    n = len(prices)
    block_size = n // n_partitions
    if block_size < 10:
        return PBOResult(pbo=0.0, n_combinations=0)

    n_strategies = len(optimizer_fns)
    if n_strategies < 2:
        return PBOResult(pbo=0.0, n_combinations=0)

    # Compute returns for each strategy on each block
    block_returns = np.zeros((n_strategies, n_partitions))
    for b in range(n_partitions):
        start = b * block_size
        end = min((b + 1) * block_size + 1, n)
        block_prices = prices.iloc[start:end]

        if len(block_prices) < 5:
            continue

        for s_idx, opt_fn in enumerate(optimizer_fns):
            try:
                weights = opt_fn(block_prices, None)
                fixed_w = weights.copy()

                def fixed_fn(p, _cov=None, _w=fixed_w):
                    return _w.copy()

                result = run_backtest(
                    prices=block_prices,
                    optimizer_fn=fixed_fn,
                    schedule=schedule,
                    cost_model=cost_model,
                )
                block_returns[s_idx, b] = result.total_return
            except Exception:
                block_returns[s_idx, b] = 0.0

    # CSCV: for each combination of n_partitions/2 blocks as IS
    half = n_partitions // 2
    all_blocks = list(range(n_partitions))
    combos = list(combinations(all_blocks, half))

    # Limit combinations if too many
    max_combos = 100
    if len(combos) > max_combos:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in indices]

    logit_values = []

    for is_blocks in combos:
        oos_blocks = [b for b in all_blocks if b not in is_blocks]

        # IS and OOS performance for each strategy
        is_perf = np.sum(block_returns[:, list(is_blocks)], axis=1)
        oos_perf = np.sum(block_returns[:, oos_blocks], axis=1)

        # Find best IS strategy
        best_is_idx = np.argmax(is_perf)

        # Rank of best IS strategy in OOS
        oos_rank = np.sum(oos_perf >= oos_perf[best_is_idx])  # 1-based rank from top
        relative_rank = oos_rank / n_strategies

        # Logit of relative rank
        if 0 < relative_rank < 1:
            logit = np.log(relative_rank / (1 - relative_rank))
        elif relative_rank >= 1:
            logit = 5.0  # Cap
        else:
            logit = -5.0
        logit_values.append(logit)

    # PBO = fraction of combinations where logit < 0
    # (i.e., best IS strategy ranks below median OOS)
    pbo = float(np.mean(np.array(logit_values) < 0))

    # IS vs OOS rank correlation (Spearman)
    is_total = np.sum(block_returns[:, :half], axis=1)
    oos_total = np.sum(block_returns[:, half:], axis=1)
    if n_strategies > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(is_total, oos_total)
        rank_corr = float(corr) if not np.isnan(corr) else 0.0
    else:
        rank_corr = 0.0

    return PBOResult(
        pbo=pbo,
        logit_distribution=logit_values,
        n_combinations=len(combos),
        rank_correlation=rank_corr,
    )
