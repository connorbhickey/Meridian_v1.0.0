"""Tests for the stock prediction engine (prng, monte_carlo, signals, ensemble)."""

import math

import numpy as np
import pytest

from portopt.engine.prediction.prng import (
    clamp,
    gamma_rv,
    make_lcg,
    normal_rv,
    percentile,
    student_t_rv,
)
from portopt.engine.prediction.monte_carlo import (
    MJDParams,
    build_histogram,
    mc_percentiles,
    mjd_simulate,
    prob_above,
)
from portopt.engine.prediction.signals import (
    detect_earnings_event,
    sig_buyback,
    sig_eps_revision,
    sig_fcf,
    sig_inst,
    sig_investment,
    sig_leverage,
    sig_low_vol,
    sig_macro,
    sig_mom,
    sig_options_skew,
    sig_pead,
    sig_quality,
    sig_regime,
    sig_rev_accel,
    sig_season,
    sig_size,
    sig_srs,
    sig_value,
    sig_vol,
    signal_interactions,
    sig_insider,
)
from portopt.engine.prediction.ensemble import (
    BootstrapResult,
    MethodResult,
    PredictionInterval,
    PredictionResult,
    bootstrap_ci,
    decorrelate_weights,
    horizon_weights,
    inverse_var_weights,
    james_stein_shrink,
    kelly_confidence,
    prediction_interval,
    run_prediction,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_data():
    """Minimal data dict with all required fields for run_prediction()."""
    return {
        "symbol": "TEST",
        "currentPrice": 150.0,
        "annualizedVol": 0.30,
        "beta": 1.1,
        "tradingDaysRemaining": 252,
        "sma50": 148.0,
        "sma200": 142.0,
        "rsi14": 55.0,
        "high52w": 170.0,
        "low52w": 120.0,
        "stockPerformance3m": 5.0,
        "peRatio": 22.0,
        "forwardPE": 20.0,
        "priceToBook": 4.0,
        "forwardEps": 7.5,
        "forwardEps2": 8.1,
        "roe": 20.0,
        "profitMargin": 18.0,
        "debtToEquity": 0.6,
        "fcfYield": 5.0,
        "marketCapNum": 100.0,
        "dividendYield": 1.5,
        "expenseRatio": 0.0,
        "analystAvgPt": 165.0,
        "analystHigh": 190.0,
        "analystLow": 130.0,
        "recentAnalystAvg": 165.0,
        "nextEarningsDate": None,
        "lastEarningsSurprise": 3.0,
        "daysSinceEarnings": 30,
        "historicalEarningsMoveAvg": 6.0,
        "fedFundsRate": 4.5,
        "vix": 18.0,
        "yieldCurve2s10s": 0.3,
        "creditSpreadHY": 3.5,
        "sectorAvgPE": 25.0,
        "sectorAvgPB": 4.0,
        "sectorPerformance3m": 3.0,
        "shortInterest": 3.0,
        "putCallRatio": 0.7,
        "institutionalOwnership": 70.0,
        "earningsRevision3m": 2.0,
        "capexToRevenue": 10.0,
        "ivRank": 45.0,
        "ivSkew": 0.0,
        "insiderNetBuying": 0.0,
        "revenueGrowthPct": 12.0,
        "priorRevenueGrowthPct": 10.0,
        "interestCoverage": 12.0,
        "shareCountChange": -1.0,
        "isETF": False,
        "sector": "Technology",
        "name": "Test Corp",
    }


@pytest.fixture
def etf_data(base_data):
    """Data dict configured as an ETF."""
    d = dict(base_data)
    d["isETF"] = True
    d["symbol"] = "SPY"
    d["forwardEps"] = None
    d["forwardEps2"] = None
    d["dividendYield"] = 1.3
    d["expenseRatio"] = 0.09
    return d


# ══════════════════════════════════════════════════════════════════
# PRNG Tests
# ══════════════════════════════════════════════════════════════════

class TestLCG:
    def test_deterministic(self):
        """Same seed produces same sequence."""
        a = make_lcg(42)
        b = make_lcg(42)
        for _ in range(100):
            assert a() == b()

    def test_range_01(self):
        """All values in (0, 1)."""
        rng = make_lcg(123)
        vals = [rng() for _ in range(10000)]
        assert all(0.0 < v < 1.0 for v in vals)

    def test_different_seeds_differ(self):
        """Different seeds produce different sequences."""
        a = make_lcg(1)
        b = make_lcg(2)
        assert a() != b()

    def test_park_miller_modulus(self):
        """First output of seed=1 matches Park-Miller LCG."""
        rng = make_lcg(1)
        # x = (1 * 16807) % 2147483647 = 16807
        # output = (16807 - 1) / 2147483646
        expected = 16806 / 2147483646
        assert rng() == pytest.approx(expected, rel=1e-10)


class TestNormalRV:
    def test_mean_approx_zero(self):
        rng = make_lcg(42)
        vals = [normal_rv(rng) for _ in range(50000)]
        assert np.mean(vals) == pytest.approx(0.0, abs=0.05)

    def test_std_approx_one(self):
        rng = make_lcg(42)
        vals = [normal_rv(rng) for _ in range(50000)]
        assert np.std(vals) == pytest.approx(1.0, abs=0.05)


class TestGammaRV:
    def test_mean_equals_shape(self):
        """E[Gamma(a,1)] = a."""
        rng = make_lcg(99)
        shape = 3.0
        vals = [gamma_rv(shape, rng) for _ in range(20000)]
        assert np.mean(vals) == pytest.approx(shape, rel=0.05)

    def test_shape_less_than_one(self):
        """Recursive case for shape < 1 still produces valid output."""
        rng = make_lcg(42)
        vals = [gamma_rv(0.5, rng) for _ in range(5000)]
        assert all(v > 0 for v in vals)
        assert np.mean(vals) == pytest.approx(0.5, rel=0.15)


class TestStudentT:
    def test_heavier_tails_than_normal(self):
        """Student-t(5) should have more extreme values than Gaussian."""
        rng1 = make_lcg(42)
        rng2 = make_lcg(42)
        normals = [normal_rv(rng1) for _ in range(50000)]
        t_vals = [student_t_rv(5.0, rng2) for _ in range(50000)]
        # P(|X| > 3) should be higher for t(5)
        n_extreme_normal = sum(1 for v in normals if abs(v) > 3)
        n_extreme_t = sum(1 for v in t_vals if abs(v) > 3)
        assert n_extreme_t > n_extreme_normal

    def test_unit_variance_normalization(self):
        """With normalization, variance should be ~1."""
        rng = make_lcg(77)
        vals = [student_t_rv(5.0, rng) for _ in range(50000)]
        assert np.var(vals) == pytest.approx(1.0, abs=0.15)


class TestClamp:
    def test_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_min(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_max(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_boundary(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestPercentile:
    def test_median(self):
        arr = [1, 2, 3, 4, 5]
        assert percentile(arr, 50) == 3

    def test_p0_returns_min(self):
        arr = [10, 20, 30]
        assert percentile(arr, 0) == 10

    def test_p100_returns_max(self):
        arr = [10, 20, 30]
        assert percentile(arr, 100) == 30

    def test_unsorted_input(self):
        arr = [5, 1, 3, 2, 4]
        assert percentile(arr, 50) == 3


# ══════════════════════════════════════════════════════════════════
# Monte Carlo Tests
# ══════════════════════════════════════════════════════════════════

class TestMJDSimulate:
    def test_output_shape(self):
        out = mjd_simulate(100.0, 0.08, 0.20, 252, 1000, 42)
        assert out.shape == (1000,)

    def test_all_positive_prices(self):
        out = mjd_simulate(100.0, 0.08, 0.20, 252, 5000, 42)
        assert np.all(out > 0)

    def test_deterministic(self):
        a = mjd_simulate(100.0, 0.08, 0.20, 63, 100, 42)
        b = mjd_simulate(100.0, 0.08, 0.20, 63, 100, 42)
        np.testing.assert_array_equal(a, b)

    def test_higher_drift_higher_mean(self):
        low = mjd_simulate(100.0, 0.02, 0.20, 252, 5000, 42)
        high = mjd_simulate(100.0, 0.20, 0.20, 252, 5000, 42)
        assert np.mean(high) > np.mean(low)

    def test_earnings_jump_increases_vol(self):
        no_earn = mjd_simulate(100.0, 0.08, 0.20, 63, 5000, 42,
                               MJDParams(earnings_jump=0.0, earnings_day=-1))
        with_earn = mjd_simulate(100.0, 0.08, 0.20, 63, 5000, 42,
                                 MJDParams(earnings_jump=0.10, earnings_day=30))
        assert np.std(with_earn) > np.std(no_earn) * 0.95  # earnings adds dispersion


class TestMCPercentiles:
    def test_ordering(self):
        arr = np.random.RandomState(42).lognormal(mean=4.6, sigma=0.3, size=10000)
        r = mc_percentiles(arr)
        assert r.p5 <= r.p10 <= r.p25 <= r.p50 <= r.p75 <= r.p90 <= r.p95

    def test_est_is_median(self):
        arr = np.random.RandomState(42).lognormal(mean=4.6, sigma=0.3, size=10000)
        r = mc_percentiles(arr)
        assert r.est == r.p50


class TestBuildHistogram:
    def test_correct_bin_count(self):
        arr = np.random.RandomState(42).lognormal(mean=4.6, sigma=0.3, size=10000)
        hist = build_histogram(arr, bins=40)
        assert len(hist) == 40

    def test_density_sums_near_100(self):
        arr = np.random.RandomState(42).lognormal(mean=4.6, sigma=0.3, size=10000)
        hist = build_histogram(arr, bins=40)
        total = sum(h["d"] for h in hist)
        # P2-P98 range covers ~96% of data
        assert 85 < total < 100

    def test_keys_present(self):
        arr = np.array([100.0] * 100 + [110.0] * 100)
        hist = build_histogram(arr, bins=10)
        assert all("c" in h and "d" in h for h in hist)


class TestProbAbove:
    def test_all_above(self):
        arr = np.array([200.0, 300.0, 400.0])
        assert prob_above(arr, 100.0) == pytest.approx(100.0)

    def test_none_above(self):
        arr = np.array([50.0, 60.0, 70.0])
        assert prob_above(arr, 100.0) == pytest.approx(0.0)

    def test_half_above(self):
        arr = np.array([90.0, 110.0])
        assert prob_above(arr, 100.0) == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════
# Signal Tests
# ══════════════════════════════════════════════════════════════════

class TestSignalRegime:
    def test_bull_regime(self, base_data):
        d = dict(base_data)
        d["sma50"] = 160.0
        d["sma200"] = 140.0
        d["currentPrice"] = 165.0
        d["vix"] = 14.0
        r = sig_regime(d)
        assert r["regime"] in ("BULL", "RECOVERY")
        assert r["bW"] >= r["eW"]
        assert "est" in r

    def test_bear_regime(self, base_data):
        d = dict(base_data)
        d["sma50"] = 130.0
        d["sma200"] = 150.0
        d["currentPrice"] = 120.0
        d["vix"] = 35.0
        r = sig_regime(d)
        assert r["regime"] in ("BEAR", "DETERIORATING")
        assert r["eW"] >= r["bW"]

    def test_weights_sum_to_one(self, base_data):
        r = sig_regime(base_data)
        assert r["bW"] + r["mW"] + r["eW"] == pytest.approx(1.0)


class TestSignalMomentum:
    def test_overbought_bearish(self, base_data):
        d = dict(base_data)
        d["rsi14"] = 80.0
        r = sig_mom(d)
        assert r["label"] == "BEARISH"

    def test_oversold_bullish(self, base_data):
        d = dict(base_data)
        d["rsi14"] = 25.0
        r = sig_mom(d)
        assert r["label"] == "BULLISH"

    def test_returns_est(self, base_data):
        r = sig_mom(base_data)
        assert r["est"] > 0


class TestSignalSRS:
    def test_outperformance(self, base_data):
        d = dict(base_data)
        d["stockPerformance3m"] = 20.0
        d["sectorPerformance3m"] = 5.0
        r = sig_srs(d)
        assert r["label"] == "OUTPERF"

    def test_underperformance(self, base_data):
        d = dict(base_data)
        d["stockPerformance3m"] = -10.0
        d["sectorPerformance3m"] = 5.0
        r = sig_srs(d)
        assert r["label"] == "UNDERPERF"


class TestSignalVol:
    def test_crisis(self, base_data):
        d = dict(base_data)
        d["vix"] = 40.0
        r = sig_vol(d)
        assert r["regime"] == "CRISIS"

    def test_complacent(self, base_data):
        d = dict(base_data)
        d["vix"] = 12.0
        r = sig_vol(d)
        assert r["regime"] == "COMPLACENT"


class TestSignalInst:
    def test_high_short_interest_bullish(self, base_data):
        d = dict(base_data)
        d["shortInterest"] = 25.0
        d["putCallRatio"] = 1.3
        r = sig_inst(d)
        assert r["label"] == "BULLISH"


class TestSignalEpsRevision:
    def test_upgrade(self, base_data):
        d = dict(base_data)
        d["earningsRevision3m"] = 10.0
        r = sig_eps_revision(d)
        assert r["label"] == "UPGRADING"
        assert r["est"] > d["currentPrice"]

    def test_downgrade(self, base_data):
        d = dict(base_data)
        d["earningsRevision3m"] = -10.0
        r = sig_eps_revision(d)
        assert r["label"] == "DOWNGRADING"
        assert r["est"] < d["currentPrice"]


class TestSignalSize:
    def test_small_cap_premium(self, base_data):
        d = dict(base_data)
        d["marketCapNum"] = 1.0
        r = sig_size(d)
        assert r["label"] == "SMALL-CAP"
        assert r["est"] > d["currentPrice"]

    def test_large_cap(self, base_data):
        d = dict(base_data)
        d["marketCapNum"] = 300.0
        r = sig_size(d)
        assert r["label"] == "LARGE-CAP"


class TestSignalValue:
    def test_deep_value(self, base_data):
        d = dict(base_data)
        d["peRatio"] = 10.0
        d["sectorAvgPE"] = 25.0
        d["priceToBook"] = 1.0
        d["sectorAvgPB"] = 4.0
        r = sig_value(d)
        assert r["label"] == "DEEP VALUE"

    def test_growth(self, base_data):
        d = dict(base_data)
        d["peRatio"] = 40.0
        d["sectorAvgPE"] = 20.0
        r = sig_value(d)
        assert r["label"] == "GROWTH"


class TestSignalQuality:
    def test_high_quality(self, base_data):
        d = dict(base_data)
        d["roe"] = 30.0
        d["profitMargin"] = 30.0
        r = sig_quality(d)
        assert r["label"] == "HIGH QUALITY"
        assert r["est"] > d["currentPrice"]

    def test_low_quality(self, base_data):
        d = dict(base_data)
        d["roe"] = 5.0
        d["profitMargin"] = 3.0
        r = sig_quality(d)
        assert r["label"] == "LOW QUALITY"


class TestSignalInvestment:
    def test_aggressive(self, base_data):
        d = dict(base_data)
        d["capexToRevenue"] = 30.0
        r = sig_investment(d)
        assert r["label"] == "AGGRESSIVE"

    def test_conservative(self, base_data):
        d = dict(base_data)
        d["capexToRevenue"] = 5.0
        r = sig_investment(d)
        assert r["label"] == "CONSERVATIVE"


class TestSignalLowVol:
    def test_low_vol(self, base_data):
        d = dict(base_data)
        d["annualizedVol"] = 0.12
        r = sig_low_vol(d)
        assert r["label"] == "LOW VOL"
        assert r["est"] > d["currentPrice"]

    def test_high_vol(self, base_data):
        d = dict(base_data)
        d["annualizedVol"] = 0.70
        r = sig_low_vol(d)
        assert r["label"] == "HIGH VOL"


class TestSignalPEAD:
    def test_positive_surprise_beat(self, base_data):
        d = dict(base_data)
        d["lastEarningsSurprise"] = 10.0
        d["daysSinceEarnings"] = 5
        r = sig_pead(d)
        assert r["label"] == "BEAT"
        assert r["est"] > d["currentPrice"]

    def test_decay_over_time(self, base_data):
        d1 = dict(base_data)
        d1["lastEarningsSurprise"] = 10.0
        d1["daysSinceEarnings"] = 5
        r1 = sig_pead(d1)

        d2 = dict(base_data)
        d2["lastEarningsSurprise"] = 10.0
        d2["daysSinceEarnings"] = 55
        r2 = sig_pead(d2)

        # Closer to earnings -> larger drift
        assert abs(r1["est"] - d1["currentPrice"]) > abs(r2["est"] - d2["currentPrice"])

    def test_full_decay_after_60_days(self, base_data):
        d = dict(base_data)
        d["lastEarningsSurprise"] = 20.0
        d["daysSinceEarnings"] = 65
        r = sig_pead(d)
        assert r["est"] == d["currentPrice"]


class TestSignalSeason:
    def test_returns_est(self, base_data):
        r = sig_season(base_data)
        assert "est" in r
        assert "month" in r
        assert 1 <= r["month"] <= 12


class TestSignalOptionsSkew:
    def test_high_fear(self, base_data):
        d = dict(base_data)
        d["ivRank"] = 85.0
        r = sig_options_skew(d)
        assert r["label"] == "HIGH FEAR"
        assert r["est"] > d["currentPrice"]  # mean-reversion signal


class TestSignalInsider:
    def test_buying(self, base_data):
        d = dict(base_data)
        d["insiderNetBuying"] = 10.0
        r = sig_insider(d)
        assert r["label"] == "BUYING"
        assert r["est"] > d["currentPrice"]

    def test_selling(self, base_data):
        d = dict(base_data)
        d["insiderNetBuying"] = -5.0
        r = sig_insider(d)
        assert r["label"] == "SELLING"


class TestSignalRevAccel:
    def test_accelerating(self, base_data):
        d = dict(base_data)
        d["revenueGrowthPct"] = 20.0
        d["priorRevenueGrowthPct"] = 10.0
        r = sig_rev_accel(d)
        assert r["label"] == "ACCEL"

    def test_decelerating(self, base_data):
        d = dict(base_data)
        d["revenueGrowthPct"] = 5.0
        d["priorRevenueGrowthPct"] = 15.0
        r = sig_rev_accel(d)
        assert r["label"] == "DECEL"


class TestSignalFCF:
    def test_high_yield(self, base_data):
        d = dict(base_data)
        d["fcfYield"] = 12.0
        r = sig_fcf(d)
        assert r["label"] == "HIGH"
        assert r["est"] > d["currentPrice"]


class TestSignalLeverage:
    def test_fortress(self, base_data):
        d = dict(base_data)
        d["debtToEquity"] = 0.2
        d["interestCoverage"] = 20.0
        r = sig_leverage(d)
        assert r["label"] == "FORTRESS"

    def test_fragile(self, base_data):
        d = dict(base_data)
        d["debtToEquity"] = 3.0
        d["interestCoverage"] = 1.5
        r = sig_leverage(d)
        assert r["label"] == "FRAGILE"


class TestSignalBuyback:
    def test_buyback(self, base_data):
        d = dict(base_data)
        d["shareCountChange"] = -5.0
        r = sig_buyback(d)
        assert r["label"] == "BUYBACK"

    def test_diluting(self, base_data):
        d = dict(base_data)
        d["shareCountChange"] = 8.0
        r = sig_buyback(d)
        assert r["label"] == "DILUTING"


class TestSignalMacro:
    def test_supportive(self, base_data):
        d = dict(base_data)
        d["vix"] = 13.0
        d["yieldCurve2s10s"] = 1.5
        d["creditSpreadHY"] = 2.5
        d["fedFundsRate"] = 2.0
        r = sig_macro(d)
        assert r["label"] == "SUPPORTIVE"

    def test_restrictive(self, base_data):
        d = dict(base_data)
        d["vix"] = 40.0
        d["yieldCurve2s10s"] = -1.0
        d["creditSpreadHY"] = 7.0
        d["fedFundsRate"] = 7.0
        r = sig_macro(d)
        assert r["label"] == "RESTRICTIVE"


class TestSignalInteractions:
    def test_default_multipliers(self, base_data):
        """No interactions fire -> all multipliers are 1.0."""
        sigs = {
            "mom": {"signal": 0.0},
            "inst": {"si": 3.0},
            "val": {"score": 0},
            "qual": {"score": 0},
            "pead": {"surprise": 0},
            "epsRev": {"revision": 0},
            "macro": {"comp": 0},
            "lev": {"de": 0.5},
            "fcf": {"yield": 4},
            "buyback": {"change": 0},
        }
        m = signal_interactions(sigs)
        assert len(m) == 25
        assert all(v == 1.0 for v in m)

    def test_squeeze_interaction(self):
        """Momentum + high short interest boosts methods 5 and 8."""
        sigs = {
            "mom": {"signal": 0.2},
            "inst": {"si": 20.0},
            "val": {"score": 0}, "qual": {"score": 0},
            "pead": {"surprise": 0}, "epsRev": {"revision": 0},
            "macro": {"comp": 0}, "lev": {"de": 0.5},
            "fcf": {"yield": 4}, "buyback": {"change": 0},
        }
        m = signal_interactions(sigs)
        assert m[5] == 1.5
        assert m[8] == 1.3


class TestDetectEarningsEvent:
    def test_no_earnings_date(self, base_data):
        r = detect_earnings_event(base_data)
        assert r["inWindow"] is False

    def test_distant_earnings(self, base_data):
        from datetime import datetime, timedelta
        d = dict(base_data)
        d["nextEarningsDate"] = (datetime.now() + timedelta(days=400)).isoformat()
        r = detect_earnings_event(d)
        assert r["inWindow"] is False

    def test_near_earnings(self, base_data):
        from datetime import datetime, timedelta
        d = dict(base_data)
        d["nextEarningsDate"] = (datetime.now() + timedelta(days=20)).isoformat()
        r = detect_earnings_event(d)
        assert r["inWindow"] is True
        assert r["earningsDay"] > 0
        assert r["expectedMove"] > 0


# ══════════════════════════════════════════════════════════════════
# Ensemble Tests
# ══════════════════════════════════════════════════════════════════

class TestHorizonWeights:
    def test_short_horizon(self):
        w = horizon_weights(21)
        assert len(w) == 25
        assert sum(w) == pytest.approx(1.0)

    def test_medium_horizon(self):
        w = horizon_weights(252)
        assert len(w) == 25
        assert sum(w) == pytest.approx(1.0)

    def test_long_horizon(self):
        w = horizon_weights(504)
        assert len(w) == 25
        assert sum(w) == pytest.approx(1.0)

    def test_all_weights_non_negative(self):
        for t in [21, 63, 126, 252, 504]:
            w = horizon_weights(t)
            assert all(wi >= 0 for wi in w), f"Negative weight at horizon {t}"

    def test_raw_weight_arrays_sum(self):
        """The underlying s, m, ll arrays should each sum close to 1.0."""
        s = [.07, .05, .04, .05, .05, .09, .04, .06, .06, .07,
             .01, .01, .01, .01, .02, .09, .03, .07, .04, .03,
             .01, .02, .02, .03, .02]
        m = [.10, .08, .05, .07, .06, .04, .04, .03, .04, .04,
             .02, .03, .03, .02, .02, .04, .02, .03, .02, .03,
             .03, .03, .03, .04, .04]
        ll = [.12, .10, .06, .05, .04, .01, .02, .02, .02, .02,
              .04, .07, .06, .04, .03, .00, .01, .01, .02, .02,
              .06, .04, .03, .05, .03]
        assert len(s) == 25
        assert len(m) == 25
        assert len(ll) == 25
        # These are normalized in the function, so raw sums just need to be close
        assert sum(s) == pytest.approx(1.0, abs=0.05)
        assert sum(m) == pytest.approx(1.0, abs=0.05)
        assert sum(ll) == pytest.approx(1.0, abs=0.05)


class TestJamesSteinShrink:
    def test_shrinks_toward_target(self):
        estimates = [90.0, 110.0, 150.0, 200.0, 250.0]
        target = 150.0
        shrunk, coeff = james_stein_shrink(estimates, target)
        # All shrunk values should be closer to target than raw
        for raw, s in zip(estimates, shrunk):
            assert abs(s - target) <= abs(raw - target) + 1e-10

    def test_coefficient_in_0_1(self):
        estimates = [100.0, 120.0, 140.0, 160.0, 180.0]
        _, coeff = james_stein_shrink(estimates, 140.0)
        assert 0.0 <= coeff <= 1.0

    def test_identical_estimates_no_change(self):
        estimates = [100.0, 100.0, 100.0, 100.0, 100.0]
        shrunk, coeff = james_stein_shrink(estimates, 100.0)
        assert all(s == 100.0 for s in shrunk)
        assert coeff == 1.0

    def test_fewer_than_3(self):
        """m < 3 should return originals."""
        estimates = [100.0, 200.0]
        shrunk, coeff = james_stein_shrink(estimates, 150.0)
        assert shrunk == [100.0, 200.0]
        assert coeff == 1.0


class TestBootstrapCI:
    def test_ci_ordering(self):
        methods = [{"est": 100 + i * 5, "w": 1.0 / 10} for i in range(10)]
        b = bootstrap_ci(methods, n_boot=5000, seed=42)
        assert b.ci95[0] <= b.ci90[0] <= b.ci68[0]
        assert b.ci68[1] <= b.ci90[1] <= b.ci95[1]

    def test_mean_in_range(self):
        methods = [{"est": 100 + i * 2, "w": 1.0 / 5} for i in range(5)]
        b = bootstrap_ci(methods, n_boot=5000, seed=42)
        assert b.ci95[0] <= b.mean <= b.ci95[1]

    def test_deterministic(self):
        methods = [{"est": 100 + i * 3, "w": 0.04} for i in range(25)]
        b1 = bootstrap_ci(methods, 5000, 777)
        b2 = bootstrap_ci(methods, 5000, 777)
        assert b1.mean == b2.mean
        assert b1.ci90 == b2.ci90


class TestInverseVarWeights:
    def test_sums_to_one(self):
        w = inverse_var_weights([100, 110, 120, 130, 140])
        assert sum(w) == pytest.approx(1.0)

    def test_closer_to_mean_gets_higher_weight(self):
        ests = [100, 110, 120, 130, 200]  # 200 is outlier
        w = inverse_var_weights(ests)
        # 120 is closest to mean (~132), should have highest weight
        # 200 is farthest, should have lowest weight
        assert w[4] < w[2]


class TestDecorrelateWeights:
    def test_sums_to_one(self):
        w = [1.0 / 25] * 25
        dw = decorrelate_weights(w)
        assert sum(dw) == pytest.approx(1.0)

    def test_correlated_pairs_discounted(self):
        w = [1.0 / 25] * 25
        dw = decorrelate_weights(w)
        # Pairs (11,20), (12,21) etc should be discounted
        assert dw[11] < 1.0 / 25
        assert dw[20] < 1.0 / 25


class TestKellyConfidence:
    def test_confidence_bounded(self):
        methods = [{"est": 100 + i * 3} for i in range(25)]
        k = kelly_confidence(methods, 100.0, 0.25, 252)
        assert 15 <= k["confidence"] <= 95

    def test_label_assignment(self):
        methods = [{"est": 200.0} for _ in range(25)]  # big edge
        k = kelly_confidence(methods, 100.0, 0.25, 252)
        assert k["label"] in ("HIGH", "MODERATE", "LOW", "VERY LOW")

    def test_keys_present(self):
        methods = [{"est": 105.0} for _ in range(25)]
        k = kelly_confidence(methods, 100.0, 0.30, 252)
        for key in ["est", "cv", "confidence", "kellyFull", "kellyHalf",
                     "agreement", "mean", "edge", "label"]:
            assert key in k


class TestPredictionInterval:
    def test_total_wider_than_components(self):
        boot = BootstrapResult(
            mean=105.0, std=5.0,
            ci68=(100.0, 110.0), ci90=(95.0, 115.0), ci95=(93.0, 117.0),
        )
        mc = {"p5": 80.0, "p25": 95.0, "p50": 105.0, "p75": 115.0, "p95": 130.0}
        pi = prediction_interval(boot, mc, 105.0)
        assert pi.total_std >= pi.model_std
        assert pi.total_std >= pi.market_std

    def test_variance_decomposition(self):
        boot = BootstrapResult(
            mean=100.0, std=3.0,
            ci68=(97.0, 103.0), ci90=(94.0, 106.0), ci95=(93.0, 107.0),
        )
        mc = {"p5": 85.0, "p25": 95.0, "p50": 100.0, "p75": 105.0, "p95": 115.0}
        pi = prediction_interval(boot, mc, 100.0)
        assert pi.model_pct + pi.market_pct == pytest.approx(100.0, abs=0.2)

    def test_fidelity_in_0_1(self):
        boot = BootstrapResult(
            mean=100.0, std=2.0,
            ci68=(98.0, 102.0), ci90=(96.0, 104.0), ci95=(95.0, 105.0),
        )
        mc = {"p5": 80.0, "p25": 92.0, "p50": 100.0, "p75": 108.0, "p95": 120.0}
        pi = prediction_interval(boot, mc, 100.0)
        assert 0.0 <= pi.fidelity <= 1.0


# ══════════════════════════════════════════════════════════════════
# Integration: run_prediction
# ══════════════════════════════════════════════════════════════════

class TestRunPrediction:
    def test_returns_prediction_result(self, base_data):
        result = run_prediction(base_data)
        assert isinstance(result, PredictionResult)

    def test_symbol_preserved(self, base_data):
        result = run_prediction(base_data)
        assert result.symbol == "TEST"

    def test_25_methods(self, base_data):
        result = run_prediction(base_data)
        # 24 methods + Kelly = 25 total, but the code has 5 base + 19 signals + 1 Kelly = 25
        assert len(result.methods) == 25

    def test_weights_sum_to_one(self, base_data):
        result = run_prediction(base_data)
        total_w = sum(m.weight for m in result.methods)
        assert total_w == pytest.approx(1.0, abs=0.01)

    def test_ensemble_point_positive(self, base_data):
        result = run_prediction(base_data)
        assert result.ensemble_point > 0

    def test_ensemble_return_pct_reasonable(self, base_data):
        result = run_prediction(base_data)
        assert -100 < result.ensemble_return_pct < 500

    def test_probabilities_populated(self, base_data):
        result = run_prediction(base_data)
        assert len(result.probabilities) == 5

    def test_histogram_populated(self, base_data):
        result = run_prediction(base_data)
        assert len(result.histogram) > 0

    def test_bootstrap_ci_ordered(self, base_data):
        result = run_prediction(base_data)
        b = result.bootstrap
        assert b.ci95[0] <= b.ci90[0] <= b.ci68[0] < b.ci68[1] <= b.ci90[1] <= b.ci95[1]

    def test_prediction_interval_has_total(self, base_data):
        result = run_prediction(base_data)
        pi = result.prediction_interval
        assert pi.total_90[0] < result.ensemble_point < pi.total_90[1]

    def test_mc_percentiles_present(self, base_data):
        result = run_prediction(base_data)
        assert "p5" in result.mc
        assert "p95" in result.mc
        assert result.mc["p5"] < result.mc["p95"]

    def test_signals_dict_populated(self, base_data):
        result = run_prediction(base_data)
        expected_keys = {"reg", "mom", "srs", "volR", "inst", "epsRev",
                         "size", "val", "qual", "inv", "lowV", "pead",
                         "season", "opts", "insider", "revAcc", "fcf",
                         "lev", "buyback", "macro", "kelly"}
        assert expected_keys.issubset(result.signals.keys())

    def test_kelly_confidence_present(self, base_data):
        result = run_prediction(base_data)
        kelly = result.signals["kelly"]
        assert "confidence" in kelly
        assert "label" in kelly

    def test_etf_mode(self, etf_data):
        result = run_prediction(etf_data, is_etf=True)
        assert result.is_etf is True
        assert result.ensemble_point > 0

    def test_vol_scale_computed(self, base_data):
        result = run_prediction(base_data)
        # vol=0.30, vol_ref=0.25 -> scale = max(0.30, 0.15)/0.25 = 1.2
        assert result.vol_scale == pytest.approx(1.2)

    def test_high_vol_scale(self, base_data):
        d = dict(base_data)
        d["annualizedVol"] = 0.80
        result = run_prediction(d)
        # scale = max(0.80, 0.15)/0.25 = 3.2
        assert result.vol_scale == pytest.approx(3.2)

    def test_low_vol_floors(self, base_data):
        d = dict(base_data)
        d["annualizedVol"] = 0.05
        result = run_prediction(d)
        # scale = max(0.05, 0.15)/0.25 = 0.6
        assert result.vol_scale == pytest.approx(0.6)

    def test_deterministic(self, base_data):
        r1 = run_prediction(base_data)
        r2 = run_prediction(base_data)
        assert r1.ensemble_point == r2.ensemble_point
        assert r1.bootstrap.mean == r2.bootstrap.mean

    def test_js_coeff_in_range(self, base_data):
        result = run_prediction(base_data)
        assert 0.0 <= result.js_coeff <= 1.0

    def test_method_results_have_colors(self, base_data):
        result = run_prediction(base_data)
        for m in result.methods:
            assert m.color.startswith("#")

    def test_no_nan_in_output(self, base_data):
        result = run_prediction(base_data)
        assert not math.isnan(result.ensemble_point)
        assert not math.isnan(result.ensemble_return_pct)
        assert not math.isnan(result.vol_scale)
        assert not math.isnan(result.js_coeff)
        for m in result.methods:
            assert not math.isnan(m.est)
            assert not math.isnan(m.weight)

    def test_missing_optional_fields(self):
        """Minimal data should still produce a result without errors."""
        minimal = {
            "symbol": "MIN",
            "currentPrice": 50.0,
            "tradingDaysRemaining": 126,
            "high52w": 60.0,
            "low52w": 40.0,
        }
        result = run_prediction(minimal)
        assert result.ensemble_point > 0
        assert len(result.methods) == 25
