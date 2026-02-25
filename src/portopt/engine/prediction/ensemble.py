"""Ensemble machinery — weights, shrinkage, bootstrap, prediction intervals.

Combines 25 methods via:
  40% horizon-adaptive weights
  30% inverse-variance weights
  30% signal-interaction weights (decorrelated)

References:
  - James & Stein 1961 (shrinkage estimator)
  - Efron 1979 (bootstrap confidence intervals)
  - Kelly 1956 / Thorp 1997 (Kelly criterion)
  - Tukey 1977 (Winsorization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from portopt.engine.prediction.monte_carlo import (
    MJDParams,
    build_histogram,
    mc_percentiles,
    mjd_simulate,
    prob_above,
)
from portopt.engine.prediction.prng import clamp, make_lcg, percentile
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


# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """One method's output inside the ensemble."""
    name: str
    est: float
    color: str
    source: str
    weight: float = 0.0
    vol_scaled: bool = False
    raw_est: float | None = None


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval summary."""
    mean: float = 0.0
    std: float = 0.0
    ci68: tuple[float, float] = (0.0, 0.0)
    ci90: tuple[float, float] = (0.0, 0.0)
    ci95: tuple[float, float] = (0.0, 0.0)


@dataclass
class PredictionInterval:
    """Decomposed 3-tier prediction interval."""
    model_std: float = 0.0
    market_std: float = 0.0
    total_std: float = 0.0
    model_90: tuple[float, float] = (0.0, 0.0)
    market_90: tuple[float, float] = (0.0, 0.0)
    total_90: tuple[float, float] = (0.0, 0.0)
    model_pct: float = 0.0
    market_pct: float = 0.0
    fidelity: float = 0.0


@dataclass
class PredictionResult:
    """Complete output of run_prediction()."""
    symbol: str = ""
    is_etf: bool = False
    vol_scale: float = 1.0

    # Ensemble
    ensemble_point: float = 0.0
    ensemble_return_pct: float = 0.0

    # Components
    methods: list[MethodResult] = field(default_factory=list)
    probabilities: list[dict] = field(default_factory=list)
    pe_scenarios: list[dict] = field(default_factory=list)
    reversion_scenarios: list[dict] = field(default_factory=list)

    # MC distribution
    mc: dict = field(default_factory=dict)
    histogram: list[dict] = field(default_factory=list)
    blend_eps: float = 0.0

    # Earnings event
    earnings: dict = field(default_factory=dict)

    # Statistical
    bootstrap: BootstrapResult = field(default_factory=BootstrapResult)
    prediction_interval: PredictionInterval = field(default_factory=PredictionInterval)
    js_coeff: float = 1.0

    # Analyst
    analyst_data: dict = field(default_factory=dict)

    # All signals
    signals: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Ensemble building blocks
# ──────────────────────────────────────────────────────────────────

def horizon_weights(t: int, is_etf: bool = False) -> list[float]:
    """Horizon-adaptive weight vectors for 25 methods.

    Short (T≤63), Medium (63<T≤252), Long (T>252) interpolation.
    """
    # Short / Medium / Long base weights (25 methods)
    s = [.07, .05, .04, .05, .05, .09, .04, .06, .06, .07,
         .01, .01, .01, .01, .02, .09, .03, .07, .04, .03,
         .01, .02, .02, .03, .02]
    m = [.10, .08, .05, .07, .06, .04, .04, .03, .04, .04,
         .02, .03, .03, .02, .02, .04, .02, .03, .02, .03,
         .03, .03, .03, .04, .04]
    ll = [.12, .10, .06, .05, .04, .01, .02, .02, .02, .02,
          .04, .07, .06, .04, .03, .00, .01, .01, .02, .02,
          .06, .04, .03, .05, .03]

    # Interpolation parameter
    if t <= 63:
        a = 0.0
    elif t <= 252:
        a = (t - 63) / 189
    elif t <= 504:
        a = 1.0 + (t - 252) / 252
    else:
        a = 2.0

    if a <= 1.0:
        w = [s[i] + a * (m[i] - s[i]) for i in range(25)]
    else:
        w = [m[i] + (a - 1) * (ll[i] - m[i]) for i in range(25)]

    total = sum(w)
    return [x / total for x in w]


def james_stein_shrink(
    estimates: list[float], target: float
) -> tuple[list[float], float]:
    """James-Stein shrinkage toward *target*.

    θ̂_JS = ν + max(0, 1 − (m−2)s²/‖y−ν‖²) · (y − ν)

    Returns (shrunk_estimates, shrinkage_coefficient).
    """
    m = len(estimates)
    if m < 3:
        return list(estimates), 1.0

    diffs = [e - target for e in estimates]
    ssq = sum(d * d for d in diffs)
    if ssq == 0:
        return list(estimates), 1.0

    mean = sum(estimates) / m
    s2 = sum((e - mean) ** 2 for e in estimates) / (m - 1)
    coeff = max(0.0, 1.0 - (m - 2) * s2 / ssq)

    shrunk = [target + coeff * d for d in diffs]
    return shrunk, coeff


def bootstrap_ci(
    methods: list[dict], n_boot: int = 5000, seed: int = 777
) -> BootstrapResult:
    """Bootstrap CI using horizon weights (not IVW).

    Each method dict must have 'est' and 'w' keys.
    Uses pre-shrinkage estimates to capture true signal disagreement.
    """
    rng = make_lcg(seed)
    n = len(methods)
    boot_ens = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sum_w = 0.0
        sum_we = 0.0
        for _ in range(n):
            j = int(rng() * n)
            j = min(j, n - 1)  # safety
            sum_w += methods[j]["w"]
            sum_we += methods[j]["w"] * methods[j]["est"]
        boot_ens[b] = sum_we / sum_w if sum_w > 0 else methods[0]["est"]

    sorted_arr = np.sort(boot_ens)

    def p(q):
        idx = min(int(q / 100 * len(sorted_arr)), len(sorted_arr) - 1)
        return round(float(sorted_arr[idx]), 2)

    mean_val = float(np.mean(boot_ens))
    var_val = float(np.var(boot_ens))

    return BootstrapResult(
        mean=round(mean_val, 2),
        std=round(math.sqrt(var_val), 2),
        ci68=(p(16), p(84)),
        ci90=(p(5), p(95)),
        ci95=(p(2.5), p(97.5)),
    )


def inverse_var_weights(estimates: list[float]) -> list[float]:
    """Inverse-variance weighting (for point estimate only)."""
    mean = sum(estimates) / len(estimates)
    min_var = mean * mean * 1e-6
    variances = [max((e - mean) ** 2, min_var) for e in estimates]
    ivs = [1.0 / v for v in variances]
    total = sum(ivs)
    return [iv / total for iv in ivs]


def decorrelate_weights(weights: list[float]) -> list[float]:
    """Discount correlated signal pairs."""
    pairs = [(11, 20, 0.6), (12, 21, 0.5), (9, 15, 0.5),
             (5, 6, 0.4), (7, 14, 0.5), (23, 7, 0.3)]
    adj = list(weights)
    for i, j, rho in pairs:
        if i < len(adj) and j < len(adj):
            d = 1 - rho * 0.5
            adj[i] *= d
            adj[j] *= d
    total = sum(adj)
    return [w / total for w in adj]


def kelly_confidence(
    methods: list[dict], s: float, vol: float, t: int
) -> dict:
    """Kelly criterion confidence and Bayesian calibration.

    f* = edge / σ², where σ² = vol² × T/252 (horizon-adjusted).
    """
    ests = [m["est"] for m in methods]
    mean = sum(ests) / len(ests)
    variance = sum((e - mean) ** 2 for e in ests) / len(ests)
    cv = math.sqrt(variance) / abs(mean) if mean != 0 else 0.0

    edge = mean / s - 1
    sig_sq = vol * vol * max(t, 1) / 252
    kelly_full = edge / sig_sq if sig_sq > 0 else 0.0
    kelly_half = kelly_full * 0.5
    abs_k = abs(kelly_half)

    conf = clamp(abs_k * 200, 15, 95)
    agreement = max(0.2, 1 - cv * 3)
    bayes_est = s * (1 - agreement) + mean * agreement

    return {
        "est": round(bayes_est, 2),
        "cv": round(cv, 4),
        "confidence": round(conf, 1),
        "kellyFull": round(kelly_full * 100, 1),
        "kellyHalf": round(kelly_half * 100, 1),
        "agreement": round(agreement * 100, 1),
        "mean": round(mean, 2),
        "edge": round(edge * 100, 2),
        "label": "HIGH" if conf > 75 else ("MODERATE" if conf > 50 else ("LOW" if conf > 30 else "VERY LOW")),
    }


def prediction_interval(
    boot: BootstrapResult, mc: dict, ens_point: float
) -> PredictionInterval:
    """Decomposed 3-tier prediction interval.

    Total PI = √(model² + market²) per tail (quadrature).
    """
    model_std = boot.std
    mc_p5 = mc.get("p5", ens_point)
    mc_p95 = mc.get("p95", ens_point)
    mc_p25 = mc.get("p25", ens_point)
    mc_p75 = mc.get("p75", ens_point)

    iqr = mc_p75 - mc_p25
    market_std = round(iqr / 1.349, 2) if iqr > 0 else 0.01
    total_std = round(math.sqrt(model_std ** 2 + market_std ** 2), 2)

    # Per-tail quadrature
    b_lo90 = ens_point - boot.ci90[0]
    b_hi90 = boot.ci90[1] - ens_point
    m_lo90 = ens_point - mc_p5
    m_hi90 = mc_p95 - ens_point

    total_lo = round(ens_point - math.sqrt(m_lo90 ** 2 + b_lo90 ** 2), 2)
    total_hi = round(ens_point + math.sqrt(m_hi90 ** 2 + b_hi90 ** 2), 2)

    total_sq = total_std ** 2 if total_std > 0 else 1.0

    return PredictionInterval(
        model_std=round(model_std, 2),
        market_std=market_std,
        total_std=total_std,
        model_90=boot.ci90,
        market_90=(round(mc_p5, 2), round(mc_p95, 2)),
        total_90=(total_lo, total_hi),
        model_pct=round(model_std ** 2 / total_sq * 100, 1),
        market_pct=round(market_std ** 2 / total_sq * 100, 1),
        fidelity=round(model_std / (total_std or 1.0), 4),
    )


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

# Signal colors (match JSX ordering)
_SIGNAL_COLORS = [
    "#F97316", "#14B8A6", "#EF4444", "#A78BFA", "#06B6D4",
    "#84CC16", "#D946EF", "#F43F5E", "#0EA5E9", "#22D3EE",
    "#FB923C", "#A3E635", "#C084FC", "#2DD4BF", "#FBBF24",
    "#34D399", "#F87171", "#818CF8", "#4ADE80",
]

_SIGNAL_NAMES = [
    "Momentum", "Sector RS", "Vol Regime", "Inst. Sentiment",
    "EPS Revision", "Size Factor", "Value Factor", "Quality",
    "Investment", "Low Vol", "PEAD", "Seasonality",
    "Options Skew", "Insider", "Rev Accel", "FCF Yield",
    "Leverage", "Buyback", "Macro Regime",
]


def run_prediction(d: dict, is_etf: bool = False) -> PredictionResult:
    """Run the full 25-method ensemble prediction.

    Args:
        d: Data dictionary with all required fields (see data_provider.py)
        is_etf: True for ETFs (uses yield model instead of earnings)

    Returns:
        Complete PredictionResult with all outputs.
    """
    s = d["currentPrice"]
    vol = d.get("annualizedVol") or 0.3
    beta = d.get("beta") or 1.0
    e1 = d.get("forwardEps")
    e2 = d.get("forwardEps2")
    high52w = d["high52w"]
    low52w = d["low52w"]
    analyst_avg = d.get("analystAvgPt")
    analyst_high = d.get("analystHigh")
    analyst_low = d.get("analystLow")
    recent_analyst = d.get("recentAnalystAvg")
    t = d["tradingDaysRemaining"]
    div_yield = d.get("dividendYield") or 0.0
    expense_ratio = d.get("expenseRatio") or 0.0
    rf = (d.get("fedFundsRate") or 4.5) / 100
    mrp = 0.055
    n_sims = 10000

    # Blended EPS
    if e1 and e2:
        b_eps = 0.42 * e1 + 0.58 * e2
    elif e1:
        b_eps = e1 * 1.05
    else:
        b_eps = 0.0

    # ═══════════════════════════════════════
    # VOL-ADAPTIVE SCALING (v6 core)
    # ═══════════════════════════════════════
    vol_ref = 0.25
    vol_scale = max(vol, 0.15) / vol_ref

    # ── Earnings Event ──
    earnings = detect_earnings_event(d)

    # ── S1: MJD Monte Carlo ──
    reg = sig_regime(d)
    mjd_opts = MJDParams(
        nu=5, lambda_=2, jump_mu=-0.02, jump_sig=0.08,
        earnings_jump=earnings["expectedMove"] if earnings["inWindow"] else 0.0,
        earnings_day=earnings["earningsDay"] if earnings["inWindow"] else -1,
    )
    bull_drift = 0.25 * max(beta, 1)
    bear_drift = -0.10 * max(beta, 1)

    bull = mjd_simulate(s, bull_drift, vol, t, n_sims, 42, mjd_opts)
    base = mjd_simulate(s, rf + beta * mrp, vol, t, n_sims, 137, mjd_opts)
    bear = mjd_simulate(s, bear_drift, vol, t, n_sims, 491, mjd_opts)

    # Regime-weighted blend
    n_b = int(n_sims * reg["bW"])
    n_m = int(n_sims * reg["mW"])
    n_e = int(n_sims * reg["eW"])
    blend = np.concatenate([bull[:n_b], base[:n_m], bear[:n_e]])

    mc_r = mc_percentiles(blend)
    mc_dict = {
        "est": mc_r.est, "mean": mc_r.mean,
        "p5": mc_r.p5, "p10": mc_r.p10, "p25": mc_r.p25,
        "p50": mc_r.p50, "p75": mc_r.p75, "p90": mc_r.p90, "p95": mc_r.p95,
    }

    # ── S2: Earnings / Yield Valuation ──
    if not is_etf and b_eps > 0:
        fp = max(s / e1 if e1 and e1 > 0 else 15, 5)  # forward P/E
        pe_scenarios = [
            {"label": f"Bear({round(fp * 0.7)}x)", "m": round(fp * 0.7), "p": 0.20},
            {"label": f"Base({round(fp)}x)", "m": round(fp), "p": 0.45},
            {"label": f"Bull({round(fp * 1.3)}x)", "m": round(fp * 1.3), "p": 0.25},
            {"label": f"Mega({round(fp * 1.6)}x)", "m": round(fp * 1.6), "p": 0.10},
        ]
        for sc in pe_scenarios:
            sc["price"] = round(b_eps * sc["m"], 2)
        earn_est = round(sum(sc["price"] * sc["p"] for sc in pe_scenarios), 2)
    else:
        dy = (div_yield or 1.5) / 100
        er = (expense_ratio or 0.2) / 100
        ny = dy - er
        t_y = t / 252
        pe_scenarios = [
            {"label": "Bear(5%)", "m": 5, "p": 0.25,
             "price": round(s * (1.05 + ny) ** t_y, 2)},
            {"label": "Base(8%)", "m": 8, "p": 0.45,
             "price": round(s * (1.08 + ny) ** t_y, 2)},
            {"label": "Bull(12%)", "m": 12, "p": 0.20,
             "price": round(s * (1.12 + ny) ** t_y, 2)},
            {"label": "Mega(15%)", "m": 15, "p": 0.10,
             "price": round(s * (1.15 + ny) ** t_y, 2)},
        ]
        earn_est = round(sum(sc["price"] * sc["p"] for sc in pe_scenarios), 2)

    # ── S3: Mean Reversion ──
    mean_target = d.get("sma200") or ((high52w + low52w) / 2)
    rev_scenarios = [
        {"target": round(s * 1.05, 2), "prob": 0.15},
        {"target": round(mean_target, 2), "prob": 0.35},
        {"target": round(low52w + 0.5 * (high52w - low52w), 2), "prob": 0.30},
        {"target": round(low52w + 0.75 * (high52w - low52w), 2), "prob": 0.15},
        {"target": round(high52w * 0.95, 2), "prob": 0.05},
    ]
    rev_est = round(sum(sc["target"] * sc["prob"] for sc in rev_scenarios), 2)

    # ── S4: Analyst Consensus ──
    rec = recent_analyst or (analyst_avg * 0.85 if analyst_avg else s * 1.05)
    broad = analyst_avg or s * 1.08
    an_est = round(rec * 0.6 + broad * 0.4, 2)

    # ── S5-S23: Factor Signals ──
    mom = sig_mom(d)
    srs = sig_srs(d)
    vol_r = sig_vol(d)
    inst = sig_inst(d)
    eps_rev = sig_eps_revision(d)
    size = sig_size(d)
    val = sig_value(d)
    qual = sig_quality(d)
    inv = sig_investment(d)
    low_v = sig_low_vol(d)
    pead = sig_pead(d)
    season = sig_season(d)
    opts = sig_options_skew(d)
    insider = sig_insider(d)
    rev_acc = sig_rev_accel(d)
    fcf = sig_fcf(d)
    lev = sig_leverage(d)
    buyback = sig_buyback(d)
    macro = sig_macro(d)

    signal_results = [mom, srs, vol_r, inst, eps_rev, size, val, qual,
                      inv, low_v, pead, season, opts, insider, rev_acc,
                      fcf, lev, buyback, macro]

    # Vol-scale factor signal estimates
    scaled_signal_ests = []
    for sig in signal_results:
        raw_pct = (sig["est"] - s) / s
        scaled_pct = raw_pct * vol_scale
        scaled_signal_ests.append(round(s * (1 + scaled_pct), 2))

    # Build pre-methods array (25 methods)
    pre_methods = [
        {"n": "MJD Monte Carlo", "est": mc_r.est, "c": "#3B82F6",
         "src": "Merton 1976 + Student-t(ν=5)", "volScaled": False},
        {"n": "ETF Yield & Growth" if is_etf else "Earnings Val",
         "est": earn_est, "c": "#10B981",
         "src": "DCF / Gordon Growth", "volScaled": False},
        {"n": "Mean Reversion", "est": rev_est, "c": "#F59E0B",
         "src": "Ornstein-Uhlenbeck", "volScaled": False},
        {"n": "Analyst Consensus", "est": an_est, "c": "#8B5CF6",
         "src": "Thomson Reuters I/B/E/S", "volScaled": False},
        {"n": "Regime Drift", "est": reg["est"], "c": "#EC4899",
         "src": "HMM proxy", "volScaled": False},
    ]
    for i, name in enumerate(_SIGNAL_NAMES):
        pre_methods.append({
            "n": name,
            "est": scaled_signal_ests[i],
            "c": _SIGNAL_COLORS[i],
            "src": name,
            "volScaled": True,
            "rawEst": signal_results[i]["est"],
        })

    # ── Winsorize at P5/P95 (Tukey 1977) ──
    raw_arr = [m["est"] for m in pre_methods]
    w_lo = percentile(raw_arr, 5)
    w_hi = percentile(raw_arr, 95)
    for m in pre_methods:
        m["est"] = round(clamp(m["est"], w_lo, w_hi), 2)

    # ── Save RAW estimates for bootstrap (before shrinkage) ──
    raw_ests_for_boot = [m["est"] for m in pre_methods]

    # ── James-Stein Shrinkage toward analyst consensus ──
    js_shrunk, js_coeff = james_stein_shrink(
        [m["est"] for m in pre_methods], an_est
    )
    for i, m in enumerate(pre_methods):
        if "rawEst" not in m or m["rawEst"] is None:
            m["rawEst"] = m["est"]
        m["est"] = round(js_shrunk[i], 2)

    # ── Kelly Calibrator (method 25) ──
    kelly = kelly_confidence(pre_methods, s, vol, t)
    all_methods = list(pre_methods) + [
        {"n": "Kelly Calibrator", "est": kelly["est"], "c": "#E879F9",
         "src": "Kelly 1956 / Thorp 1997"},
    ]

    # ── Weight blending: 40% horizon + 30% IVW + 30% interaction ──
    all_ests = [m["est"] for m in all_methods]
    horiz_w = horizon_weights(t, is_etf)
    iv_w = inverse_var_weights(all_ests)
    inter_mults = signal_interactions({
        "mom": mom, "inst": inst, "val": val, "qual": qual,
        "pead": pead, "epsRev": eps_rev, "macro": macro,
        "lev": lev, "fcf": fcf, "buyback": buyback,
    })
    inter_w = [horiz_w[i] * (inter_mults[i] if i < len(inter_mults) else 1.0)
               for i in range(25)]
    i_sum = sum(inter_w)
    inter_wn = [x / i_sum for x in inter_w] if i_sum > 0 else horiz_w

    blend_w = [horiz_w[i] * 0.40 + iv_w[i] * 0.30 + inter_wn[i] * 0.30
               for i in range(25)]
    final_w = decorrelate_weights(blend_w)

    for i, m in enumerate(all_methods):
        m["w"] = final_w[i]

    # ── Ensemble Point Estimate ──
    ens = round(sum(m["est"] * m["w"] for m in all_methods), 2)

    # ── Bootstrap CI on RAW estimates with horizon weights ──
    raw_for_boot = [
        {"est": raw_ests_for_boot[i], "w": horiz_w[i]}
        for i in range(len(raw_ests_for_boot))
    ]
    raw_for_boot.append({
        "est": kelly["mean"],
        "w": horiz_w[-1] if horiz_w else 0.04,
    })
    boot = bootstrap_ci(raw_for_boot, 5000, 777)

    # ── Prediction Interval ──
    pi = prediction_interval(boot, mc_dict, ens)

    # ── Probability Table ──
    probs = [
        {"label": f"P(>${round(s * 1.25)})", "value": prob_above(blend, s * 1.25)},
        {"label": f"P(>${round(s * 1.5)})", "value": prob_above(blend, s * 1.5)},
        {"label": f"P(>${round(s * 2)})", "value": prob_above(blend, s * 2)},
        {"label": f"P(<${round(s * 0.66)})",
         "value": round(100 - prob_above(blend, s * 0.66), 1)},
        {"label": f"P(<${round(s)})",
         "value": round(100 - prob_above(blend, s), 1)},
    ]

    # ── Build MethodResult objects ──
    method_results = [
        MethodResult(
            name=m["n"], est=m["est"], color=m["c"],
            source=m.get("src", ""), weight=m.get("w", 0),
            vol_scaled=m.get("volScaled", False),
            raw_est=m.get("rawEst"),
        )
        for m in all_methods
    ]

    return PredictionResult(
        symbol=d.get("symbol", ""),
        is_etf=is_etf,
        vol_scale=round(vol_scale, 2),
        ensemble_point=ens,
        ensemble_return_pct=round((ens / s - 1) * 100, 1),
        methods=method_results,
        probabilities=probs,
        pe_scenarios=pe_scenarios,
        reversion_scenarios=rev_scenarios,
        mc=mc_dict,
        histogram=build_histogram(blend),
        blend_eps=round(b_eps, 2),
        earnings=earnings,
        bootstrap=boot,
        prediction_interval=pi,
        js_coeff=round(js_coeff, 4),
        analyst_data={
            "broad": round(broad, 2),
            "recent": round(rec, 2),
            "weighted": an_est,
            "high": analyst_high or round(broad * 1.3, 2),
            "low": analyst_low or round(broad * 0.7, 2),
        },
        signals={
            "reg": reg, "mom": mom, "srs": srs, "volR": vol_r,
            "inst": inst, "epsRev": eps_rev, "size": size, "val": val,
            "qual": qual, "inv": inv, "lowV": low_v, "pead": pead,
            "season": season, "opts": opts, "insider": insider,
            "revAcc": rev_acc, "fcf": fcf, "lev": lev,
            "buyback": buyback, "macro": macro, "kelly": kelly,
        },
    )
