"""Signal functions S1–S20 for the 25-method ensemble.

Each signal function takes a data dict and returns a dict with at
minimum an ``est`` key (price estimate) plus metadata.

Vol-adaptive scaling is applied LATER in the ensemble orchestrator
(§5), not here. These are raw, pre-scaling estimates.

References:
  - Fama & French 1993/2015 (size, value, quality, investment)
  - Ball & Brown 1968 (PEAD)
  - Adrian & Shin 2010 (macro regime)
"""

from __future__ import annotations

import math
from datetime import datetime

from portopt.engine.prediction.prng import clamp


# ── S1: Regime Detection (HMM-proxy) ──────────────────────────────

def sig_regime(d: dict) -> dict:
    """Detect market regime from technicals + VIX.

    Returns regime label, component score, regime weights (bW/mW/eW),
    drift estimate, and price target.
    """
    s = d["currentPrice"]
    sma50 = d.get("sma50")
    sma200 = d.get("sma200")
    high52w = d["high52w"]
    low52w = d["low52w"]
    vix = d.get("vix") or 18.0

    # SMA cross score
    sma = 0.0
    if sma50 and sma200:
        if sma50 > sma200:
            sma = min((sma50 / sma200 - 1) * 10, 1.0)
        else:
            sma = max((sma50 / sma200 - 1) * 10, -1.0)

    # Price position
    pp = 0.0
    if sma50 and sma200:
        if s > sma50 and s > sma200:
            pp = 1.0
        elif s > sma200:
            pp = 0.3
        elif s > sma50:
            pp = -0.2
        else:
            pp = -0.8

    # Range score
    rng = high52w - low52w
    pct = (s - low52w) / rng if rng > 0 else 0.5
    rs = (pct - 0.5) * 2

    # VIX premium
    if vix > 30:
        vp = -0.3
    elif vix > 25:
        vp = -0.15
    elif vix > 20:
        vp = -0.05
    else:
        vp = 0.05

    comp = sma * 0.3 + pp * 0.3 + rs * 0.25 + vp * 0.15

    if comp > 0.35:
        reg, bw, mw, ew = "BULL", 0.55, 0.35, 0.10
    elif comp > 0.1:
        reg, bw, mw, ew = "RECOVERY", 0.40, 0.40, 0.20
    elif comp > -0.1:
        reg, bw, mw, ew = "NEUTRAL", 0.30, 0.40, 0.30
    elif comp > -0.35:
        reg, bw, mw, ew = "DETERIORATING", 0.15, 0.35, 0.50
    else:
        reg, bw, mw, ew = "BEAR", 0.10, 0.25, 0.65

    rf = (d.get("fedFundsRate") or 4.5) / 100
    beta = d.get("beta", 1.0) or 1.0
    drift = (
        bw * 0.25 * max(beta, 1)
        + mw * (rf + beta * 0.055)
        + ew * (-0.10 * max(beta, 1))
    )
    t_ratio = d["tradingDaysRemaining"] / 252
    est = round(s * math.exp(drift * t_ratio), 2)

    return {
        "regime": reg, "comp": round(comp, 3),
        "bW": bw, "mW": mw, "eW": ew, "drift": drift, "est": est,
    }


# ── S2: Momentum / Reversion ─────────────────────────────────────

def sig_mom(d: dict) -> dict:
    s = d["currentPrice"]
    sma50 = d.get("sma50")
    rsi = d.get("rsi14") or 50.0

    if rsi > 70:
        rs = -0.3
    elif rsi > 60:
        rs = 0.15
    elif rsi > 40:
        rs = 0.0
    elif rsi > 30:
        rs = -0.1
    else:
        rs = 0.25

    m50 = (s / sma50 - 1) if sma50 else 0.0
    ms = -m50 * 0.5 if abs(m50) > 0.2 else m50 * 0.8
    sig = rs * 0.45 + ms * 0.55

    return {
        "est": round(s * (1 + sig * 0.15), 2),
        "signal": round(sig, 3),
        "rsi": rsi,
        "label": "BULLISH" if sig > 0.1 else ("BEARISH" if sig < -0.1 else "NEUTRAL"),
    }


# ── S3: Sector Relative Strength ─────────────────────────────────

def sig_srs(d: dict) -> dict:
    sp = d.get("stockPerformance3m") or 0.0
    sec = d.get("sectorPerformance3m") or 0.0
    spread = sp - sec

    if spread > 15:
        sig = -0.06
    elif spread > 5:
        sig = 0.02
    elif spread > -5:
        sig = 0.0
    elif spread > -15:
        sig = 0.04
    else:
        sig = 0.08

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "spread": round(spread, 1),
        "label": "OUTPERF" if spread > 5 else ("UNDERPERF" if spread < -5 else "INLINE"),
    }


# ── S4: Vol Regime ────────────────────────────────────────────────

def sig_vol(d: dict) -> dict:
    v = d.get("vix") or 18.0
    b = d.get("beta") or 1.0

    if v > 35:
        sk = -0.06
    elif v > 28:
        sk = -0.04
    elif v > 22:
        sk = -0.02
    elif v > 15:
        sk = 0.01
    else:
        sk = 0.03

    adj = sk * (1 + (b - 1) * 0.3)
    return {
        "est": round(d["currentPrice"] * (1 + adj), 2),
        "skew": round(adj * 100, 2),
        "regime": "CRISIS" if v > 28 else ("ELEVATED" if v > 22 else ("NORMAL" if v > 15 else "COMPLACENT")),
        "vix": v,
    }


# ── S5: Institutional Sentiment ──────────────────────────────────

def sig_inst(d: dict) -> dict:
    si = d.get("shortInterest") or 3.0
    pcr = d.get("putCallRatio") or 0.7
    io = d.get("institutionalOwnership") or 65.0

    s1 = 0.08 if si > 20 else (0.03 if si > 10 else (0.0 if si > 5 else -0.01))
    s2 = 0.05 if pcr > 1.2 else (0.02 if pcr > 0.9 else (0.0 if pcr > 0.6 else -0.03))
    s3 = 0.01 if io > 85 else (0.005 if io > 60 else -0.01)

    c = s1 * 0.4 + s2 * 0.35 + s3 * 0.25
    return {
        "est": round(d["currentPrice"] * (1 + c), 2),
        "comp": round(c * 100, 2), "si": si, "pcr": pcr, "io": io,
        "label": "BULLISH" if c > 0.02 else ("BEARISH" if c < -0.01 else "NEUTRAL"),
    }


# ── S6: EPS Revision Momentum ────────────────────────────────────

def sig_eps_revision(d: dict) -> dict:
    rev = d.get("earningsRevision3m") or 0.0
    sig = clamp(rev / 20, -0.15, 0.15)
    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "revision": round(rev, 1),
        "label": "UPGRADING" if rev > 5 else ("DOWNGRADING" if rev < -5 else "STABLE"),
    }


# ── S7: Size Factor (Fama-French SMB) ────────────────────────────

def sig_size(d: dict) -> dict:
    cap = d.get("marketCapNum") or 50.0  # in billions
    if cap < 2:
        prem = 0.04
    elif cap < 10:
        prem = 0.025
    elif cap < 50:
        prem = 0.01
    elif cap < 200:
        prem = 0.0
    else:
        prem = -0.005

    t_ratio = d["tradingDaysRemaining"] / 252
    return {
        "est": round(d["currentPrice"] * math.exp(prem * t_ratio), 2),
        "capB": round(cap, 1),
        "prem": round(prem * 100, 2),
        "label": "SMALL-CAP" if cap < 10 else ("MID-CAP" if cap < 50 else "LARGE-CAP"),
    }


# ── S8: Value Factor (HML) ───────────────────────────────────────

def sig_value(d: dict) -> dict:
    pe = d.get("peRatio") or 20.0
    sec_pe = d.get("sectorAvgPE") or 20.0
    pb = d.get("priceToBook") or 3.0
    sec_pb = d.get("sectorAvgPB") or 3.0

    pe_disc = (pe / sec_pe - 1) if sec_pe > 0 else 0.0
    pb_disc = (pb / sec_pb - 1) if sec_pb > 0 else 0.0
    val_score = -(pe_disc * 0.5 + pb_disc * 0.5)
    sig = clamp(val_score * 0.08, -0.06, 0.06)

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "pe": pe, "secPe": sec_pe,
        "score": round(val_score * 100, 1),
        "label": "DEEP VALUE" if val_score > 0.15 else ("VALUE" if val_score > 0 else "GROWTH"),
    }


# ── S9: Quality / Profitability (RMW) ────────────────────────────

def sig_quality(d: dict) -> dict:
    roe = d.get("roe") or 15.0
    margin = d.get("profitMargin") or 10.0

    q = 0.0
    if roe > 25:
        q += 0.03
    elif roe > 15:
        q += 0.015
    elif roe > 8:
        q += 0.0
    else:
        q -= 0.02

    if margin > 25:
        q += 0.02
    elif margin > 12:
        q += 0.01
    elif margin > 5:
        q += 0.0
    else:
        q -= 0.015

    return {
        "est": round(d["currentPrice"] * (1 + q), 2),
        "roe": round(roe, 1), "margin": round(margin, 1),
        "score": round(q * 100, 2),
        "label": "HIGH QUALITY" if q > 0.03 else ("ADEQUATE" if q > 0 else "LOW QUALITY"),
    }


# ── S10: Investment Factor (CMA) ─────────────────────────────────

def sig_investment(d: dict) -> dict:
    capex = d.get("capexToRevenue") or 8.0

    if capex > 25:
        sig = -0.025
    elif capex > 15:
        sig = -0.01
    elif capex > 8:
        sig = 0.005
    elif capex > 3:
        sig = 0.015
    else:
        sig = 0.02

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "capex": round(capex, 1),
        "label": "AGGRESSIVE" if capex > 15 else "CONSERVATIVE",
    }


# ── S11: Low Volatility Anomaly (BAB) ────────────────────────────

def sig_low_vol(d: dict) -> dict:
    vol = d.get("annualizedVol") or 0.3

    if vol < 0.15:
        sig = 0.02
    elif vol < 0.25:
        sig = 0.01
    elif vol < 0.40:
        sig = 0.0
    elif vol < 0.60:
        sig = -0.015
    else:
        sig = -0.03

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "vol": round(vol * 100, 0),
        "label": "LOW VOL" if vol < 0.25 else ("AVERAGE" if vol < 0.45 else "HIGH VOL"),
    }


# ── S12: PEAD — Ball & Brown 1968 ────────────────────────────────

def sig_pead(d: dict) -> dict:
    surprise = d.get("lastEarningsSurprise") or 0.0
    days = d.get("daysSinceEarnings") or 45

    decay = max(0.0, 1 - days / 60)
    drift_pct = clamp(surprise * 0.003 * decay, -0.08, 0.08)

    return {
        "est": round(d["currentPrice"] * (1 + drift_pct), 2),
        "surprise": round(surprise, 1), "days": days,
        "label": "BEAT" if surprise > 5 else "INLINE",
    }


# ── S13: Seasonality ─────────────────────────────────────────────

def sig_season(d: dict) -> dict:
    month = datetime.now().month
    premia = {
        1: 0.02, 2: 0.005, 3: 0.01, 4: 0.015, 5: -0.005, 6: -0.01,
        7: 0.005, 8: -0.01, 9: -0.015, 10: 0.005, 11: 0.015, 12: 0.02,
    }
    return {
        "est": round(d["currentPrice"] * (1 + premia.get(month, 0.0)), 2),
        "month": month,
    }


# ── S14: Options / IV Skew ───────────────────────────────────────

def sig_options_skew(d: dict) -> dict:
    iv_rank = d.get("ivRank") or 50.0
    skew = d.get("ivSkew") or 0.0

    if iv_rank > 80:
        sig = 0.03
    elif iv_rank > 60:
        sig = 0.01
    elif iv_rank > 40:
        sig = 0.0
    elif iv_rank > 20:
        sig = -0.005
    else:
        sig = -0.02

    sig += clamp(skew * 0.01, -0.02, 0.02)

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "ivRank": iv_rank,
        "label": "HIGH FEAR" if iv_rank > 70 else "NORMAL",
    }


# ── S15: Insider Activity ────────────────────────────────────────

def sig_insider(d: dict) -> dict:
    net = d.get("insiderNetBuying") or 0.0

    if net > 5:
        sig = 0.04
    elif net > 1:
        sig = 0.02
    elif net > -0.5:
        sig = 0.0
    elif net > -3:
        sig = -0.01
    else:
        sig = -0.025

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "net": round(net, 1),
        "label": "BUYING" if net > 1 else ("SELLING" if net < -1 else "NEUTRAL"),
    }


# ── S16: Revenue Acceleration ────────────────────────────────────

def sig_rev_accel(d: dict) -> dict:
    g1 = d.get("revenueGrowthPct") or 8.0
    g0 = d.get("priorRevenueGrowthPct") or 8.0
    accel = g1 - g0
    sig = clamp(accel * 0.003, -0.04, 0.04)

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "accel": round(accel, 1),
        "label": "ACCEL" if accel > 3 else ("DECEL" if accel < -3 else "STABLE"),
    }


# ── S17: FCF Yield ───────────────────────────────────────────────

def sig_fcf(d: dict) -> dict:
    fcfy = d.get("fcfYield") or 4.0

    if fcfy > 10:
        sig = 0.04
    elif fcfy > 6:
        sig = 0.02
    elif fcfy > 3:
        sig = 0.005
    elif fcfy > 0:
        sig = -0.005
    else:
        sig = -0.03

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "yield": round(fcfy, 1),
        "label": "HIGH" if fcfy > 6 else "FAIR",
    }


# ── S18: Leverage / Health ───────────────────────────────────────

def sig_leverage(d: dict) -> dict:
    de = d.get("debtToEquity") or 0.8
    ic = d.get("interestCoverage") or 10.0

    sig = 0.0
    if de < 0.3:
        sig += 0.015
    elif de < 0.8:
        sig += 0.005
    elif de < 2:
        sig -= 0.005
    else:
        sig -= 0.02

    if ic > 15:
        sig += 0.01
    elif ic > 5:
        sig += 0.005
    elif ic > 2:
        sig -= 0.005
    else:
        sig -= 0.02

    label = "FORTRESS" if de < 0.5 and ic > 10 else ("FRAGILE" if de > 2 or ic < 3 else "MODERATE")

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "de": round(de, 2), "ic": round(ic, 1),
        "label": label,
    }


# ── S19: Buyback Signal ──────────────────────────────────────────

def sig_buyback(d: dict) -> dict:
    sc = d.get("shareCountChange") or 0.0

    if sc < -5:
        sig = 0.035
    elif sc < -2:
        sig = 0.02
    elif sc < 1:
        sig = 0.005
    elif sc < 5:
        sig = -0.01
    else:
        sig = -0.03

    return {
        "est": round(d["currentPrice"] * (1 + sig), 2),
        "change": round(sc, 1),
        "label": "BUYBACK" if sc < -2 else ("DILUTING" if sc > 2 else "STABLE"),
    }


# ── S20: Macro Regime — Adrian & Shin 2010 ───────────────────────

def sig_macro(d: dict) -> dict:
    vix = d.get("vix") or 18.0
    ffr = d.get("fedFundsRate") or 4.5
    yc = d.get("yieldCurve2s10s") or 0.2
    cs = d.get("creditSpreadHY") or 3.5

    if vix > 35:
        vs = -0.04
    elif vix > 28:
        vs = -0.025
    elif vix > 22:
        vs = -0.01
    elif vix > 15:
        vs = 0.01
    else:
        vs = 0.02

    if yc < -0.5:
        ycs = -0.04
    elif yc < 0:
        ycs = -0.02
    elif yc < 0.5:
        ycs = 0.0
    elif yc < 1.5:
        ycs = 0.01
    else:
        ycs = 0.015

    if cs > 6:
        css = -0.04
    elif cs > 5:
        css = -0.02
    elif cs > 4:
        css = -0.01
    elif cs > 3:
        css = 0.005
    else:
        css = 0.015

    if ffr > 6:
        fs = -0.02
    elif ffr > 5:
        fs = -0.01
    elif ffr > 3:
        fs = 0.0
    elif ffr > 1:
        fs = 0.01
    else:
        fs = 0.02

    comp = vs * 0.3 + ycs * 0.3 + css * 0.2 + fs * 0.2

    return {
        "est": round(d["currentPrice"] * (1 + comp), 2),
        "comp": round(comp * 100, 2),
        "label": "SUPPORTIVE" if comp > 0.01 else ("RESTRICTIVE" if comp < -0.015 else "NEUTRAL"),
    }


# ── Signal Interactions (multiplicative weight boosts) ────────────

def signal_interactions(sig: dict) -> list[float]:
    """Compute interaction multipliers for 25 methods.

    Returns a list of 25 floats (default 1.0, boosted where interactions fire).
    """
    m = [1.0] * 25

    # Momentum × Short Interest → squeeze potential
    if sig["mom"].get("signal", 0) > 0.1 and (sig["inst"].get("si") or 3) > 15:
        m[5] *= 1.5
        m[8] *= 1.3

    # Value × Quality → Buffett factor (threshold=2, NOT 200)
    if sig["val"].get("score", 0) > 0 and sig["qual"].get("score", 0) > 2:
        m[11] *= 1.4
        m[12] *= 1.4

    # PEAD × EPS Revision → confirmation
    if sig["pead"].get("surprise", 0) > 5 and sig["epsRev"].get("revision", 0) > 3:
        m[15] *= 1.5
        m[9] *= 1.3

    # Macro tight × High leverage → amplified downside
    if sig["macro"].get("comp", 0) < -1.5 and (sig["lev"].get("de") or 0.8) > 2:
        m[21] *= 1.5
        m[23] *= 1.3

    # FCF strong × Buyback → capital return
    if (sig["fcf"].get("yield") or 4) > 6 and (sig["buyback"].get("change") or 0) < -2:
        m[20] *= 1.3
        m[22] *= 1.3

    return m


# ── Earnings Event Detection ─────────────────────────────────────

def detect_earnings_event(d: dict) -> dict:
    """Detect if an earnings event falls within the prediction horizon."""
    ned = d.get("nextEarningsDate")
    if not ned:
        return {"inWindow": False, "earningsDay": -1, "expectedMove": 0.0}

    try:
        if isinstance(ned, str):
            e_date = datetime.fromisoformat(ned)
        else:
            e_date = ned
        now = datetime.now()
        cal_days = (e_date - now).days
        trad_days = round(cal_days * 252 / 365)
        if 0 < trad_days < d.get("tradingDaysRemaining", 252):
            expected = (d.get("historicalEarningsMoveAvg") or 6) / 100
            return {
                "inWindow": True,
                "earningsDay": trad_days,
                "expectedMove": expected,
            }
    except Exception:
        pass

    return {"inWindow": False, "earningsDay": -1, "expectedMove": 0.0}
