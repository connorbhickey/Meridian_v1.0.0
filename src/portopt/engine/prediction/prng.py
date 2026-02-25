"""Custom PRNG & distribution sampling — matches JSX implementation exactly.

Uses identical LCG / Box-Muller / Marsaglia-Tsang algorithms so that
the same seed produces the same output cross-platform.

References:
  - Park-Miller LCG: modulus 2^31−1, multiplier 16807
  - Box-Muller 1958: rejection-based normal generation
  - Marsaglia & Tsang 2000: gamma RV via squeeze method
  - Student-t(ν): T = Z / √(χ²/ν), normalized to unit variance
"""

from __future__ import annotations

import math


def make_lcg(seed: int):
    """Create a Park-Miller LCG returning values in (0, 1).

    Matches JSX: ``const lcg = s => { let x = s % 2147483647; ... }``
    """
    x = seed % 2147483647
    if x <= 0:
        x += 2147483646

    def _next() -> float:
        nonlocal x
        x = (x * 16807) % 2147483647
        return (x - 1) / 2147483646

    return _next


def normal_rv(rng) -> float:
    """Box-Muller rejection sampler for N(0,1).

    Matches JSX ``nrv`` — rejection on unit disk, then transform.
    """
    while True:
        u = 2.0 * rng() - 1.0
        v = 2.0 * rng() - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


def gamma_rv(shape: float, rng) -> float:
    """Gamma RV — Marsaglia & Tsang 2000.

    Matches JSX ``gammaRV`` exactly (recursive for shape < 1).
    """
    if shape < 1.0:
        return gamma_rv(shape + 1.0, rng) * (rng() ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    while True:
        while True:
            x = normal_rv(rng)
            v = 1.0 + c * x
            if v > 0.0:
                break
        v = v * v * v
        u = rng()
        if u < 1.0 - 0.0331 * x * x * x * x:
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v


def student_t_rv(nu: float, rng) -> float:
    """Student-t(ν) RV, normalized to unit variance.

    T = Z / √(χ²/ν)  ×  √((ν−2)/ν)

    ν=5: P(|X|>4σ) ≈ 0.5% vs 0.006% Gaussian.
    """
    z = normal_rv(rng)
    chi2 = 2.0 * gamma_rv(nu / 2.0, rng)
    return z / math.sqrt(chi2 / nu) * math.sqrt((nu - 2.0) / nu)


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(hi, val))


def percentile(arr, p: float) -> float:
    """Compute *p*-th percentile (0–100) of a sorted or unsorted array.

    Matches JSX ``pctl``: floor-index into sorted copy.
    """
    s = sorted(arr)
    idx = min(int(p / 100.0 * len(s)), len(s) - 1)
    return s[idx]
