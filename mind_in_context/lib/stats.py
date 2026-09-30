"""Lightweight stats helpers used by analyses + figures (no matplotlib here)."""

from __future__ import annotations

import numpy as np


def per_segment_stats(lags: np.ndarray, curves: np.ndarray,
                      lag_min: int, lag_max: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment mean / within-segment std / slope over lags ∈ [lag_min, lag_max].

    `curves` shape: (n_segments, n_lags). Slope is the linear-fit gradient in
    units-per-millisecond (multiply by 1000 to get units-per-second).
    """
    mask = (lags >= lag_min) & (lags <= lag_max)
    x = lags[mask].astype(float)
    means = np.array([np.mean(c[mask]) for c in curves])
    stds = np.array([np.std(c[mask], ddof=1) for c in curves])
    slopes = np.array([np.polyfit(x, c[mask], 1)[0] for c in curves])
    return means, stds, slopes


def normalize_to_0_100(values: np.ndarray,
                       lo: float | None = None,
                       hi: float | None = None
                       ) -> tuple[np.ndarray, float, float]:
    """Linearly rescale `values` so [lo, hi] -> [0, 100]."""
    lo = float(np.min(values)) if lo is None else float(lo)
    hi = float(np.max(values)) if hi is None else float(hi)
    span = max(hi - lo, 1e-12)
    out = (np.clip(values, lo, hi) - lo) / span * 100.0
    return out, lo, hi


def trend_perm_test(values: np.ndarray, n_perm: int = 10_000, seed: int = 0
                    ) -> tuple[float, float, float]:
    """Pearson r of `values` against their order, with a shuffle null.

    Returns (r, p_two_sided, p_one_sided_in_sign_of_r). Same test as the
    per-segment trend statistics of Fig. 2: the values are permuted across
    positions n_perm times and p = (#null at least as extreme + 1) / (n_perm + 1).
    """
    values = np.asarray(values, dtype=float)
    x = np.arange(values.size, dtype=float)
    rng = np.random.default_rng(seed)
    r = float(np.corrcoef(x, values)[0, 1])
    null = np.array([np.corrcoef(x, rng.permutation(values))[0, 1]
                     for _ in range(n_perm)])
    p_two = float((np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1))
    p_one = float((np.sum(null >= r) + 1) / (n_perm + 1)) if r >= 0 else \
            float((np.sum(null <= r) + 1) / (n_perm + 1))
    return r, p_two, p_one
