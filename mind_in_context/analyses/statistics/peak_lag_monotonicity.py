"""Statistical test for Fig 5 - does the peak lag of the brain×LLM RSA
curve increase monotonically with context length?

For each ROI:
  - load results/rsa_by_context/<roi>_<model>.npz
  - extract one peak lag per context: argmax of `curves_peak_norm` inside
    the same zoom window used by fig5 (LANG 200-400 ms, AUD 25-225 ms);
    contexts whose peak falls on the mask edge are dropped (identical
    rule to fig5._draw_zoom - those curves actually peak outside the zoom)
  - observed = Spearman ρ(context_words, peak_lags)
  - "cheap" permutation null: shuffle peak_lags across contexts N times,
    recompute ρ each time
  - report directional p (P(null ≥ obs), since the claim is "more
    context -> later peak") and Kendall's τ as a non-parametric companion

Saves: results/peak_lag_monotonicity/<roi>_<model>.npz

With --layer L the peak lags are instead read from the layer scan
results/rsa_peak_by_layer/<roi>_<model>.npz (the file behind Fig. 5),
contexts 2-20 with the full-transcript point excluded, and the result is
saved as results/peak_lag_monotonicity/<roi>_<model>_layer<L>.npz. This is
the test quoted in the main text for layer 25.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import kendalltau, spearmanr

from mind_in_context.lib.io import load_result, make_argparser, save_result

ZOOM = {"lang": (200, 400), "aud": (25, 225)}
N_PERM = 10_000
RNG_SEED = 0


def _peak_lags(d, zoom: tuple[int, int]):
    lags = np.asarray(d["eval_lags_ms"])
    curves = np.asarray(d["curves_peak_norm"])
    ctx_words = np.asarray(d["context_words"], dtype=np.float64)

    mask = (lags >= zoom[0]) & (lags <= zoom[1])
    lags_z = lags[mask]
    n_in = int(mask.sum())

    ctxs, peaks = [], []
    dropped = []
    for ci in range(curves.shape[0]):
        if ctx_words[ci] <= 3:
            dropped.append((ctx_words[ci], "ctx<=3 excluded"))
            continue
        c = curves[ci][mask]
        if not np.any(np.isfinite(c)):
            dropped.append((ctx_words[ci], "all-nan"))
            continue
        pi = int(np.nanargmax(c))
        if pi == 0 or pi == n_in - 1:
            dropped.append((ctx_words[ci], f"edge_idx={pi}"))
            continue
        ctxs.append(ctx_words[ci])
        peaks.append(float(lags_z[pi]))
    return np.array(ctxs), np.array(peaks), dropped


def _peak_lags_from_layer_scan(roi: str, model: str, layer: int):
    """Peak lags of one layer from the Fig. 5 layer scan (full context excluded)."""
    d = load_result(f"rsa_peak_by_layer/{roi}_{model}")
    layers = np.asarray(d["layers"], dtype=int)
    if layer not in layers:
        raise SystemExit(f"layer {layer} not in the scan: {layers.tolist()}")
    row = int(np.where(layers == layer)[0][0])
    keep = ~np.asarray(d["is_full"], dtype=bool)
    ctx = np.asarray(d["context_words"], dtype=np.float64)[keep]
    peaks = np.asarray(d["peak_lags_ms"], dtype=np.float64)[row][keep]
    return ctx, peaks, [(int(w), "full transcript excluded")
                        for w in np.asarray(d["context_words"])[~keep]]


def _run(roi: str, model: str, layer: int | None = None) -> None:
    if layer is None:
        d = load_result(f"rsa_by_context/{roi}_{model}")
        ctx, peaks, dropped = _peak_lags(d, ZOOM[roi])
        tag = ""
    else:
        ctx, peaks, dropped = _peak_lags_from_layer_scan(roi, model, layer)
        tag = f"_layer{layer}"
    print(f"\n=== {roi.upper()} ({model}{tag}) ===")
    print(f"  zoom window: {ZOOM[roi]} ms")
    print(f"  contexts kept: {len(ctx)}   dropped: {len(dropped)}")
    for w, why in dropped:
        print(f"    drop ctx={w}  reason={why}")
    order = np.argsort(ctx)
    print("  (ctx_words, peak_lag_ms):")
    for w, p in zip(ctx[order], peaks[order]):
        print(f"    ctx={int(w):>5}   peak={p:6.1f} ms")

    obs_rho, obs_p_analytic = spearmanr(ctx, peaks)
    tau, tau_p = kendalltau(ctx, peaks)

    rng = np.random.default_rng(RNG_SEED)
    null = np.empty(N_PERM, dtype=np.float64)
    for i in range(N_PERM):
        null[i] = spearmanr(ctx, rng.permutation(peaks))[0]
    p_perm_one  = (np.sum(null >= obs_rho) + 1) / (N_PERM + 1)
    p_perm_two  = (np.sum(np.abs(null) >= abs(obs_rho)) + 1) / (N_PERM + 1)

    print(f"  observed Spearman rho = {obs_rho:+.4f}   (analytic p = {obs_p_analytic:.3g})")
    print(f"  observed Kendall  tau = {tau:+.4f}    (analytic p = {tau_p:.3g})")
    print(f"  permutation (N={N_PERM}, seed={RNG_SEED}):")
    print(f"    one-sided p (null >= obs)        = {p_perm_one:.4g}")
    print(f"    two-sided p (|null| >= |obs|)    = {p_perm_two:.4g}")

    out = save_result(
        f"peak_lag_monotonicity/{roi}_{model}{tag}",
        context_words=ctx.astype(np.float64),
        peak_lags_ms=peaks.astype(np.float64),
        observed_spearman=np.float64(obs_rho),
        observed_spearman_p_analytic=np.float64(obs_p_analytic),
        observed_kendall=np.float64(tau),
        observed_kendall_p_analytic=np.float64(tau_p),
        null_spearman=null,
        p_perm_one_sided=np.float64(p_perm_one),
        p_perm_two_sided=np.float64(p_perm_two),
        n_perm=np.int64(N_PERM),
        rng_seed=np.int64(RNG_SEED),
        zoom_ms=np.array(ZOOM[roi], dtype=np.float64),
    )
    print(f"  -> {out}")


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    parser.add_argument("--rois", nargs="+", default=["lang", "aud"],
                        choices=["lang", "aud"])
    parser.add_argument("--layer", type=int, default=None,
                        help="Take peak lags from the Fig. 5 layer scan for "
                             "this layer instead of rsa_by_context.")
    args = parser.parse_args()
    for roi in args.rois:
        _run(roi, args.model, args.layer)


if __name__ == "__main__":
    main()
