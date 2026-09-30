"""Statistical test for Fig 6B - do deeper LLM layers have steeper
peak-lag-vs-context curves (i.e. saturate later)?

Claim in the text: "earlier layers in the model appeared to saturate at
shorter contexts - reflected in flatter slopes in early layers compared to
deeper layers." We quantify this by, for each scanned layer, fitting a
slope to its peak-lag(context) curve and then correlating that slope with
layer depth.

Pipeline (per ROI):
  - load results/rsa_peak_by_layer/<roi>_<model>.npz
    (peak_lags_ms is (n_layers, n_ctx); contexts include word counts plus
     the 'full' transcript, flagged by is_full)
  - restrict to the dense word-count contexts (2..MAX_CTX words; the 'full'
    point is dropped - it is an outlier on the x-axis and not part of the
    smooth buildup curve)
  - per layer, fit a slope two ways:
        slope_lin  = OLS slope of peak_lag vs context (words)
        slope_log  = OLS slope of peak_lag vs ln(context)
    slope_log is the headline number: the curves saturate, and Fig 5 colours
    context on a log scale, so peak_lag is ~linear in ln(context).
  - correlate slope vs layer number (Pearson r + p, Spearman ρ as a
    non-parametric companion)

Saves: results/layer_slope_correlation/<roi>_<model>.npz

CLI:  --rois {lang,aud}...  --model {llama3,mistral7b}  --max-ctx N
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from mind_in_context.lib.io import load_result, make_argparser, save_result

MAX_CTX = 20  # dense word-count grid; 'full' transcript is always excluded


def _layer_slopes(d, max_ctx: int):
    """Return (layers, ctx_words, slope_lin, slope_log, peak_lags_kept)."""
    layers = np.asarray(d["layers"], dtype=np.int64)
    ctx_words = np.asarray(d["context_words"], dtype=np.float64)
    is_full = np.asarray(d["is_full"], dtype=bool)
    peak = np.asarray(d["peak_lags_ms"], dtype=np.float64)  # (n_layers, n_ctx)

    keep = (~is_full) & (ctx_words >= 2) & (ctx_words <= max_ctx)
    x = ctx_words[keep]
    order = np.argsort(x)
    x = x[order]
    lx = np.log(x)

    slope_lin = np.empty(len(layers), dtype=np.float64)
    slope_log = np.empty(len(layers), dtype=np.float64)
    peak_kept = np.empty((len(layers), x.size), dtype=np.float64)
    for i in range(len(layers)):
        y = peak[i][keep][order]
        peak_kept[i] = y
        slope_lin[i] = np.polyfit(x, y, 1)[0]
        slope_log[i] = np.polyfit(lx, y, 1)[0]
    return layers, x, slope_lin, slope_log, peak_kept


def _run(roi: str, model: str, max_ctx: int) -> None:
    d = load_result(f"rsa_peak_by_layer/{roi}_{model}")
    layers, ctx, slope_lin, slope_log, _ = _layer_slopes(d, max_ctx)
    L = layers.astype(np.float64)

    print(f"\n=== {roi.upper()} ({model}) ===")
    print(f"  contexts used: {ctx.astype(int).tolist()}  ('full' excluded)")
    print(f"  {'layer':>5}  {'slope_lin(ms/word)':>18}  {'slope_log(ms/ln-word)':>22}")
    for l, a, b in zip(layers.tolist(), slope_lin.tolist(), slope_log.tolist()):
        print(f"  {l:>5}  {a:>18.3f}  {b:>22.3f}")

    # Correlate slope vs layer depth.
    r_lin, p_lin = pearsonr(L, slope_lin)
    rho_lin, prho_lin = spearmanr(L, slope_lin)
    r_log, p_log = pearsonr(L, slope_log)
    rho_log, prho_log = spearmanr(L, slope_log)

    print(f"\n  slope_log vs layer:  Pearson r = {r_log:+.4f} (p = {p_log:.4g})"
          f"   Spearman ρ = {rho_log:+.4f} (p = {prho_log:.4g})")
    print(f"  slope_lin vs layer:  Pearson r = {r_lin:+.4f} (p = {p_lin:.4g})"
          f"   Spearman ρ = {rho_lin:+.4f} (p = {prho_lin:.4g})")
    print(f"  (n = {len(layers)} layers: {layers.tolist()})")

    out = save_result(
        f"layer_slope_correlation/{roi}_{model}",
        layers=layers,
        context_words=ctx.astype(np.float64),
        slope_lin=slope_lin,
        slope_log=slope_log,
        pearson_r_log=np.float64(r_log),
        pearson_p_log=np.float64(p_log),
        spearman_rho_log=np.float64(rho_log),
        spearman_p_log=np.float64(prho_log),
        pearson_r_lin=np.float64(r_lin),
        pearson_p_lin=np.float64(p_lin),
        spearman_rho_lin=np.float64(rho_lin),
        spearman_p_lin=np.float64(prho_lin),
        max_ctx=np.int64(max_ctx),
    )
    print(f"  -> {out}")


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    parser.add_argument("--rois", nargs="+", default=["lang", "aud"],
                        choices=["lang", "aud"])
    parser.add_argument("--max-ctx", type=int, default=MAX_CTX)
    args = parser.parse_args()
    for roi in args.rois:
        try:
            _run(roi, args.model, args.max_ctx)
        except FileNotFoundError:
            print(f"\n=== {roi.upper()} ({args.model}) ===")
            print(f"  [skip] results/rsa_peak_by_layer/{roi}_{args.model}.npz "
                  f"not found - run analyses/rsa_peak_by_layer.py first.")


if __name__ == "__main__":
    main()
