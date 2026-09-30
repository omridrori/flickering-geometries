"""Leave-one-patient-out robustness for the three headline timing statistics.

Coverage is very uneven: two patients hold 63% of the 73 language-area
contacts, so every reported latency has to be shown not to rest on either of
them. The pipeline is rerun once per held-out patient, plus once on the full
ROI as a validation fold, and the range across folds is what gets reported.

Each statistic mirrors the settings of the analysis that produced the published
number, so the "all" fold reproduces it exactly:

  ERG dip        analyses/erg.py --full: lags -500..1000 step 25, +-50 ms
                 window, delta = cfg.shared.erg_delta_ms (100 ms, the smallest
                 gap with non-overlapping activation windows), winsorize, no
                 baseline, 25 ms cache. Dip = argmin over 50..400 ms.

  Brain-LLM      analyses/brain_llm_rsa.py: lags -4000..4000 step 25, +-100 ms
  peak           window, winsorize, no baseline, full-context
                 embeddings, +-50 ms lag smoothing. Only the all-electrode
                 curve is needed, so the low-audio and shifted controls are
                 skipped.

  Context        analyses/rsa_by_context.py: contexts 2..20, +-100 ms window,
  monotonicity   no winsorize, no baseline, 1 ms cache, +-1 ms lag smoothing.
                 Lags are restricted to statistics/peak_lag_monotonicity.py's
                 own read-out window (ZOOM), so the argmax is unchanged while
                 costing a fifth of the lags.

With --lag-step 1 the two latencies are re-measured off the 1 ms cache, since a
25 ms grid cannot tell a genuinely stable latency from one that wanders inside
a bin. The context sweep is already 1 ms and is skipped in that mode.

Output: results/leave_one_patient_out/<roi>.npz  (or <roi>_step1ms.npz)
Used by: figures/figS_lopo.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import (BaselineExtractor, _global_electrode_keys,
                                      roi_indices)
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import (compute_rdm_condensed_gpu,
                                     lag_window_average, rank_centered_gpu,
                                     spearman, spearman_from_ranked,
                                     spearman_two_ranked)

ERG_HALF_WINDOW_MS = 50
ERG_DIP_SEARCH = (50, 400)

RSA_HALF_WINDOW_MS = 100
RSA_SMOOTH_HALF_WINDOW_MS = 50

CTX_LAYER = 15
CTX_WORDS = list(range(2, 21))
CTX_LAGS = np.arange(200, 401, 1, dtype=int)      # peak_lag_monotonicity ZOOM
CTX_HALF_WINDOW_MS = 100
CTX_SMOOTH_HALF_WINDOW_MS = 1

# Coarse grid (published) and the narrowed windows used at 1 ms.
GRID = {25: {"erg": (-500, 1000), "rsa": (-4000, 4000)},
        1: {"erg": (50, 400), "rsa": (100, 650)}}
RSA_READ_1MS = (150, 600)


def _folds(roi_key):
    """[(name, electrode_indices, n_dropped)] - full ROI first, then one per patient."""
    roi = roi_indices(roi_key)
    keys = _global_electrode_keys()
    subs = np.array([keys[i][0] for i in roi])
    counts = Counter(subs.tolist())
    out = [("all", roi, 0)]
    for sub, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append((f"sub-{sub}", roi[subs != sub], int(counts[sub])))
    return out, counts


def _erg(idx, lags, cache, delta_ms, bar):
    s = load_config()["shared"]
    ext = BaselineExtractor(electrode_indices=idx, eval_lags_ms=lags,
                            half_window_ms=ERG_HALF_WINDOW_MS,
                            winsorize_sd=s["winsorize_sd"],
                            baseline_window_ms=None, delta_ms=delta_ms,
                            cache=cache)
    words = np.arange(ext.n_words)
    rho = np.full(len(lags), np.nan)
    for t, lag in enumerate(lags):
        rho[t] = spearman(ext.rdm(int(lag), words),
                          ext.rdm(int(lag) + delta_ms, words))
        bar.update(1)
    return rho


def _rsa(idx, lags, cache, llm_ranked, smooth_pts, bar):
    s = load_config()["shared"]
    ext = BaselineExtractor(electrode_indices=idx, eval_lags_ms=lags,
                            half_window_ms=RSA_HALF_WINDOW_MS,
                            winsorize_sd=s["winsorize_sd"],
                            baseline_window_ms=None, cache=cache)
    raw = np.full(len(lags), np.nan)
    for t, lag in enumerate(lags):
        raw[t] = spearman_from_ranked(llm_ranked, ext.rdm(int(lag)))
        bar.update(1)
    return lag_window_average(raw, smooth_pts)


def _ctx(idx, ctx_ranked, bar):
    ext = BaselineExtractor(electrode_indices=idx, eval_lags_ms=CTX_LAGS,
                            half_window_ms=CTX_HALF_WINDOW_MS,
                            winsorize_sd=None, baseline_window_ms=None,
                            cache="1ms")
    curves = np.full((len(CTX_WORDS), len(CTX_LAGS)), np.nan)
    for k, lag in enumerate(CTX_LAGS):
        rb = rank_centered_gpu(compute_rdm_condensed_gpu(ext.activity(int(lag))))
        for c, ctx in enumerate(CTX_WORDS):
            curves[c, k] = spearman_two_ranked(rb, ctx_ranked[ctx])
        del rb
        bar.update(1)
    peaks = np.array([CTX_LAGS[int(np.nanargmax(
        lag_window_average(curves[c], CTX_SMOOTH_HALF_WINDOW_MS)))]
        for c in range(len(CTX_WORDS))], dtype=int)
    r, p = spearmanr(CTX_WORDS, peaks)
    return peaks, float(r), float(p)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--lag-step", type=int, default=25, choices=(1, 25),
                        help="Lag spacing for the two latencies. 1 re-measures "
                             "them off the 1 ms cache and skips the context "
                             "sweep, which is already 1 ms. Default 25, "
                             "matching the published analyses.")
    args = parser.parse_args()

    step = args.lag_step
    erg_delta = int(load_config()["shared"]["erg_delta_ms"])
    cache = "1ms" if step < 25 else "25ms"
    fine = step < 25
    erg_lags = np.arange(*GRID[step]["erg"], step, dtype=int)
    erg_lags = np.append(erg_lags, GRID[step]["erg"][1]).astype(int)
    rsa_lags = np.arange(GRID[step]["rsa"][0], GRID[step]["rsa"][1] + 1, step,
                         dtype=int)
    smooth_pts = max(1, RSA_SMOOTH_HALF_WINDOW_MS // step)

    folds, counts = _folds(args.roi)
    n_roi = len(folds[0][1])
    print(f"  ROI: {args.roi}, model: {args.model}, lag step {step} ms "
          f"(cache {cache}), {n_roi} contacts from {len(counts)} patients")
    print("  " + "  ".join(f"sub-{s}:{k}" for s, k in
                           sorted(counts.items(), key=lambda kv: -kv[1])))

    layer = load_config()["models"][args.model]["layer"]
    emb = np.load(llm.full_context_embeddings_path(args.model, layer))
    llm_ranked = rank_centered_gpu(llm.cosine_rdm(emb.astype(np.float64)))
    del emb
    ctx_ranked = {}
    if not fine:
        for c in tqdm(CTX_WORDS, desc="    context RDMs", leave=False):
            ctx_ranked[c] = rank_centered_gpu(
                llm.cosine_rdm_cached(args.model, CTX_LAYER, c))

    per_fold = len(erg_lags) + len(rsa_lags) + (0 if fine else len(CTX_LAGS))
    names, n_elec, n_drop = [], [], []
    erg_curves, rsa_curves, ctx_peaks, ctx_rho, ctx_p = [], [], [], [], []

    outer = tqdm(folds, desc="  folds", position=0)
    for name, idx, dropped in outer:
        outer.set_description(f"  fold {name}")
        bar = tqdm(total=per_fold, desc=f"    {name} ({len(idx)} contacts)",
                   position=1, leave=False, unit="lag")
        bar.set_postfix_str("ERG")
        erg = _erg(idx, erg_lags, cache, erg_delta, bar)
        bar.set_postfix_str("brain-LLM")
        rsa = _rsa(idx, rsa_lags, cache, llm_ranked, smooth_pts, bar)
        if not fine:
            bar.set_postfix_str("context")
            pk, r, p = _ctx(idx, ctx_ranked, bar)
            ctx_peaks.append(pk)
            ctx_rho.append(r)
            ctx_p.append(p)
        bar.close()

        names.append(name)
        n_elec.append(len(idx))
        n_drop.append(dropped)
        erg_curves.append(erg)
        rsa_curves.append(rsa)

        m = (erg_lags >= ERG_DIP_SEARCH[0]) & (erg_lags <= ERG_DIP_SEARCH[1])
        dip = int(erg_lags[m][int(np.nanargmin(erg[m]))])
        if fine:
            w = (rsa_lags >= RSA_READ_1MS[0]) & (rsa_lags <= RSA_READ_1MS[1])
        else:
            w = np.ones(len(rsa_lags), dtype=bool)
        peak = int(rsa_lags[w][int(np.nanargmax(rsa[w]))])
        tail = "" if fine else f"  ctx rho {ctx_rho[-1]:+.3f}"
        outer.write(f"  {name:<9} {len(idx):>3} contacts  ERG dip {dip:>4} ms"
                    f"  brain-LLM peak {peak:>4} ms{tail}")
    outer.close()

    payload = dict(
        fold_names=np.array(names),
        n_electrodes=np.array(n_elec, dtype=np.int64),
        n_dropped=np.array(n_drop, dtype=np.int64),
        erg_lags_ms=erg_lags,
        erg_curves=np.array(erg_curves),
        erg_dip_search=np.array(ERG_DIP_SEARCH, dtype=np.int64),
        erg_delta_ms=np.int64(erg_delta),
        erg_half_window_ms=np.int64(ERG_HALF_WINDOW_MS),
        rsa_lags_ms=rsa_lags,
        rsa_curves=np.array(rsa_curves),
        rsa_half_window_ms=np.int64(RSA_HALF_WINDOW_MS),
        rsa_smooth_half_window_ms=np.int64(RSA_SMOOTH_HALF_WINDOW_MS),
        rsa_read_window=np.array(RSA_READ_1MS if fine
                                 else (rsa_lags[0], rsa_lags[-1]), dtype=np.int64),
        lag_step_ms=np.int64(step),
        layer=np.int64(layer),
    )
    if not fine:
        payload.update(
            context_words=np.array(CTX_WORDS, dtype=np.int64),
            ctx_lags_ms=CTX_LAGS,
            ctx_peaks=np.array(ctx_peaks, dtype=np.int64),
            ctx_rho=np.array(ctx_rho),
            ctx_p=np.array(ctx_p),
            ctx_layer=np.int64(CTX_LAYER),
        )

    suffix = f"{args.roi}_step1ms" if fine else args.roi
    out = save_result(f"leave_one_patient_out/{suffix}", **payload)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
