"""Brain × LLM dynamic RSA per context length - Fig 5.

For each context length k in cfg.shared.context_lengths (2..20 + "full"):
  - Load the model embeddings for that window
    (`<data dir>/embeddings/<model>/context_windows/...window_<K>...`).
  - Build LLM RDM via lib.llm.cosine_rdm (column-center +
    row L2-normalize + cosine pdist).
  - At every lag in [-1000, +1000] ms compute Spearman ρ between the
    brain RDM (±100 ms half-window, no baseline) and the LLM RDM.
  - Smooth the curve along the lag axis with ±50 ms half-window.
  - Peak-normalize (each curve / its own max).

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
Output: results/rsa_by_context/<roi>_<model>.npz
Used by: figures/fig5.py, figures/fig6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import (
    compute_rdm_condensed_gpu, lag_window_average,
    rank_centered_gpu, spearman_two_ranked,
)

LAG_START_MS = -1000
LAG_END_MS = 1000
LAG_STEP_MS = 1                 # 1 ms, so the peak lag is not quantised by the step
HALF_WINDOW_MS = 100            # window over which brain activity is averaged
SMOOTH_HALF_WINDOW_MS = 50      # lag-axis smoothing applied to the curves


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--contexts", nargs="+", default=None,
                        help="Override context list (ints + 'full'). "
                             "Default: cfg.shared.context_lengths.")
    parser.add_argument("--lag-start", type=int, default=LAG_START_MS)
    parser.add_argument("--lag-end", type=int, default=LAG_END_MS)
    parser.add_argument("--lag-step", type=int, default=LAG_STEP_MS,
                        help="Spacing between evaluation lags in ms. "
                             "Default 1. < 25 forces 1 ms cache.")
    parser.add_argument("--smooth-half-window-ms", type=int, default=SMOOTH_HALF_WINDOW_MS,
                        help="Lag-axis smoothing half-window in ms (default 50; "
                             "applied to the curves AFTER computation).")
    parser.add_argument("--force", action="store_true",
                        help="Discard existing results and recompute from scratch.")
    args = parser.parse_args()

    s = load_config()["shared"]
    eval_lags = np.arange(args.lag_start, args.lag_end + 1, args.lag_step, dtype=int)
    cache_resolution = "1ms" if args.lag_step < 25 else "25ms"
    layer = load_config()["models"][args.model]["layer"]
    if args.contexts is not None:
        requested = [c if c == "full" else int(c) for c in args.contexts]
    else:
        requested = list(s["context_lengths"])

    # ---------- Incremental load: skip already-computed contexts.
    from mind_in_context.lib.io import RESULTS_DIR
    out_path = RESULTS_DIR / f"rsa_by_context/{args.roi}_{args.model}.npz"
    existing = None
    if out_path.exists() and not args.force:
        try:
            tmp = np.load(out_path)
            if np.array_equal(tmp["eval_lags_ms"], eval_lags):
                existing = tmp
                done_words = set(tmp["context_words"].tolist())
                done_full = bool(tmp["is_full"].any())
                contexts = [c for c in requested
                            if (c == "full" and not done_full) or
                            (c != "full" and int(c) not in done_words)]
                print(f"  [incremental] {out_path.name} exists with same lag "
                      f"range; reusing {len(tmp['context_words'])} cached "
                      f"contexts, computing {len(contexts)} new.")
            else:
                print(f"  [warn] {out_path.name} exists but lag range differs; "
                      f"recomputing all.")
                contexts = requested
        except Exception:
            contexts = requested
    else:
        contexts = requested
    n_ctx = len(contexts)

    print(f"  ROI: {args.roi}, model: {args.model}, layer {layer}, "
          f"{n_ctx} contexts, lags [{int(eval_lags[0])}, {int(eval_lags[-1])}] ms "
          f"(step {args.lag_step} ms, {len(eval_lags)} pts, cache={cache_resolution})")

    # No winsorising in the brain-LLM comparison: pass None.
    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=None,
        baseline_window_ms=None,
        cache=cache_resolution,
    )
    print(f"  electrodes: {extractor.n_electrodes}, words: {extractor.n_words}")

    n_lags = len(eval_lags)
    curves_raw = np.full((n_ctx, n_lags), np.nan)

    # Pre-rank all LLM RDMs on GPU (n_contexts × ~50 MB ≈ 1 GB - manageable).
    # Then stream brain RDMs one lag at a time so we never hold more than one
    # 50 MB brain tensor on device simultaneously.
    print("  pre-loading & ranking LLM RDMs ...")
    llm_ranked_all: list = []
    valid_ci: list[int] = []
    for ci, ctx in enumerate(tqdm(contexts, desc="    LLM", leave=False)):
        try:
            llm_rdm_vec = llm.cosine_rdm_cached(args.model, layer, ctx)
        except FileNotFoundError as e:
            tqdm.write(f"  [skip] context {ctx}: missing {Path(str(e)).name}")
            continue
        llm_ranked_all.append(rank_centered_gpu(llm_rdm_vec))
        valid_ci.append(ci)

    smooth_half_pts_live = max(1, args.smooth_half_window_ms // args.lag_step)
    for li, lag in enumerate(tqdm(eval_lags, desc="    lags")):
        act = extractor.activity(int(lag))
        brain_rdm_gpu = compute_rdm_condensed_gpu(act)
        brain_ranked = rank_centered_gpu(brain_rdm_gpu)
        del brain_rdm_gpu          # free the raw RDM immediately
        for idx, ci in enumerate(valid_ci):
            curves_raw[ci, li] = spearman_two_ranked(brain_ranked, llm_ranked_all[idx])
        del brain_ranked            # free before next lag

    # Live peak-lag report per context (after full computation).
    for idx, ci in enumerate(valid_ci):
        ctx = contexts[ci]
        smoothed = lag_window_average(curves_raw[ci], smooth_half_pts_live)
        if np.any(np.isfinite(smoothed)):
            peak_idx = int(np.nanargmax(smoothed))
            peak_lag = int(eval_lags[peak_idx])
            peak_val = float(smoothed[peak_idx])
            print(f"    ctx={ctx!s:>4}  peak_lag={peak_lag:>5} ms   "
                  f"max_rho_smoothed={peak_val:+.4f}")

    # Numeric context lengths for the freshly-computed batch.
    new_words = np.array([5136 if c == "full" else int(c)
                          for c in contexts], dtype=np.int64)
    new_is_full = np.array([c == "full" for c in contexts], dtype=np.bool_)

    # Merge with existing if any.
    if existing is not None and len(contexts) > 0:
        all_words = np.concatenate([existing["context_words"], new_words])
        all_is_full = np.concatenate([existing["is_full"], new_is_full])
        all_curves = np.concatenate([existing["curves_raw"], curves_raw], axis=0)
    elif existing is not None:
        all_words = existing["context_words"]
        all_is_full = existing["is_full"]
        all_curves = existing["curves_raw"]
    else:
        all_words = new_words
        all_is_full = new_is_full
        all_curves = curves_raw

    # Sort by ascending context size (full last).
    order = np.argsort([w + 1_000_000 if f else w
                        for w, f in zip(all_words, all_is_full)])
    all_words = all_words[order]
    all_is_full = all_is_full[order]
    all_curves = all_curves[order]

    # Recompute smoothing + peak-normalisation on the merged set
    # (both are per-curve so this is cheap).
    smooth_half_pts = max(1, args.smooth_half_window_ms // args.lag_step)
    curves_smoothed = np.array([lag_window_average(c, smooth_half_pts)
                                for c in all_curves])
    peaks = np.nanmax(curves_smoothed, axis=1, keepdims=True)
    peaks = np.where(peaks > 0, peaks, np.nan)
    curves_peak_norm = curves_smoothed / peaks

    out = save_result(
        f"rsa_by_context/{args.roi}_{args.model}",
        eval_lags_ms=eval_lags,
        context_words=all_words,
        is_full=all_is_full,
        curves_raw=all_curves,
        curves_smoothed=curves_smoothed,
        curves_peak_norm=curves_peak_norm,
        peak_lags_ms=np.array([
            int(eval_lags[np.nanargmax(c)]) if np.any(np.isfinite(c)) else 0
            for c in curves_smoothed
        ], dtype=np.int64),
        layer=np.int64(layer),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(args.smooth_half_window_ms),
    )
    print(f"  -> {out}  (total contexts: {len(all_words)})")


if __name__ == "__main__":
    main()
