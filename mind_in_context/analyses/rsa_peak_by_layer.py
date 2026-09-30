"""Peak-lag vs context-length across LLM layers - Fig 6B.

Same brain×LLM RSA pipeline as rsa_by_context.py, but iterating over
multiple LLM layers (config: shared.layers_for_layer_scan). Brain RDMs
are pre-ranked once (they don't depend on layer); LLM RDMs come from
`lib.llm.cosine_rdm_cached(model, layer, ctx)`.

If a layer matches the model's primary layer in config, peak lags can be
copied from results/rsa_by_context/<roi>_<model>.npz instead of recomputed
(matched on lag range + smooth half-window + context list).

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
Output: results/rsa_peak_by_layer/<roi>_<model>.npz
Used by: figures/fig6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (
    RESULTS_DIR, load_config, load_result, make_argparser, save_result,
)
from mind_in_context.lib.rdm import (
    compute_rdm_condensed_gpu, lag_window_average,
    rank_centered_gpu, spearman_two_ranked,
)

LAG_START_MS = 150
LAG_END_MS = 450
LAG_STEP_MS = 1
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 10


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Override layer list. "
                             "Default: cfg.shared.layers_for_layer_scan.")
    parser.add_argument("--contexts", nargs="+", default=None,
                        help="Override context list (ints + 'full'). "
                             "Default: cfg.shared.context_lengths.")
    parser.add_argument("--lag-start", type=int, default=LAG_START_MS)
    parser.add_argument("--lag-end", type=int, default=LAG_END_MS)
    parser.add_argument("--lag-step", type=int, default=LAG_STEP_MS)
    parser.add_argument("--smooth-half-window-ms", type=int,
                        default=SMOOTH_HALF_WINDOW_MS)
    parser.add_argument("--force", action="store_true",
                        help="Discard cached results and recompute everything.")
    parser.add_argument("--no-reuse-rsa-by-context", action="store_true",
                        help="Don't pre-fill the primary layer from "
                             "results/rsa_by_context/<roi>_<model>.npz.")
    args = parser.parse_args()

    cfg = load_config()
    s = cfg["shared"]
    primary_layer = int(cfg["models"][args.model]["layer"])

    if args.layers is not None:
        layers = [int(l) for l in args.layers]
    else:
        layers = [int(l) for l in s["layers_for_layer_scan"]]
    if args.contexts is not None:
        contexts = [c if c == "full" else int(c) for c in args.contexts]
    else:
        contexts = list(s["context_lengths"])

    eval_lags = np.arange(args.lag_start, args.lag_end + 1, args.lag_step,
                          dtype=int)
    cache_resolution = "1ms" if args.lag_step < 25 else "25ms"
    smooth_half_pts = max(1, args.smooth_half_window_ms // args.lag_step)

    print(f"  ROI: {args.roi}, model: {args.model}, "
          f"{len(layers)} layers {layers}, {len(contexts)} contexts, "
          f"lags [{int(eval_lags[0])}, {int(eval_lags[-1])}] ms "
          f"(step {args.lag_step} ms, {len(eval_lags)} pts, "
          f"cache={cache_resolution}, smooth±{args.smooth_half_window_ms} ms)")

    # ---------- Try to reuse the primary-layer rsa_by_context result.
    reuse_layer_to_peaks: dict[int, dict] = {}
    if not args.no_reuse_rsa_by_context and primary_layer in layers:
        try:
            d = load_result(f"rsa_by_context/{args.roi}_{args.model}")
            if (np.array_equal(d["eval_lags_ms"], eval_lags)
                    and int(d["smooth_half_window_ms"]) == args.smooth_half_window_ms
                    and int(d["layer"]) == primary_layer):
                reuse_layer_to_peaks[primary_layer] = {
                    "context_words": d["context_words"],
                    "is_full": d["is_full"],
                    "peak_lags_ms": d["peak_lags_ms"],
                    "curves_smoothed": d["curves_smoothed"],
                }
                print(f"  [reuse] layer {primary_layer}: pre-filled from "
                      f"rsa_by_context/{args.roi}_{args.model}.npz")
            else:
                print(f"  [reuse] skipping pre-fill - lag range / smooth / "
                      f"layer mismatch with rsa_by_context.")
        except FileNotFoundError:
            pass

    # ---------- Incremental load: skip already-computed (layer, ctx).
    out_path = RESULTS_DIR / f"rsa_peak_by_layer/{args.roi}_{args.model}.npz"
    existing = None
    if out_path.exists() and not args.force:
        try:
            tmp = np.load(out_path)
            if (np.array_equal(tmp["eval_lags_ms"], eval_lags)
                    and int(tmp["smooth_half_window_ms"]) == args.smooth_half_window_ms):
                existing = {k: tmp[k] for k in tmp.files}
                print(f"  [incremental] {out_path.name} exists with same "
                      f"lag range; reusing "
                      f"{existing['layers'].size} layers × "
                      f"{existing['context_words'].size} contexts.")
        except Exception:
            existing = None

    # ---------- Decide which (layer, ctx) cells need fresh compute.
    n_words_full = 5136
    ctx_words = np.array([n_words_full if c == "full" else int(c)
                          for c in contexts], dtype=np.int64)
    is_full = np.array([c == "full" for c in contexts], dtype=np.bool_)

    n_layers = len(layers)
    n_ctx = len(contexts)
    peak_lags = np.full((n_layers, n_ctx), -10_000, dtype=np.int64)
    curves_smoothed = np.full((n_layers, n_ctx, len(eval_lags)), np.nan)

    # Pre-fill from existing (matched by layer + context_words + is_full).
    if existing is not None:
        for li_old, l_old in enumerate(existing["layers"].tolist()):
            if l_old not in layers:
                continue
            li_new = layers.index(int(l_old))
            for ci_old, (w_old, f_old) in enumerate(zip(
                    existing["context_words"].tolist(),
                    existing["is_full"].tolist())):
                # Match by (context_words, is_full).
                for ci_new in range(n_ctx):
                    if (int(w_old) == int(ctx_words[ci_new])
                            and bool(f_old) == bool(is_full[ci_new])):
                        peak_lags[li_new, ci_new] = int(
                            existing["peak_lags_ms"][li_old, ci_old])
                        curves_smoothed[li_new, ci_new] = \
                            existing["curves_smoothed"][li_old, ci_old]

    # Pre-fill from rsa_by_context (primary layer).
    for layer_pre, d in reuse_layer_to_peaks.items():
        li = layers.index(layer_pre)
        for ci, (w, f) in enumerate(zip(ctx_words.tolist(),
                                         is_full.tolist())):
            for ci_old, (w_old, f_old) in enumerate(zip(
                    d["context_words"].tolist(), d["is_full"].tolist())):
                if int(w_old) == int(w) and bool(f_old) == bool(f):
                    peak_lags[li, ci] = int(d["peak_lags_ms"][ci_old])
                    curves_smoothed[li, ci] = d["curves_smoothed"][ci_old]

    # Cells still needing compute: peak_lags == -10_000.
    todo: list[tuple[int, int]] = []
    for li in range(n_layers):
        for ci in range(n_ctx):
            if peak_lags[li, ci] == -10_000:
                todo.append((li, ci))
    print(f"  cells to compute: {len(todo)} / {n_layers * n_ctx}")

    if todo:
        # Build extractor once - brain RDMs depend on (lag, roi), not layer.
        extractor = BaselineExtractor(
            roi_key=args.roi,
            eval_lags_ms=eval_lags,
            half_window_ms=HALF_WINDOW_MS,
            winsorize_sd=None,
            baseline_window_ms=None,
            cache=cache_resolution,
        )
        print(f"  electrodes: {extractor.n_electrodes}, "
              f"words: {extractor.n_words}")

        # Pre-rank all LLM RDMs on GPU (n_todo × ~50 MB; typically ≤ 6 GB).
        # Then stream brain RDMs one lag at a time so only one 50 MB tensor
        # lives on device simultaneously.
        print("  pre-loading & ranking LLM RDMs ...")
        llm_ranked_cache: dict[tuple[int, int], object] = {}
        skip_cells: set[tuple[int, int]] = set()
        for li_t, ci_t in tqdm(todo, desc="    LLM", leave=False):
            key = (li_t, ci_t)
            if key in llm_ranked_cache or key in skip_cells:
                continue
            layer_v = layers[li_t]
            ctx_v = contexts[ci_t]
            try:
                rdm_vec = llm.cosine_rdm_cached(args.model, layer_v, ctx_v)
                llm_ranked_cache[key] = rank_centered_gpu(rdm_vec)
            except FileNotFoundError as e:
                tqdm.write(f"  [skip] layer={layer_v} ctx={ctx_v}: "
                           f"missing {Path(str(e)).name}")
                skip_cells.add(key)

        active_todo = [(li_t, ci_t) for li_t, ci_t in todo
                       if (li_t, ci_t) not in skip_cells]
        curves_raw_temp: dict[tuple[int, int], np.ndarray] = {
            (li_t, ci_t): np.full(len(eval_lags), np.nan)
            for li_t, ci_t in active_todo
        }

        for lag_i, lag in enumerate(tqdm(eval_lags, desc="    lags")):
            act = extractor.activity(int(lag))
            brain_rdm_gpu = compute_rdm_condensed_gpu(act)
            brain_r = rank_centered_gpu(brain_rdm_gpu)
            del brain_rdm_gpu
            for key, curve in curves_raw_temp.items():
                curve[lag_i] = spearman_two_ranked(brain_r, llm_ranked_cache[key])
            del brain_r

        for (li_t, ci_t), curve in curves_raw_temp.items():
            smoothed = lag_window_average(curve, smooth_half_pts)
            if not np.any(np.isfinite(smoothed)):
                continue
            peak_idx = int(np.nanargmax(smoothed))
            peak_lag = int(eval_lags[peak_idx])
            peak_lags[li_t, ci_t] = peak_lag
            curves_smoothed[li_t, ci_t] = smoothed
            layer_v = layers[li_t]
            ctx_v = contexts[ci_t]
            print(f"    layer={layer_v:2d}  ctx={ctx_v!s:>4}  "
                  f"peak_lag={peak_lag:>4} ms  "
                  f"max_rho={float(smoothed[peak_idx]):+.4f}")

    # Cells we never managed to fill (e.g. missing embeddings) -> 0.
    peak_lags = np.where(peak_lags == -10_000, 0, peak_lags)

    out = save_result(
        f"rsa_peak_by_layer/{args.roi}_{args.model}",
        eval_lags_ms=eval_lags,
        layers=np.array(layers, dtype=np.int64),
        context_words=ctx_words,
        is_full=is_full,
        peak_lags_ms=peak_lags,
        curves_smoothed=curves_smoothed,
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(args.smooth_half_window_ms),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
