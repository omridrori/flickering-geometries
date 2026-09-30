"""ERG curve - Event Related Geometry.

ρ_ERG(t) = Spearman(RDM(t), RDM(t + Δ)) on raw (winsorized) activity.

Defaults:
  lag_step = 25 ms, range −500..1000, ±50 ms half-window, Δ = 100 ms,
  winsorize ON (3σ), baseline OFF, 25 ms brain cache.

Modes:
  --full           single curve over all 5,136 words (one row in output)
  (default)        per equal-word-count segment + mean ± SEM

Opt-ins (rare):
  --baseline       enable baseline correction (window from cfg.shared.baseline_window_ms)
  --no-winsorize   disable winsorize
  --lag-step N     change resolution; <25 switches to the 1 ms brain cache.

Output: results/erg/<roi>_<suffix>.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import compute_grouped_erg
from mind_in_context.lib.segments import eqword_segments, full_time_span


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--full", action="store_true",
                        help="Single curve over all words (no segmentation).")
    parser.add_argument("--lag-step", type=int, default=None)
    parser.add_argument("--lag-start", type=int, default=None)
    parser.add_argument("--lag-end", type=int, default=None)
    parser.add_argument("--half-window-ms", type=int, default=None,
                        help="Per-word activity averaging half-window. "
                             "Default: cfg.shared.half_window_ms (50).")
    parser.add_argument("--delta", type=int, default=None,
                        help="ERG gap Δ in ms (multiple of the lag step). "
                             "Default: cfg.shared.erg_delta_ms (100), the "
                             "smallest gap with non-overlapping activation "
                             "windows. A non-default value writes to a "
                             "'_d<delta>' result file.")
    parser.add_argument("--baseline", action="store_true",
                        help="Enable baseline correction (default: OFF).")
    parser.add_argument("--no-winsorize", action="store_true",
                        help="Disable winsorize (default: ON, 3σ).")
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="Override output file suffix (without .npz). "
                             "Default is auto-built from mode + lag_step.")
    args = parser.parse_args()

    s = load_config()["shared"]
    lag_step = args.lag_step or s["lag_step_ms"]
    lag_start = args.lag_start if args.lag_start is not None else s["lag_start_ms"]
    lag_end = args.lag_end if args.lag_end is not None else s["lag_end_ms"]
    half_window_ms = args.half_window_ms if args.half_window_ms is not None else s["half_window_ms"]
    # Delta is independent of the lag spacing: the curve is evaluated every
    # lag_step ms, the partner matrix sits delta ms away.
    default_delta = int(s["erg_delta_ms"])
    delta_ms = default_delta if args.delta is None else int(args.delta)
    if delta_ms % lag_step:
        raise ValueError(f"delta {delta_ms} is not a multiple of {lag_step} ms")

    eval_lags = np.arange(lag_start, lag_end + 1, lag_step, dtype=int)
    cache = "1ms" if lag_step < 25 else "25ms"

    winsorize_sd = None if args.no_winsorize else s["winsorize_sd"]
    baseline_window = tuple(s["baseline_window_ms"]) if args.baseline else None

    print(f"  ROI: {args.roi}, mode: {'full' if args.full else 'eqword'}, "
          f"lag step: {lag_step} ms, eval cache: {cache}, delta: {delta_ms} ms, "
          f"half-window: ±{half_window_ms} ms, baseline: "
          f"{'on' if args.baseline else 'off'}, "
          f"winsorize: {'off' if args.no_winsorize else 'on'}")

    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=half_window_ms,
        winsorize_sd=winsorize_sd,
        baseline_window_ms=baseline_window,
        delta_ms=delta_ms,
        cache=cache,
    )
    print(f"  electrodes: {extractor.n_electrodes},  words: {extractor.n_words}")

    if args.full:
        groups = [np.arange(extractor.n_words)]
        time_edges = np.array([full_time_span(extractor.n_words)])
    else:
        groups, time_edges = eqword_segments(extractor.n_words, s["n_eqword_segments"])
    seg_n_words = np.array([len(g) for g in groups])

    rho = compute_grouped_erg(extractor, eval_lags, groups, delta_ms)

    if args.out_suffix is not None:
        suffix = args.out_suffix
    else:
        suffix = "full" if args.full else "eqword"
        if lag_step != 25:
            suffix = f"{suffix}_step{lag_step}ms"
        if half_window_ms != s["half_window_ms"]:
            suffix = f"{suffix}_hw{half_window_ms}"
        if args.baseline:
            suffix = f"{suffix}_baseline"
        if args.no_winsorize:
            suffix = f"{suffix}_nowins"
        if delta_ms != default_delta:
            suffix = f"{suffix}_d{delta_ms}"
    out = save_result(
        f"erg/{args.roi}_{suffix}",
        eval_lags_ms=eval_lags,
        rho=rho,
        segment_n_words=seg_n_words,
        segment_time_edges=time_edges,
        baseline_window=np.array(s["baseline_window_ms"], dtype=np.int64),
        delta_ms=np.int64(delta_ms),
        lag_step_ms=np.int64(lag_step),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
