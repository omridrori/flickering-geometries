"""ERG + PSTH per equal-word-count segment, baseline-corrected.

For N eqword segments: the ERG autocorrelation curve ρ(RDM(t), RDM(t+Δ)) and
the PSTH (mean baseline-corrected activation) at every evaluation lag, plus
per-segment summaries (mean / slope) over 100-600 ms. Baseline is ON for both
(pre-onset drift removed) - without it the last, longest segments pick up a
slow drift that inflates apparent geometric stability.

Fig 2 consumes two runs of this analysis:
  --n-segments 5    -> the ERG/PSTH curves        (panels A/B and G/H)
  --n-segments 20   -> the per-segment scatter     (panels C-F)

CLI:  --roi {lang,aud}  --n-segments <int>   (default: 20)
Output: results/segment_erg_psth/<roi>_<N>seg.npz
Used by: figures/fig2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import compute_grouped_erg
from mind_in_context.lib.segments import eqword_segments
from mind_in_context.lib.stats import per_segment_stats

# Summary window for the per-segment mean / slope.
LAG_MIN_MS = 100
LAG_MAX_MS = 600

# ERG gap. The activation window is +-half_window_ms, i.e. 100 ms wide, so
# Delta = 100 ms is the smallest gap for which RDM(t) and RDM(t + Delta) are
# built from non-overlapping samples. Smaller gaps share data between the two
# matrices (Delta = 25 shares 75% of it) and inflate rho_ERG accordingly.
DELTA_MS = int(load_config()["shared"]["erg_delta_ms"])


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--n-segments", type=int, default=20,
                        help="Number of equal-word-count segments.")
    parser.add_argument("--delta", type=int, default=None,
                        help=f"ERG gap Δ in ms (multiple of the lag step). "
                             f"Defaults to {DELTA_MS} ms, the smallest gap "
                             f"with non-overlapping activation windows. A "
                             f"non-default value writes to a '_d<delta>' "
                             f"result file so the published one is untouched.")
    args = parser.parse_args()

    s = load_config()["shared"]
    eval_lags = np.arange(s["lag_start_ms"], s["lag_end_ms"] + 1,
                          s["lag_step_ms"], dtype=int)
    delta_ms = DELTA_MS if args.delta is None else int(args.delta)
    if delta_ms % s["lag_step_ms"]:
        raise ValueError(f"delta {delta_ms} is not a multiple of "
                         f"{s['lag_step_ms']} ms")
    # Empty for the default Δ, so the published filename is unchanged.
    delta_tag = "" if delta_ms == DELTA_MS else f"_d{delta_ms}"
    n_segments = args.n_segments

    print(f"  ROI: {args.roi}, {n_segments} eqword segments, baseline ON, "
          f"delta {delta_ms} ms")
    # A fresh extractor per run: the RDM cache is keyed per lag, so one
    # extractor must only ever see a SINGLE segmentation. Reusing it across
    # different segment counts returns cached RDMs from the wrong word subsets.
    extractor = BaselineExtractor(
        roi_key=args.roi, eval_lags_ms=eval_lags,
        half_window_ms=s["half_window_ms"], winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=tuple(s["baseline_window_ms"]), delta_ms=delta_ms)
    print(f"  electrodes: {extractor.n_electrodes},  words: {extractor.n_words}")

    groups, time_edges = eqword_segments(extractor.n_words, n_segments)

    erg = compute_grouped_erg(extractor, eval_lags, groups, delta_ms)
    psth = np.full((n_segments, len(eval_lags)), np.nan)
    for t, lag in enumerate(tqdm(eval_lags, desc="    PSTH")):
        act = extractor.activity(int(lag))
        for k, g in enumerate(groups):
            psth[k, t] = act[g, :].mean()

    erg_mean, _, erg_slope = per_segment_stats(eval_lags, erg,
                                               LAG_MIN_MS, LAG_MAX_MS)
    psth_mean, _, psth_slope = per_segment_stats(eval_lags, psth,
                                                 LAG_MIN_MS, LAG_MAX_MS)

    out = save_result(
        f"segment_erg_psth/{args.roi}_{n_segments}seg{delta_tag}",
        eval_lags_ms=eval_lags,
        erg=erg, psth=psth,
        erg_mean=erg_mean, erg_slope_per_ms=erg_slope,
        psth_mean=psth_mean, psth_slope_per_ms=psth_slope,
        segment_time_edges=time_edges,
        lag_min_ms=np.int64(LAG_MIN_MS), lag_max_ms=np.int64(LAG_MAX_MS),
        delta_ms=np.int64(delta_ms),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
