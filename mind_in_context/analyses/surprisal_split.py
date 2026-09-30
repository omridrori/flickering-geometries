"""ERG curve recomputed on Q1 (low) and Q5 (high) surprisal word subsets.

Same baseline-corrected ρ_ERG(t) = Spearman(RDM(t), RDM(t+Δ)) as analyses/erg.py
in --full mode, but the RDM is restricted to:
  - All words           (full RDM, n words)
  - Q1_low              (~bottom 20% by GPT-2-XL surprisal)
  - Q5_high             (~top 20% by GPT-2-XL surprisal)

Surprisal source: stimuli/gpt2-xl/transcript.tsv 'true_prob' column.
Per-word surprisal = sum over its sub-tokens of -log2 p(token | preceding tokens).

CLI:  --roi {lang,aud}
Output: results/surprisal_split/<roi>.npz
Used by: figures/fig3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (
    GPT2XL_TRANSCRIPT_TSV, load_config, make_argparser, save_result,
)
from mind_in_context.lib.rdm import compute_grouped_erg

# ERG gap. The activation window is +-half_window_ms, i.e. 100 ms wide, so
# Delta = 100 ms is the smallest gap for which RDM(t) and RDM(t + Delta) are
# built from non-overlapping samples. This matches Figure 2
# (analyses/segment_erg_psth.py); at Delta = 25 the two matrices share 75% of
# their samples, which inflates rho_ERG.
#
# Delta is deliberately NOT tied to the lag step: the curve is still evaluated
# every lag_step_ms, only the partner RDM sits Delta ms away.
DELTA_MS = int(load_config()["shared"]["erg_delta_ms"])


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--lag-start", type=int, default=None,
                        help="Override lag range start (default: from config).")
    parser.add_argument("--lag-end", type=int, default=None,
                        help="Override lag range end (default: from config).")
    parser.add_argument("--delta", type=int, default=None,
                        help=f"ERG gap Δ in ms (multiple of the lag step). "
                             f"Defaults to {DELTA_MS} ms, matching Figure 2. "
                             f"A non-default value writes to a '_d<delta>' "
                             f"result file so the published one is untouched.")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip pre-onset baseline correction.")
    args = parser.parse_args()

    s = load_config()["shared"]
    lag_step = s["lag_step_ms"]
    delta_ms = DELTA_MS if args.delta is None else int(args.delta)
    if delta_ms % lag_step:
        raise ValueError(f"delta {delta_ms} is not a multiple of {lag_step} ms")
    delta_tag = "" if delta_ms == DELTA_MS else f"_d{delta_ms}"
    lag_start = args.lag_start if args.lag_start is not None else s["lag_start_ms"]
    lag_end = args.lag_end if args.lag_end is not None else s["lag_end_ms"]
    eval_lags = np.arange(lag_start, lag_end + 1, lag_step, dtype=int)
    baseline_window = (None if args.no_baseline
                       else tuple(s["baseline_window_ms"]))

    print(f"  ROI: {args.roi}, lag range: [{lag_start}, {lag_end}] ms "
          f"(step {lag_step} ms), delta: {delta_ms} ms, "
          f"baseline: {baseline_window if baseline_window else 'OFF'}")
    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=s["half_window_ms"],
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=baseline_window,
        delta_ms=delta_ms,
    )
    print(f"  electrodes: {extractor.n_electrodes},  words: {extractor.n_words}")

    q1_idx, q5_idx, q_lo, q_hi = _surprisal_groups(extractor.n_words,
                                                   q_low=0.20, q_high=0.80)
    all_idx = np.arange(extractor.n_words)
    groups = [all_idx, q1_idx, q5_idx]
    print(f"   all={len(all_idx)}  q1_low={len(q1_idx)}  q5_high={len(q5_idx)}")
    print(f"   surprisal cuts: Q1<={q_lo:.3f}, Q5>={q_hi:.3f} (bits)")

    rho = compute_grouped_erg(extractor, eval_lags, groups, delta_ms)

    save_kwargs = dict(
        eval_lags_ms=eval_lags,
        rho_all=rho[0],
        rho_q1_low=rho[1],
        rho_q5_high=rho[2],
        n_words=np.int64(extractor.n_words),
        n_q1=np.int64(len(q1_idx)),
        n_q5=np.int64(len(q5_idx)),
        surprisal_cut_q1=np.float64(q_lo),
        surprisal_cut_q5=np.float64(q_hi),
        baseline_corrected=np.bool_(baseline_window is not None),
        delta_ms=np.int64(delta_ms),
        lag_step_ms=np.int64(lag_step),
    )
    if baseline_window is not None:
        save_kwargs["baseline_window"] = np.array(baseline_window, dtype=np.int64)
    out = save_result(f"surprisal_split/{args.roi}{delta_tag}", **save_kwargs)
    print(f"  -> {out}")


def _surprisal_groups(n_words: int, q_low: float = 0.20, q_high: float = 0.80
                      ) -> tuple[np.ndarray, np.ndarray, float, float]:
    df = pd.read_csv(GPT2XL_TRANSCRIPT_TSV, sep="\t")
    df["surprisal"] = -np.log2(df["true_prob"].astype(float).clip(lower=1e-12))
    df_w = df.groupby("word_idx").agg(surprisal=("surprisal", "sum")).reset_index()
    df_w = df_w[df_w["word_idx"] < n_words]
    cut_lo = float(df_w["surprisal"].quantile(q_low))
    cut_hi = float(df_w["surprisal"].quantile(q_high))
    q1 = df_w[df_w["surprisal"] <= cut_lo]["word_idx"].to_numpy(dtype=np.int64)
    q5 = df_w[df_w["surprisal"] >= cut_hi]["word_idx"].to_numpy(dtype=np.int64)
    return q1, q5, cut_lo, cut_hi


if __name__ == "__main__":
    main()
