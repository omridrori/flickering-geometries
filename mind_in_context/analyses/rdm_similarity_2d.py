"""2D RDM-similarity matrix S(t1, t2) per equal-word-count segment.

For each of N segments (default 5):
  S - Spearman ρ between RDMs at every pair of lags within the segment
  R - Δ-residual: S − f_seg(|t1 − t2|), where f_seg(d) is the mean of S
      along the d-th diagonal (segment-internal expected similarity at lag-gap d)

CLI:  --roi {lang,aud}
Output: results/rdm_similarity_2d/<roi>.npz
Used by: figures/fig2.py (panel I), figures/figS2_rdm_similarity_2d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import spearman
from mind_in_context.lib.segments import eqword_segments

LAG_START_MS = -200
LAG_END_MS = 1000
N_SEGMENTS = 5


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    args = parser.parse_args()

    s = load_config()["shared"]
    eval_lags = np.arange(LAG_START_MS, LAG_END_MS + 1,
                          s["lag_step_ms"], dtype=int)
    n_lags_eval = len(eval_lags)

    print(f"  ROI: {args.roi}, {N_SEGMENTS} eqword segments, "
          f"lags [{LAG_START_MS}, {LAG_END_MS}] ms")
    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=s["half_window_ms"],
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=tuple(s["baseline_window_ms"]),
    )
    print(f"  electrodes: {extractor.n_electrodes},  words: {extractor.n_words}")

    groups, time_edges = eqword_segments(extractor.n_words, N_SEGMENTS)
    seg_n_words = np.array([len(g) for g in groups])

    S_all = np.zeros((N_SEGMENTS, n_lags_eval, n_lags_eval))
    R_all = np.zeros_like(S_all)
    for k in range(N_SEGMENTS):
        S = np.eye(n_lags_eval)
        for i in tqdm(range(n_lags_eval), desc=f"    seg {k+1}", leave=False):
            for j in range(i + 1, n_lags_eval):
                r = spearman(extractor.rdm(int(eval_lags[i]), groups[k]),
                             extractor.rdm(int(eval_lags[j]), groups[k]))
                S[i, j] = r
                S[j, i] = r
        f_d = np.array([S.diagonal(d).mean() for d in range(n_lags_eval)])
        R = S - f_d[np.abs(np.subtract.outer(np.arange(n_lags_eval),
                                              np.arange(n_lags_eval)))]
        S_all[k] = S
        R_all[k] = R

    out = save_result(
        f"rdm_similarity_2d/{args.roi}",
        eval_lags_ms=eval_lags,
        S=S_all, R=R_all,
        segment_n_words=seg_n_words,
        segment_time_edges=time_edges,
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
