"""2D lag×lag RDM-similarity matrices split by GPT-2-XL surprisal.

For each surprisal group (Q1 low, Q5 high):
  S[i,j] = similarity between the brain RDM at lag t_i and lag t_j,
          restricted to that surprisal word subset.

Two baseline methods (--baseline):
  diag  - S = Spearman ρ, then subtract diagonal mean f(|i-j|).  Classic.
  mean  - Rank-center every RDM, subtract the mean RDM across all lags,
          then S[i,j] = cosine(deviation_i, deviation_j). "Do these two
          lags deviate from the average geometry in the same way?"

Surprisal source: stimuli/gpt2-xl/transcript.tsv 'true_prob' column,
converted to bits via -log2.
Q1 = bottom 20% of words by surprisal (predictable).
Q5 = top    20% of words by surprisal (surprising).

CLI:   --roi {lang,aud}  [--baseline {diag,mean}]
       [--lag-start N] [--lag-end N] [--lag-step N]
Output: results/rdm_similarity_2d_surprisal/<roi>_<baseline>.npz
Used by: figures/fig3.py (panels C/D - Q1 vs Q5 surprisal matrices)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (
    GPT2XL_TRANSCRIPT_TSV, load_config, make_argparser, save_result,
)
from mind_in_context.lib.rdm import spearman, rank_centered_gpu

LAG_START_MS = 0
LAG_END_MS = 1000
LAG_STEP_MS = 25


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _compute_S(extractor, eval_lags: np.ndarray, word_idx: np.ndarray,
               baseline: str, label: str) -> np.ndarray:
    """Compute n_lags × n_lags similarity matrix for a word subset."""
    n = len(eval_lags)

    if baseline == "diag":
        S = np.eye(n, dtype=np.float64)
        for i in tqdm(range(n), desc=f"  [{label}]", leave=False):
            for j in range(i + 1, n):
                r = spearman(extractor.rdm(int(eval_lags[i]), word_idx),
                             extractor.rdm(int(eval_lags[j]), word_idx))
                S[i, j] = r
                S[j, i] = r
        f_d  = np.array([S.diagonal(d).mean() for d in range(n)])
        gaps = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        S    = S - f_d[gaps]
    else:  # mean
        ranked = []
        for lag in tqdm(eval_lags, desc=f"  [{label}] rank", leave=False):
            rdm = extractor.rdm(int(lag), word_idx)
            r   = rank_centered_gpu(rdm)
            ranked.append(_to_numpy(r).astype(np.float32, copy=False))

        mean_rdm = np.mean(np.stack(ranked, axis=0), axis=0)
        devs  = [r - mean_rdm for r in ranked]
        norms = np.array([np.linalg.norm(d) for d in devs], dtype=np.float64)

        S = np.zeros((n, n), dtype=np.float64)
        for i in tqdm(range(n), desc=f"  [{label}] S", leave=False):
            for j in range(i, n):
                rho = (float(np.dot(devs[i].astype(np.float64),
                                    devs[j].astype(np.float64)))
                       / (norms[i] * norms[j] + 1e-12))
                S[i, j] = S[j, i] = rho

    return S.astype(np.float32)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--lag-start",  type=int, default=LAG_START_MS)
    parser.add_argument("--lag-end",    type=int, default=LAG_END_MS)
    parser.add_argument("--lag-step",   type=int, default=LAG_STEP_MS)
    parser.add_argument("--baseline",   choices=["diag", "mean"], default="mean")
    args = parser.parse_args()

    eval_lags = np.arange(args.lag_start, args.lag_end + 1,
                          args.lag_step, dtype=int)
    # Same activation pipeline as the ERG and as the Fig 2C lag x lag matrices
    # (rdm_similarity_2d.py): config half-window, winsorization and pre-onset
    # baseline. Previously this script used +-100 ms with neither, so Figures 2C
    # and 3B,C were built from different data even though the Methods describe
    # them in one sentence.
    s = load_config()["shared"]
    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=s["half_window_ms"],
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=tuple(s["baseline_window_ms"]),
    )
    print(f"  ROI {args.roi},  {extractor.n_electrodes} electrodes, "
          f"{extractor.n_words} words")
    print(f"  {len(eval_lags)} lags  [{args.lag_start}, {args.lag_end}] ms")
    print(f"  baseline: {args.baseline}")

    q1_idx, q5_idx, cut_lo, cut_hi = _surprisal_groups(extractor.n_words)
    print(f"  surprisal cuts: Q1 ≤ {cut_lo:.2f} bits, "
          f"Q5 ≥ {cut_hi:.2f} bits  "
          f"(n_q1={len(q1_idx)}, n_q5={len(q5_idx)})")

    S_q1 = _compute_S(extractor, eval_lags, q1_idx, args.baseline, "Q1 low")
    S_q5 = _compute_S(extractor, eval_lags, q5_idx, args.baseline, "Q5 high")

    print(f"  S_q1 range [{S_q1.min():.4f}, {S_q1.max():.4f}]")
    print(f"  S_q5 range [{S_q5.min():.4f}, {S_q5.max():.4f}]")

    out = save_result(
        f"rdm_similarity_2d_surprisal/{args.roi}_{args.baseline}",
        S_q1=S_q1,
        S_q5=S_q5,
        lag_ms=eval_lags.astype(np.int64),
        n_q1=np.int64(len(q1_idx)),
        n_q5=np.int64(len(q5_idx)),
        surprisal_cut_q1=np.float64(cut_lo),
        surprisal_cut_q5=np.float64(cut_hi),
        half_window_ms=np.int64(s["half_window_ms"]),
        winsorize_sd=np.float64(s["winsorize_sd"]),
        baseline_window_ms=np.array(s["baseline_window_ms"], dtype=np.int64),
        baseline=np.array(args.baseline),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
