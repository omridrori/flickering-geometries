"""Cross-half ERG and its noise ceiling - supplementary figure.

rho_ERG is bounded above by how reliably each mega-RDM is estimated, so the
word-triggered dip could reflect a change in the geometry or only a change in
measurement quality. Answering that needs a
ceiling measured on the same footing as the ERG itself: the ordinary ERG
compares RDMs built from the SAME contacts over overlapping time windows, so
shared noise inflates it, while a split-half reliability shares nothing and
gets no such boost. Dividing one by the other is not meaningful.

Here every term is an A-against-B comparison between two disjoint halves of
the ROI contacts, so the ratio is interpretable:

    rel(t)    = rho( RDM_A(t), RDM_B(t)   )        the ceiling at lag t
    erg(t)    = rho( RDM_A(t), RDM_B(t+D) )        ERG, no shared-electrode boost
                (symmetrised with the B(t) vs A(t+D) direction)
    disatt(t) = erg(t) / sqrt( rel(t) * rel(t+D) )

disatt is the fraction of the attainable similarity the geometry retains
across D, so a dip in it is a dip that measurement reliability cannot explain.

The split is arbitrary, so the whole thing is repeated over --n-splits random
partitions and the figure shows the mean with a +/-1 SEM band. That SEM is
across re-partitions of the SAME contacts, so it measures how stable the
estimate is, not sampling error over a new cohort.

No baseline correction (matches the default wide-range ERG pipeline).

CLI:  --roi {lang,aud}  [--seed N] [--n-splits N]
Output: results/cross_half_erg/<roi>.npz
Used by: figures/figS_cross_half_erg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import deque

import numpy as np
from tqdm import tqdm

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import (compute_rdm_condensed_gpu,
                                     rank_centered_gpu, spearman_two_ranked)

N_SPLITS = 10


def _split_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint contact index sets. Odd n gives the extra contact to A."""
    order = np.random.default_rng(seed).permutation(n)
    cut = (n + 1) // 2
    return np.sort(order[:cut]), np.sort(order[cut:])


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; split i uses seed + i (default: 0).")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS,
                        help=f"Random half-splits (default: {N_SPLITS}).")
    parser.add_argument("--delta", type=int, default=None,
                        help="ERG gap Δ in ms (multiple of the lag step). "
                             "Default: cfg.shared.erg_delta_ms (100). A "
                             "non-default value writes to a '_d<delta>' file.")
    args = parser.parse_args()

    s = load_config()["shared"]
    lag_step = int(s["lag_step_ms"])
    # Delta is the ERG gap, not the lag spacing (cfg.shared.erg_delta_ms).
    default_delta = int(s["erg_delta_ms"])
    delta = default_delta if args.delta is None else int(args.delta)
    if delta % lag_step:
        raise ValueError(f"delta {delta} is not a multiple of {lag_step} ms")
    eval_lags = np.arange(s["lag_start_ms"], s["lag_end_ms"] + 1, lag_step,
                          dtype=int)
    # The grid runs delta ms past the last evaluated lag, so rho(t, t+delta) is
    # defined everywhere. delta spans K grid steps, not one: pairing each lag
    # with its immediate predecessor (as this script used to) silently fixes the
    # gap at lag_step regardless of delta, and leaves the final appended point
    # measured across a different gap than all the others.
    k_steps = delta // lag_step
    grid = np.arange(s["lag_start_ms"], s["lag_end_ms"] + delta + 1, lag_step,
                     dtype=int)
    assert len(grid) == len(eval_lags) + k_steps

    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=grid,
        half_window_ms=s["half_window_ms"],
        winsorize_sd=s["winsorize_sd"] if s["winsorize"] else None,
        baseline_window_ms=None,
        delta_ms=delta,
    )
    n = extractor.n_electrodes
    print(f"  ROI {args.roi}: {n} contacts, {extractor.n_words} words, "
          f"{args.n_splits} split(s) from seed {args.seed}")

    rel = np.empty((args.n_splits, len(grid)), dtype=np.float64)
    erg = np.full((args.n_splits, len(eval_lags)), np.nan, dtype=np.float64)
    halves_a, halves_b = [], []

    # Splits on the OUTSIDE so only two condensed RDMs are alive at a time.
    # The extractor caches activity, so re-walking the grid per split is cheap.
    for si in range(args.n_splits):
        half_a, half_b = _split_halves(n, args.seed + si)
        halves_a.append(half_a)
        halves_b.append(half_b)
        # Rank once per lag, then every pairing is a dot product. The buffers
        # hold the K+1 most recent lags, so buf[0] is exactly delta ms behind
        # the lag being processed.
        buf_a: deque = deque(maxlen=k_steps)
        buf_b: deque = deque(maxlen=k_steps)
        for i, lag in enumerate(tqdm(grid, leave=False,
                                     desc=f"    split {si + 1}/{args.n_splits}")):
            act = extractor.activity(int(lag))
            rank_a = rank_centered_gpu(compute_rdm_condensed_gpu(act[:, half_a]))
            rank_b = rank_centered_gpu(compute_rdm_condensed_gpu(act[:, half_b]))
            rel[si, i] = spearman_two_ranked(rank_a, rank_b)
            if len(buf_a) == k_steps:            # buf[0] is lag - delta
                erg[si, i - k_steps] = 0.5 * (
                    spearman_two_ranked(buf_a[0], rank_b)
                    + spearman_two_ranked(buf_b[0], rank_a))
            buf_a.append(rank_a)
            buf_b.append(rank_b)

    m = len(eval_lags)
    disatt = erg / np.sqrt(np.clip(rel[:, :m] * rel[:, k_steps:k_steps + m],
                                   1e-12, None))

    pre = (eval_lags >= -500) & (eval_lags <= -100)
    post = (eval_lags >= 0) & (eval_lags <= 500)
    print(f"\n  {'curve':>18} {'baseline':>10} {'min 0-500':>10} "
          f"{'at ms':>7} {'change':>9}")
    for name, cube in (("reliability", rel[:, :m]), ("erg_cross", erg),
                       ("erg_disattenuated", disatt)):
        mu = np.nanmean(cube, axis=0)
        base = np.nanmean(mu[pre])
        k = int(np.nanargmin(mu[post]))
        print(f"  {name:>18} {base:10.4f} {mu[post][k]:10.4f} "
              f"{eval_lags[post][k]:7d} "
              f"{100 * (mu[post][k] - base) / base:+8.2f}%")

    out = save_result(
        f"cross_half_erg/{args.roi}"
        f"{'' if delta == default_delta else f'_d{delta}'}",
        eval_lags_ms=eval_lags,
        reliability=rel[:, :m],
        reliability_next=rel[:, k_steps:k_steps + m],
        erg_cross=erg,
        erg_disattenuated=disatt,
        half_a=np.array(halves_a),
        half_b=np.array(halves_b),
        delta_ms=np.int64(delta),
        n_splits=np.int64(args.n_splits),
        n_electrodes=np.int64(n),
        n_words=np.int64(extractor.n_words),
        seed=np.int64(args.seed),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
