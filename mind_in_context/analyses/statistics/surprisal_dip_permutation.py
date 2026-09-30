"""Statistical test for Fig 3 - is the ultra-fast ERG dip selective to
high-surprisal (Q5) words relative to low-surprisal (Q1) words?

Fig 3 shows the ERG curve rho_ERG(t) = Spearman(RDM(t), RDM(t+delta))
recomputed on the bottom-20% (Q1, low surprisal) and top-20% (Q5, high
surprisal) words. The claim is that the post-onset dip is *deeper* for Q5.
This script attaches a p-value to that claim.

Statistic
---------
Per group, dip depth = min of rho_ERG(t) over the post-onset window
[DIP_WIN_MS]. Observed T = dip_Q1 - dip_Q5  (expected > 0: Q5 dips lower,
so its minimum is smaller, making the difference positive).

Null
----
Surprisal labels carry no information about the geometry. Under the null we
draw two *disjoint* random word groups of the same sizes (n_q1, n_q5) from
all words, recompute each group's ERG curve and the dip-depth difference.
Repeating N times builds the null distribution of T.

Reports one-sided p = P(null >= obs) and two-sided p = P(|null| >= |obs|),
with the (count + 1) / (N + 1) convention used elsewhere in this folder.

Saves: results/surprisal_dip_permutation/<roi>.npz

CLI:  --roi {lang,aud}  [--n-perm N] [--win LO HI] [--seed S]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from mind_in_context.analyses.surprisal_split import DELTA_MS, _surprisal_groups
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import compute_rdm_condensed, spearman

DIP_WIN_MS = (0, 500)   # post-onset window the dip lives in
N_PERM = 2000
RNG_SEED = 0


def _erg_curve(extractor, eval_lags, group, delta_ms):
    """rho_ERG(t) for one word group, without touching the extractor's
    per-subset RDM cache (which would grow unbounded across permutations).
    Lag-windowed activity *is* cached (shared across groups & permutations)."""
    rho = np.empty(len(eval_lags), dtype=np.float64)
    for i, lag in enumerate(eval_lags):
        a = compute_rdm_condensed(extractor.activity(int(lag))[group])
        b = compute_rdm_condensed(extractor.activity(int(lag) + delta_ms)[group])
        rho[i] = spearman(a, b)
    return rho


def _dip_depth(curve):
    return float(np.nanmin(curve))


def _run(roi: str, n_perm: int, win: tuple[int, int], seed: int,
         delta_ms: int | None = None, baseline: bool = False) -> None:
    s = load_config()["shared"]
    lag_step = s["lag_step_ms"]
    # Delta is the ERG gap, not the lag spacing. It must match the figure this
    # test underwrites (analyses/surprisal_split.py, Figure 3), otherwise the
    # p-value refers to a different statistic than the one plotted.
    delta_ms = DELTA_MS if delta_ms is None else int(delta_ms)
    if delta_ms % lag_step:
        raise ValueError(f"delta {delta_ms} is not a multiple of {lag_step} ms")
    eval_lags = np.arange(win[0], win[1] + 1, lag_step, dtype=int)
    baseline_window = tuple(s["baseline_window_ms"]) if baseline else None

    extractor = BaselineExtractor(
        roi_key=roi,
        eval_lags_ms=np.arange(s["lag_start_ms"], s["lag_end_ms"] + 1, lag_step),
        half_window_ms=s["half_window_ms"],
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=baseline_window,
        delta_ms=delta_ms,
    )
    n_words = extractor.n_words

    q1_idx, q5_idx, q_lo, q_hi = _surprisal_groups(n_words, q_low=0.20, q_high=0.80)
    n_q1, n_q5 = len(q1_idx), len(q5_idx)

    print(f"\n=== {roi.upper()} - surprisal dip permutation ===")
    print(f"  electrodes={extractor.n_electrodes}  words={n_words}")
    print(f"  dip window: {win} ms  ({len(eval_lags)} lags, delta={delta_ms} ms)")
    print(f"  groups: Q1_low n={n_q1}  Q5_high n={n_q5}")
    print(f"  surprisal cuts: Q1<={q_lo:.3f}  Q5>={q_hi:.3f} bits")

    curve_q1 = _erg_curve(extractor, eval_lags, q1_idx, delta_ms)
    curve_q5 = _erg_curve(extractor, eval_lags, q5_idx, delta_ms)
    dip_q1, dip_q5 = _dip_depth(curve_q1), _dip_depth(curve_q5)
    lag_q1 = int(eval_lags[int(np.nanargmin(curve_q1))])
    lag_q5 = int(eval_lags[int(np.nanargmin(curve_q5))])
    obs_T = dip_q1 - dip_q5

    print(f"  observed dip depth:  Q1={dip_q1:+.4f} @ {lag_q1} ms"
          f"   Q5={dip_q5:+.4f} @ {lag_q5} ms")
    print(f"  observed T = dip_Q1 - dip_Q5 = {obs_T:+.4f}")
    print(f"  running {n_perm} permutations (seed={seed}) ...")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_words)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(all_idx)
        g_lo = np.sort(perm[:n_q1])
        g_hi = np.sort(perm[n_q1:n_q1 + n_q5])
        d_lo = _dip_depth(_erg_curve(extractor, eval_lags, g_lo, delta_ms))
        d_hi = _dip_depth(_erg_curve(extractor, eval_lags, g_hi, delta_ms))
        null[k] = d_lo - d_hi
        if (k + 1) % 200 == 0:
            print(f"    {k + 1}/{n_perm}")

    p_one = (np.sum(null >= obs_T) + 1) / (n_perm + 1)
    p_two = (np.sum(np.abs(null) >= abs(obs_T)) + 1) / (n_perm + 1)

    print(f"  null T: mean={null.mean():+.4f}  sd={null.std():.4f}")
    print(f"  one-sided p (null >= obs) = {p_one:.4g}")
    print(f"  two-sided p (|null| >= |obs|) = {p_two:.4g}")

    out = save_result(
        f"surprisal_dip_permutation/{roi}",
        eval_lags_ms=eval_lags.astype(np.int64),
        curve_q1=curve_q1,
        curve_q5=curve_q5,
        dip_q1=np.float64(dip_q1),
        dip_q5=np.float64(dip_q5),
        dip_lag_q1_ms=np.int64(lag_q1),
        dip_lag_q5_ms=np.int64(lag_q5),
        observed_T=np.float64(obs_T),
        null_T=null,
        p_one_sided=np.float64(p_one),
        p_two_sided=np.float64(p_two),
        n_perm=np.int64(n_perm),
        n_q1=np.int64(n_q1),
        n_q5=np.int64(n_q5),
        rng_seed=np.int64(seed),
        dip_window_ms=np.array(win, dtype=np.int64),
    )
    print(f"  -> {out}")


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--win", type=int, nargs=2, default=list(DIP_WIN_MS),
                        metavar=("LO", "HI"),
                        help="Post-onset dip window in ms (default 0 500).")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--delta", type=int, default=None,
                        help=f"ERG gap Δ in ms. Defaults to {DELTA_MS}, "
                             f"matching Figure 2 and surprisal_split.py.")
    parser.add_argument("--baseline", action="store_true",
                        help="Enable pre-onset baseline correction. Default is "
                             "OFF, matching how surprisal_split.py was run for "
                             "Figure 3.")
    args = parser.parse_args()
    _run(args.roi, args.n_perm, tuple(args.win), args.seed,
         delta_ms=args.delta, baseline=args.baseline)


if __name__ == "__main__":
    main()
