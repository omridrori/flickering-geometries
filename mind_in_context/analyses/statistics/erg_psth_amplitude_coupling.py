"""Statistics for Fig 2 / Discussion - per-segment ERG vs PSTH amplitudes.

Over equal-word narrative segments we summarise, in the window imported from
segment_erg_psth (100-600 ms), four quantities: ERG mean, ERG rebound slope,
PSTH mean, PSTH slope. From these we report:

  (1) ERG-PSTH amplitude DECOUPLING (Discussion claim):
      Pearson correlation between per-segment ERG mean and per-segment PSTH
      mean ACROSS segments. The claim is that this link is weak / not
      significant at the individual-segment level.

  (2) Infra-slow TREND vs narrative position (Fig 2C/D):
      Pearson correlation of each quantity with segment position, tested by
      shuffling values across segments (one-sided in the predicted
      direction; predictions: ERG mean -, ERG slope +, PSTH mean +,
      PSTH slope +).

Pipeline matches the Fig 2 per-segment analysis: ROI lang, equal-word
segments, half-window +/-50 ms, winsorize 3 sigma, baseline ON, delta 100 ms.

Saves: results/erg_psth_amplitude_coupling/<roi>_<nseg>seg.npz

CLI:  --roi {lang,aud}  [--segments 20]  [--win 200 400]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import compute_grouped_erg
from mind_in_context.lib.segments import eqword_segments
from mind_in_context.lib.stats import per_segment_stats

# Read from the analysis this test underwrites rather than restated here: the
# two drifted apart once (550 vs 600) and the statistics then summarised a
# different window than the figure they describe.
from mind_in_context.analyses.segment_erg_psth import LAG_MAX_MS, LAG_MIN_MS

WIN_MS = (LAG_MIN_MS, LAG_MAX_MS)
N_SEGMENTS = 20
N_PERM = 10_000
RNG_SEED = 0

# Predicted direction of the trend vs segment position (True = positive).
PREDICTED_POSITIVE = {
    "ERG mean": False, "ERG slope": True, "PSTH mean": True, "PSTH slope": True,
}


def _trend_perm_p(values, predicted_positive, rng, n_perm):
    x = np.arange(len(values), dtype=float)
    r_obs = float(np.corrcoef(x, values)[0, 1])
    null = np.array([np.corrcoef(x, rng.permutation(values))[0, 1]
                     for _ in range(n_perm)])
    if predicted_positive:
        p_one = (np.sum(null >= r_obs) + 1) / (n_perm + 1)
    else:
        p_one = (np.sum(null <= r_obs) + 1) / (n_perm + 1)
    p_two = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, float(p_one), float(p_two)


def _run(roi: str, n_seg: int, win: tuple[int, int]) -> None:
    s = load_config()["shared"]
    lag_step = int(s["lag_step_ms"])
    # Delta is the ERG gap, not the lag spacing (cfg.shared.erg_delta_ms).
    delta_ms = int(s["erg_delta_ms"])
    eval_lags = np.arange(win[0], win[1] + 1, lag_step, dtype=int)

    extractor = BaselineExtractor(
        roi_key=roi,
        eval_lags_ms=eval_lags,
        half_window_ms=s["half_window_ms"],
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=tuple(s["baseline_window_ms"]),
        delta_ms=delta_ms,
    )
    groups, _ = eqword_segments(extractor.n_words, n_seg)

    erg = compute_grouped_erg(extractor, eval_lags, groups, delta_ms)
    psth = np.full((n_seg, len(eval_lags)), np.nan)
    for t, lag in enumerate(tqdm(eval_lags, desc="    PSTH", leave=False)):
        act = extractor.activity(int(lag))
        for k, g in enumerate(groups):
            if len(g):
                psth[k, t] = act[g, :].mean()

    erg_mean, _, erg_slope = per_segment_stats(eval_lags, erg, win[0], win[1])
    psth_mean, _, psth_slope = per_segment_stats(eval_lags, psth, win[0], win[1])
    erg_slope *= 1000.0
    psth_slope *= 1000.0

    # (1) Decoupling: ERG mean vs PSTH mean across segments.
    r_couple, p_couple = pearsonr(erg_mean, psth_mean)
    rho_couple, prho_couple = spearmanr(erg_mean, psth_mean)

    # (2) Trend vs segment position.
    rng = np.random.default_rng(RNG_SEED)
    metrics = {"ERG mean": erg_mean, "ERG slope": erg_slope,
               "PSTH mean": psth_mean, "PSTH slope": psth_slope}
    trend = {name: _trend_perm_p(v, PREDICTED_POSITIVE[name], rng, N_PERM)
             for name, v in metrics.items()}

    print(f"\n=== {roi.upper()} - {n_seg} eqword segments, win {win} ms ===")
    print(f"  (1) ERG-mean vs PSTH-mean DECOUPLING across segments:")
    print(f"      Pearson  r = {r_couple:+.4f}  p = {p_couple:.4g}")
    print(f"      Spearman rho = {rho_couple:+.4f}  p = {prho_couple:.4g}")
    print(f"  (2) trend vs segment position (perm p, one-sided in predicted dir):")
    for name, (r, p1, p2) in trend.items():
        print(f"      {name:<10} r={r:+.4f}  p_one={p1:.4g}  p_two={p2:.4g}")

    out = save_result(
        f"erg_psth_amplitude_coupling/{roi}_{n_seg}seg",
        win_ms=np.array(win, dtype=np.int64),
        n_segments=np.int64(n_seg),
        erg_mean=erg_mean, erg_slope=erg_slope,
        psth_mean=psth_mean, psth_slope=psth_slope,
        couple_pearson_r=np.float64(r_couple),
        couple_pearson_p=np.float64(p_couple),
        couple_spearman_rho=np.float64(rho_couple),
        couple_spearman_p=np.float64(prho_couple),
        trend_names=np.array(list(trend.keys())),
        trend_r=np.array([trend[k][0] for k in trend]),
        trend_p_one=np.array([trend[k][1] for k in trend]),
        trend_p_two=np.array([trend[k][2] for k in trend]),
        n_perm=np.int64(N_PERM), rng_seed=np.int64(RNG_SEED),
    )
    print(f"  -> {out}")


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--segments", type=int, nargs="+", default=[N_SEGMENTS])
    parser.add_argument("--win", type=int, nargs=2, default=list(WIN_MS),
                        metavar=("LO", "HI"))
    args = parser.parse_args()
    for n_seg in args.segments:
        _run(args.roi, n_seg, tuple(args.win))


if __name__ == "__main__":
    main()
