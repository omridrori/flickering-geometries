"""Size-matched null for the low-audio control - Supplementary Fig. 1.

The audio control restricts the brain-LLM RSA to the third of the ROI least
coupled to the speech envelope, and compares the result against the full ROI.
That comparison confounds two things: those contacts are less audio-driven,
but there are also a third as many of them, and a smaller contact set gives a
noisier RDM and therefore a lower correlation on its own.

This builds the missing denominator. Contacts are drawn at random from the same
ROI, in the same number as the low-audio set, and the identical RSA pipeline is
run on each draw. The spread of those curves is what subset size alone
produces, so whatever the low-audio curve does outside that spread is
attributable to audio-responsiveness rather than to counting.

Draws are uniform over the ROI, not stratified by patient. The low-audio set is
not patient-balanced either (8 of its 24 contacts are sub-05), so a
patient-matched null would answer a different question than the one asked here.

Every RSA parameter is imported from brain_llm_rsa rather than restated, so the
null cannot drift away from the curve it is a null for.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}  [--n-subsets N] [--seed S]
Output: results/low_audio_random_subsets/<roi>_<model>.npz
Used by: figures/figS_low_audio_subset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.analyses.brain_llm_rsa import (HALF_WINDOW_MS, LAG_END_MS,
                                                    LAG_START_MS, LAG_STEP_MS,
                                                    SMOOTH_HALF_WINDOW_MS)
from mind_in_context.lib import llm
from mind_in_context.lib.data import (BaselineExtractor, low_audio_electrode_indices,
                                      roi_indices)
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import lag_window_average, spearman

N_SUBSETS = 20
RNG_SEED = 0


def _rsa_curve(indices, eval_lags, llm_rdm_vec, winsorize_sd, desc) -> np.ndarray:
    """One smoothed RSA curve, exactly as analyses/brain_llm_rsa.py builds it."""
    ext = BaselineExtractor(
        electrode_indices=np.asarray(indices, dtype=np.int64),
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=winsorize_sd,
        baseline_window_ms=None,
    )
    raw = np.full(len(eval_lags), np.nan)
    for t, lag in enumerate(tqdm(eval_lags, desc=desc, leave=False)):
        raw[t] = spearman(ext.rdm(int(lag)), llm_rdm_vec)
    return lag_window_average(raw, max(1, SMOOTH_HALF_WINDOW_MS // LAG_STEP_MS))


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--n-subsets", type=int, default=N_SUBSETS,
                        help="number of random contact subsets to draw")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    cfg = load_config()
    s = cfg["shared"]
    eval_lags = np.arange(LAG_START_MS, LAG_END_MS + 1, LAG_STEP_MS, dtype=int)

    roi = roi_indices(args.roi)
    low_idx = low_audio_electrode_indices(args.roi)
    n_subset = len(low_idx)
    print(f"  ROI {args.roi}: {len(roi)} contacts, low-audio set {n_subset}")
    print(f"  {args.n_subsets} random subsets of {n_subset}, seed {args.seed}, "
          f"lags [{LAG_START_MS}, {LAG_END_MS}] ms")

    layer = cfg["models"][args.model]["layer"]
    emb = np.load(llm.full_context_embeddings_path(args.model, layer))
    emb = emb.astype(np.float64)
    probe = BaselineExtractor(
        electrode_indices=low_idx, eval_lags_ms=eval_lags[:1],
        half_window_ms=HALF_WINDOW_MS, winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    n_words = min(emb.shape[0], probe.n_words)
    del probe
    llm_rdm_vec = llm.cosine_rdm(emb[:n_words])
    print(f"  LLM RDM built (n_words={n_words})")

    low_curve = _rsa_curve(low_idx, eval_lags, llm_rdm_vec, s["winsorize_sd"],
                           "    low-audio")
    rng = np.random.default_rng(args.seed)
    curves = np.full((args.n_subsets, len(eval_lags)), np.nan)
    picks = np.full((args.n_subsets, n_subset), -1, dtype=np.int64)
    for i in range(args.n_subsets):
        picks[i] = rng.choice(roi, size=n_subset, replace=False)
        curves[i] = _rsa_curve(picks[i], eval_lags, llm_rdm_vec, s["winsorize_sd"],
                               f"    subset {i + 1}/{args.n_subsets}")

    post = (eval_lags >= 0) & (eval_lags <= 1000)
    low_peak = float(np.nanmax(low_curve[post]))
    peaks = np.nanmax(curves[:, post], axis=1)
    n_below = int((peaks <= low_peak).sum())
    out = save_result(
        f"low_audio_random_subsets/{args.roi}_{args.model}",
        eval_lags_ms=eval_lags,
        low_curve=low_curve,
        curves=curves,
        subset_indices=picks,
        low_electrode_indices=low_idx,
        n_subset=np.int64(n_subset),
        n_roi=np.int64(len(roi)),
        seed=np.int64(args.seed),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
    )
    print(f"\n  low-audio peak      {low_peak:.4f}")
    print(f"  random subset peaks {peaks.mean():.4f} ± {peaks.std(ddof=1):.4f} "
          f"[{peaks.min():.4f}, {peaks.max():.4f}]")
    print(f"  random subsets at or below the low-audio peak: {n_below} of "
          f"{args.n_subsets}  (p = {(1 + n_below) / (1 + args.n_subsets):.4f})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
