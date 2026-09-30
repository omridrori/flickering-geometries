"""Dynamic brain × LLM RSA with two controls - Fig 4A.

Key choices:

  - LLM embeddings: the full-context set (lib.llm.full_context_embeddings_path),
    each word embedded with the whole preceding narrative available.
  - LLM RDM: lib.llm.cosine_rdm (column-centre, row L2-normalise,
    cosine pdist). Different from pdist(correlation) at the level we
    care about for this figure.
  - Brain RDM: pdist(correlation) at each lag, ±100 ms half-window.
  - Lag smoothing: ±50 ms moving average along the lag axis applied to
    every curve.
  - Brain cache: 25 ms.
  - No baseline correction (wide-range RSA).

Three curves:
  rsa_all       - all ROI electrodes
  rsa_low_audio - bottom 1/3 of ROI electrodes by |peak audio xcorr|
  rsa_shifted   - all electrodes, brain word axis circularly shifted
                  by SHIFT_WORDS

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
Output: results/brain_llm_rsa/<roi>_<model>.npz
Used by: figures/fig4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor, low_audio_electrode_indices
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import lag_window_average, spearman

LAG_START_MS = -4000
LAG_END_MS = 4000
LAG_STEP_MS = 25
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 50
SHIFT_WORDS = 600
LOW_AUDIO_FRACTION = 1 / 3


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    args = parser.parse_args()

    s = load_config()["shared"]
    eval_lags = np.arange(LAG_START_MS, LAG_END_MS + 1, LAG_STEP_MS, dtype=int)

    print(f"  ROI: {args.roi}, model: {args.model}, "
          f"lags [{LAG_START_MS}, {LAG_END_MS}] ms, half_window={HALF_WINDOW_MS} ms")

    # ----- Brain extractors (no baseline correction).
    ext_all = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    print(f"  all electrodes: {ext_all.n_electrodes}")

    low_audio_idx = low_audio_electrode_indices(args.roi, LOW_AUDIO_FRACTION)
    ext_low = BaselineExtractor(
        electrode_indices=low_audio_idx,
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    print(f"  low-audio electrodes: {ext_low.n_electrodes} "
          f"(bottom {LOW_AUDIO_FRACTION:.0%} by |audio xcorr|)")

    # ----- LLM RDM: full-context embeddings, cosine metric.
    layer = load_config()["models"][args.model]["layer"]
    emb_path = llm.full_context_embeddings_path(args.model, layer)
    print(f"  loading LLM embeddings: {emb_path.name}")
    emb = np.load(emb_path).astype(np.float64)
    n_words = min(emb.shape[0], ext_all.n_words)
    emb = emb[:n_words]
    print(f"  building LLM RDM (cosine on centered+L2-normed, n_words={n_words})...")
    llm_rdm_vec = llm.cosine_rdm(emb)

    # ----- Word permutation for the shift control.
    shift_perm = (np.arange(n_words) + SHIFT_WORDS) % n_words

    # ----- Compute raw RSA curves at each lag.
    rsa_all_raw = np.full(len(eval_lags), np.nan)
    rsa_low_raw = np.full(len(eval_lags), np.nan)
    rsa_shift_raw = np.full(len(eval_lags), np.nan)
    for t, lag in enumerate(tqdm(eval_lags, desc="    RSA")):
        rsa_all_raw[t] = spearman(ext_all.rdm(int(lag)), llm_rdm_vec)
        rsa_low_raw[t] = spearman(ext_low.rdm(int(lag)), llm_rdm_vec)
        rsa_shift_raw[t] = spearman(ext_all.rdm(int(lag), shift_perm), llm_rdm_vec)

    # ----- Lag-axis smoothing.
    smooth_half_pts = max(1, SMOOTH_HALF_WINDOW_MS // LAG_STEP_MS)
    rsa_all = lag_window_average(rsa_all_raw, smooth_half_pts)
    rsa_low = lag_window_average(rsa_low_raw, smooth_half_pts)
    rsa_shifted = lag_window_average(rsa_shift_raw, smooth_half_pts)

    out = save_result(
        f"brain_llm_rsa/{args.roi}_{args.model}",
        eval_lags_ms=eval_lags,
        rsa_all=rsa_all,
        rsa_low_audio=rsa_low,
        rsa_shifted=rsa_shifted,
        rsa_all_raw=rsa_all_raw,
        rsa_low_audio_raw=rsa_low_raw,
        rsa_shifted_raw=rsa_shift_raw,
        n_all=np.int64(ext_all.n_electrodes),
        n_low_audio=np.int64(ext_low.n_electrodes),
        shift_words=np.int64(SHIFT_WORDS),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
