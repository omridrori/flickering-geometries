"""Brain x LLM RSA with every word form kept only once - Extended Data Fig. 5.

The narrative repeats itself: 5,136 word tokens are only 1,252 distinct forms.
A repeated word gets nearly the same model representation every time it occurs,
so part of the brain-model agreement in Figure 4A could come from repetition
alone rather than from meaning.

This keeps ONE randomly chosen occurrence of each form, discarding the other
three quarters of the words, and recomputes the whole lag-resolved RSA on that
subset. The draw is repeated N_SEEDS times so the result cannot depend on which
occurrence happened to be picked.

Note the contrast with analyses/surprisal_split_dedup.py (Extended Data Fig. 4), which keeps
the FIRST occurrence deterministically: there the question is about the ERG and
a fixed subset is the cleaner control, here it is about the brain-LLM
correspondence and the spread across random draws is itself the result.

Settings mirror analyses/brain_llm_rsa.py so the all-words curve reproduces the
published Figure 4A reference exactly: lags -4000..4000 ms in 25 ms steps,
+-100 ms activation window, winsorized, no baseline correction,
full-context Llama-3 embeddings, +-50 ms lag smoothing.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
Output: results/brain_llm_rsa_dedup/<roi>_<model>.npz
Used by: figures/figS_brain_llm_dedup.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (TRANSCRIPT_TSV, load_config,
                                    make_argparser, save_result)
from mind_in_context.lib.rdm import lag_window_average, spearman

LAG_START_MS, LAG_END_MS, LAG_STEP_MS = -4000, 4000, 25
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 50
N_SEEDS = 20
PEAK_WINDOW = (0, 1000)          # where the peak is read from
STRIP = ".,!?;:\"'()[]"


def _occurrences_by_form(n_words: int) -> list[np.ndarray]:
    """Word indices grouped by lowercased, punctuation-stripped word form."""
    df = pd.read_csv(TRANSCRIPT_TSV, sep="\t")
    df["word_idx"] = df["word_idx"].astype(int)
    w = (df.groupby("word_idx").agg(word=("word", "first"))
           .reset_index().sort_values("word_idx"))
    w = w[w["word_idx"] < n_words]
    forms = w["word"].astype(str).str.lower().str.strip(STRIP).to_numpy()
    by_form: dict[str, list[int]] = defaultdict(list)
    for idx, form in zip(w["word_idx"].to_numpy(), forms):
        by_form[form].append(int(idx))
    return [np.asarray(v, dtype=np.int64) for v in by_form.values()]


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    s = load_config()["shared"]
    lags = np.arange(LAG_START_MS, LAG_END_MS + 1, LAG_STEP_MS, dtype=int)
    smooth_pts = max(1, SMOOTH_HALF_WINDOW_MS // LAG_STEP_MS)

    extractor = BaselineExtractor(
        roi_key=args.roi, eval_lags_ms=lags, half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"], baseline_window_ms=None)
    n_words = extractor.n_words
    groups = _occurrences_by_form(n_words)
    print(f"  ROI {args.roi}: {extractor.n_electrodes} contacts, "
          f"{n_words} words, {len(groups)} distinct forms, "
          f"{args.n_seeds} random draws")

    layer = load_config()["models"][args.model]["layer"]
    emb = np.load(llm.full_context_embeddings_path(args.model, layer))
    emb = emb[:n_words].astype(np.float64)

    # ---- Reference: all words, the published Figure 4A curve.
    llm_full = llm.cosine_rdm(emb)
    rsa_full = np.array([spearman(extractor.rdm(int(lag)), llm_full)
                         for lag in tqdm(lags, desc="    all words")])
    rsa_full = lag_window_average(rsa_full, smooth_pts)

    # ---- One curve per random draw.
    post = (lags >= PEAK_WINDOW[0]) & (lags <= PEAK_WINDOW[1])
    curves = np.full((args.n_seeds, len(lags)), np.nan)
    peaks = np.full(args.n_seeds, np.nan)
    n_kept = np.zeros(args.n_seeds, dtype=np.int64)
    for sd in range(args.n_seeds):
        rng = np.random.default_rng(sd)
        keep = np.sort(np.array([rng.choice(v) for v in groups],
                                dtype=np.int64))
        llm_sub = llm.cosine_rdm(emb[keep])
        y = np.array([spearman(extractor.rdm(int(lag), keep), llm_sub)
                      for lag in lags])
        curves[sd] = lag_window_average(y, smooth_pts)
        peaks[sd] = lags[post][int(np.nanargmax(curves[sd][post]))]
        n_kept[sd] = len(keep)
        print(f"    seed {sd:>2}: n={len(keep)}  peak {peaks[sd]:>4.0f} ms  "
              f"rho={np.nanmax(curves[sd][post]):.4f}")

    mean_curve = np.nanmean(curves, axis=0)
    ok = np.isfinite(mean_curve) & np.isfinite(rsa_full)
    shape_r = float(np.corrcoef(mean_curve[ok], rsa_full[ok])[0, 1])
    full_peak = int(lags[post][int(np.nanargmax(rsa_full[post]))])
    mean_peak = int(lags[post][int(np.nanargmax(mean_curve[post]))])

    print(f"\n  all words       : peak {full_peak} ms  "
          f"rho = {np.nanmax(rsa_full[post]):.4f}")
    print(f"  dedup mean curve: peak {mean_peak} ms  "
          f"rho = {np.nanmax(mean_curve[post]):.4f}")
    print(f"  per-draw peaks  : {peaks.min():.0f}-{peaks.max():.0f} ms "
          f"(median {np.median(peaks):.0f})")
    print(f"  curve shape r   : {shape_r:.4f}")
    print(f"  word pairs      : "
          f"{(n_words * (n_words - 1)) / (n_kept[0] * (n_kept[0] - 1)):.1f}-fold fewer")

    out = save_result(
        f"brain_llm_rsa_dedup/{args.roi}_{args.model}",
        eval_lags_ms=lags,
        rsa_full=rsa_full,
        curves=curves,
        peaks=peaks.astype(np.int64),
        n_kept=n_kept,
        n_words=np.int64(n_words),
        n_types=np.int64(len(groups)),
        n_seeds=np.int64(args.n_seeds),
        peak_window_ms=np.array(PEAK_WINDOW, dtype=np.int64),
        full_peak_ms=np.int64(full_peak),
        mean_peak_ms=np.int64(mean_peak),
        shape_r=np.float64(shape_r),
        layer=np.int64(layer),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
    )
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
