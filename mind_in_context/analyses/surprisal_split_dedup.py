"""Surprisal split on one occurrence of each word form - Extended Data Fig. 4.

The narrative repeats: 5,136 word tokens are only 1,252 distinct forms, and the
predictable (low-surprisal) set is dominated by high-frequency function words
that recur throughout. That raises the objection that the surprisal effect in
Figure 3 is really a repetition effect. This control removes the possibility
entirely by keeping the FIRST occurrence of each word form and discarding every
later repeat, so no word contributes twice.

The surviving forms are ranked by GPT-2-XL surprisal and split at the median
into two balanced halves, and rho_ERG is recomputed for all three sets with the
same settings as analyses/surprisal_split.py.

Surprisal source: stimuli/gpt2-xl/transcript.tsv 'true_prob' column.
Per-word surprisal = sum over its sub-tokens of -log2 p(token | preceding).

CLI:  --roi {lang,aud}
Output: results/surprisal_split_dedup/<roi>.npz
Used by: figures/figS_surprisal_dedup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mind_in_context.analyses.surprisal_split import DELTA_MS
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (
    GPT2XL_TRANSCRIPT_TSV, TRANSCRIPT_TSV, load_config, make_argparser,
    save_result,
)
from mind_in_context.lib.rdm import compute_grouped_erg

# Punctuation stripped before comparing word forms, so "word." and "word"
# count as the same form.
STRIP = ".,!?;:\"'()[]"
DIP_SEARCH = (0, 500)


def _first_occurrences(n_words: int) -> pd.DataFrame:
    """One row per distinct word form, at the index where it first appears."""
    df = pd.read_csv(TRANSCRIPT_TSV, sep="\t")
    df["word_idx"] = df["word_idx"].astype(int)
    w = (df.groupby("word_idx").agg(word=("word", "first"))
           .reset_index().sort_values("word_idx"))
    w = w[w["word_idx"] < n_words].copy()
    w["form"] = w["word"].astype(str).str.lower().str.strip(STRIP)
    return w.drop_duplicates("form", keep="first")


def _surprisal_per_word() -> pd.DataFrame:
    g = pd.read_csv(GPT2XL_TRANSCRIPT_TSV, sep="\t")
    g["surprisal"] = -np.log2(g["true_prob"].astype(float).clip(lower=1e-12))
    return g.groupby("word_idx").agg(surprisal=("surprisal", "sum")).reset_index()


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--lag-start", type=int, default=None,
                        help="Override lag range start (default: from config).")
    parser.add_argument("--lag-end", type=int, default=None,
                        help="Override lag range end (default: from config).")
    parser.add_argument("--delta", type=int, default=None,
                        help=f"ERG gap Δ in ms (multiple of the lag step). "
                             f"Defaults to {DELTA_MS} ms, matching Figure 2 and "
                             f"analyses/surprisal_split.py. A non-default value "
                             f"writes to a '_d<delta>' result file.")
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
    n_words = extractor.n_words
    print(f"  electrodes: {extractor.n_electrodes},  words: {n_words}")

    first = _first_occurrences(n_words)
    merged = (first.merge(_surprisal_per_word(), on="word_idx", how="inner")
                   .sort_values("surprisal"))
    print(f"  distinct word forms: {len(first)}  "
          f"({100 * (1 - len(first) / n_words):.1f}% of tokens dropped)")

    median = float(merged["surprisal"].median())
    low = merged[merged["surprisal"] <= median]
    high = merged[merged["surprisal"] > median]
    k = min(len(low), len(high))                 # balance the halves exactly
    low, high = low.nsmallest(k, "surprisal"), high.nlargest(k, "surprisal")
    mean_low = float(low["surprisal"].mean())
    mean_high = float(high["surprisal"].mean())
    print(f"  median surprisal {median:.2f} bits -> {len(low)} low / "
          f"{len(high)} high")
    print(f"  mean surprisal   low {mean_low:.2f} bits   high {mean_high:.2f} bits")

    all_idx = np.sort(merged["word_idx"].to_numpy(dtype=np.int64))
    low_idx = np.sort(low["word_idx"].to_numpy(dtype=np.int64))
    high_idx = np.sort(high["word_idx"].to_numpy(dtype=np.int64))
    assert not set(low_idx) & set(high_idx), "the two halves overlap"

    rho = compute_grouped_erg(extractor, eval_lags, [all_idx, low_idx, high_idx],
                              delta_ms)

    post = (eval_lags >= DIP_SEARCH[0]) & (eval_lags <= DIP_SEARCH[1])
    dip_low = float(np.nanmin(rho[1][post]))
    dip_high = float(np.nanmin(rho[2][post]))
    dip_lag = int(eval_lags[post][int(np.nanargmin(rho[2][post]))])
    print(f"\n  low-surprisal  dip {dip_low:.4f} @ "
          f"{eval_lags[post][int(np.nanargmin(rho[1][post]))]} ms")
    print(f"  high-surprisal dip {dip_high:.4f} @ {dip_lag} ms")
    print(f"  dip separation {dip_low - dip_high:.4f}")

    save_kwargs = dict(
        eval_lags_ms=eval_lags,
        rho_all=rho[0],
        rho_low=rho[1],
        rho_high=rho[2],
        n_words=np.int64(n_words),
        n_all=np.int64(len(all_idx)),
        n_low=np.int64(len(low_idx)),
        n_high=np.int64(len(high_idx)),
        median_surprisal=np.float64(median),
        mean_low=np.float64(mean_low),
        mean_high=np.float64(mean_high),
        separation=np.float64(dip_low - dip_high),
        dip_lag=np.int64(dip_lag),
        dip_search=np.array(DIP_SEARCH, dtype=np.int64),
        baseline_corrected=np.bool_(baseline_window is not None),
        delta_ms=np.int64(delta_ms),
        lag_step_ms=np.int64(lag_step),
    )
    if baseline_window is not None:
        save_kwargs["baseline_window"] = np.array(baseline_window, dtype=np.int64)
    out = save_result(f"surprisal_split_dedup/{args.roi}{delta_tag}", **save_kwargs)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
