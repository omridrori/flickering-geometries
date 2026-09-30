"""Mean GPT-2-XL surprisal per chronological narrative segment.

Checks that the surprisal split of Fig. 3 is not confounded with narrative
position: if surprisal drifted over the podcast, the low- and high-surprisal
quintiles would sample different parts of the story and the surprisal effect
could be a by-product of the context-length effect of Fig. 2.

Words are cut into `--n-segments` chronological segments of equal word count
(lib.segments.eqword_segments, the same partition as the Fig. 2 scatter plots).
Per segment: mean surprisal and s.e.m. across its words. Trend: Pearson r of
segment mean against segment position, tested by shuffling segment order
(lib.stats.trend_perm_test, as for the Fig. 2 trends).

CLI:  --n-segments N (default 20)
Output: results/surprisal_by_segment/seg<N>.npz
Used by: figures/figS_surprisal_by_segment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib.io import make_argparser, save_result
from mind_in_context.lib.segments import eqword_segments
from mind_in_context.lib.stats import trend_perm_test
from mind_in_context.lib.surprisal import word_surprisal

N_PERM = 10_000
RNG_SEED = 0


def main() -> None:
    parser = make_argparser(__doc__)
    parser.add_argument("--n-segments", type=int, default=20)
    args = parser.parse_args()

    surprisal = word_surprisal()
    n_words = int(np.sum(~np.isnan(surprisal)))
    groups, time_edges = eqword_segments(n_words, args.n_segments)

    seg_mean = np.array([np.nanmean(surprisal[g]) for g in groups])
    seg_sem = np.array([np.nanstd(surprisal[g], ddof=1) / np.sqrt(len(g))
                        for g in groups])
    seg_n = np.array([len(g) for g in groups])
    r, p_two, p_one = trend_perm_test(seg_mean, N_PERM, RNG_SEED)

    print(f"  {n_words} words, {args.n_segments} segments, "
          f"overall mean {np.nanmean(surprisal):.3f} bits")
    print(f"  segment means {np.round(seg_mean, 2)}")
    print(f"  trend vs position: r = {r:+.3f}, two-sided p = {p_two:.3f} "
          f"({N_PERM} permutations)")
    out = save_result(
        f"surprisal_by_segment/seg{args.n_segments}",
        seg_mean=seg_mean, seg_sem=seg_sem, seg_n=seg_n,
        segment_time_edges=time_edges, overall_mean=np.float64(np.nanmean(surprisal)),
        r=np.float64(r), p_perm_two_sided=np.float64(p_two),
        p_perm_one_sided=np.float64(p_one), n_perm=np.int64(N_PERM),
        rng_seed=np.int64(RNG_SEED), n_words=np.int64(n_words),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
