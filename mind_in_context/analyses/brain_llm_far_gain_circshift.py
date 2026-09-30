"""Far-pair gain from an early to a late window, against a circular-shift null.

Fig 6 reports that the brain-LLM correspondence gained between an early and a
late post-onset window is carried by word pairs that are FAR apart in the
narrative. This analysis provides the significance for that statement.

The statistic
    gain(pair)  = correspondence(late window) - correspondence(early window)
    FAR GAIN    = mean of gain over word pairs 101-1,000 words apart

Why those pairs. Words that are neighbours in the narrative contribute to the
correspondence at any lag simply because they are close in time: their neural
patterns resemble one another and so do their model vectors. Distant pairs have
no such shortcut, so a gain concentrated there cannot come from temporal
proximity alone.

The null
    The model's word labels are rotated by a random offset (>= --shift-min
    words, up to half the narrative). A rotation preserves word order, and so
    preserves the autocorrelation of both signals and the near-diagonal ridge
    it produces, while destroying the word-specific brain-to-model
    correspondence. Under rotation the far-distance contribution is absent at
    BOTH windows, so the null far gain sits at zero.

    A t-test is not usable here: RDM entries are not independent observations,
    since each of the words enters thousands of pairs and every pair therefore
    shares a word - and that word's measurement noise - with thousands of
    others.

    NEAR GAIN (1-100) is saved alongside as a reference quantity; it does not
    separate real from rotated in the same way, because neighbouring pairs
    contribute under rotation too.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
      --early-window <lo> <hi>   default 50 150   (ms, inclusive)
      --late-window  <lo> <hi>   default 350 450  (ms, inclusive)
      --n-shifts <int>           default 1000
      --shift-min <int>          default 600 words, as in the Fig 4A control
      --dist-max <int>           default 1000
      --near-max <int>           default 100  (near = 1..near-max)
      --seed <int>               default 0

Output: results/brain_llm_far_gain_circshift/<roi>_<model>.npz
Used by: figures/fig6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.distance import squareform

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result

HALF_WINDOW_MS = 100        # matches brain_llm_correspondence.py
EARLY_WINDOW_MS = (50, 150)
LATE_WINDOW_MS = (350, 450)
DIST_MAX = 1000
NEAR_MAX = 100
N_SHIFTS = 1000
SHIFT_MIN_WORDS = 600


def _window_lags(lo: int, hi: int, step: int) -> list[int]:
    """Every evaluation lag inside an inclusive window."""
    lags = list(range(int(lo), int(hi) + 1, int(step)))
    if not lags:
        raise SystemExit(f"window {lo}..{hi} ms contains no {step} ms lag")
    return lags


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--early-window", type=int, nargs=2,
                        default=EARLY_WINDOW_MS, metavar=("LO", "HI"))
    parser.add_argument("--late-window", type=int, nargs=2,
                        default=LATE_WINDOW_MS, metavar=("LO", "HI"))
    parser.add_argument("--n-shifts", type=int, default=N_SHIFTS)
    parser.add_argument("--shift-min", type=int, default=SHIFT_MIN_WORDS)
    parser.add_argument("--dist-max", type=int, default=DIST_MAX)
    parser.add_argument("--near-max", type=int, default=NEAR_MAX)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    s = load_config()["shared"]
    step = int(s["lag_step_ms"])
    early_lags = _window_lags(*args.early_window, step)
    late_lags = _window_lags(*args.late_window, step)
    print(f"  early {args.early_window[0]}-{args.early_window[1]} ms: {early_lags}")
    print(f"  late  {args.late_window[0]}-{args.late_window[1]} ms: {late_lags}")

    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=early_lags + late_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    n_words = extractor.n_words
    print(f"  electrodes: {extractor.n_electrodes},  words: {n_words}")

    # Brain side: one z-scored RDM per lag, averaged within each window. The
    # LLM RDM is shared by both windows, so the gain factors as
    # (z_late - z_early) * z_llm and no brain RDM is recomputed per shift.
    z = {}
    for i, lag in enumerate(early_lags + late_lags, 1):
        rdm = extractor.rdm(int(lag))
        z[lag] = ((rdm - rdm.mean()) / rdm.std()).astype(np.float32)
        print(f"  [{i}/{len(z)}] brain RDM at {lag} ms")
    z_early = np.mean([z[l] for l in early_lags], axis=0).astype(np.float32)
    z_late = np.mean([z[l] for l in late_lags], axis=0).astype(np.float32)
    del z

    layer = load_config()["models"][args.model]["layer"]
    emb_path = llm.full_context_embeddings_path(args.model, layer)
    print(f"  loading LLM embeddings: {emb_path.name}")
    emb = np.load(emb_path).astype(np.float64)[:n_words]
    llm_rdm = llm.cosine_rdm(emb)
    z_llm = ((llm_rdm - llm_rdm.mean()) / llm_rdm.std()).astype(np.float32)
    llm_square = squareform(z_llm).astype(np.float32)

    # Condensed pair indices, restricted to the distances that are reported, so
    # a rotation is one fancy-index gather rather than a matrix rebuild.
    rows, cols = np.triu_indices(n_words, k=1)
    dist = (cols - rows).astype(np.int32)
    keep = dist <= args.dist_max
    rows = rows[keep].astype(np.int16)
    cols = cols[keep].astype(np.int16)
    dist = dist[keep]
    counts = np.bincount(dist, minlength=args.dist_max + 1)[1:].astype(np.float64)
    d_brain = (z_late - z_early)[keep]
    print(f"  pairs within distance {args.dist_max}: {dist.size:,}")

    near = slice(0, args.near_max)
    far = slice(args.near_max, args.dist_max)

    def gain_profile(llm_condensed: np.ndarray) -> np.ndarray:
        """Mean early-to-late gain at every word-pair distance."""
        weighted = (d_brain * llm_condensed).astype(np.float64)
        return np.bincount(dist, weights=weighted,
                           minlength=args.dist_max + 1)[1:] / counts

    gain_real = gain_profile(llm_square[rows, cols])
    far_real = float(gain_real[far].mean())
    near_real = float(gain_real[near].mean())
    print(f"  real: far gain {far_real:+.5f}, near gain {near_real:+.5f}")

    rng = np.random.default_rng(args.seed)
    order = np.arange(n_words)
    shift_max = n_words // 2
    sizes = rng.integers(args.shift_min, shift_max + 1, size=args.n_shifts)
    null_far = np.empty(args.n_shifts)
    null_near = np.empty(args.n_shifts)
    for i, size in enumerate(sizes):
        rotated = ((order + int(size)) % n_words).astype(np.int16)
        profile = gain_profile(llm_square[rotated[rows], rotated[cols]])
        null_far[i] = profile[far].mean()
        null_near[i] = profile[near].mean()
        if (i + 1) % 50 == 0:
            print(f"  shift {i + 1}/{args.n_shifts}")

    p_far = (np.sum(null_far >= far_real) + 1) / (args.n_shifts + 1)
    p_near = (np.sum(null_near >= near_real) + 1) / (args.n_shifts + 1)
    print(f"  far gain p = {p_far:.4f}   (null {null_far.mean():+.5f} "
          f"± {null_far.std(ddof=1):.5f}, max {null_far.max():+.5f})")

    out = save_result(
        f"brain_llm_far_gain_circshift/{args.roi}_{args.model}",
        distances=np.arange(1, args.dist_max + 1, dtype=np.int64),
        gain_real=gain_real,
        far_real=np.float64(far_real),
        near_real=np.float64(near_real),
        null_far=null_far,
        null_near=null_near,
        shift_sizes=sizes.astype(np.int64),
        p_far=np.float64(p_far),
        p_near=np.float64(p_near),
        early_window_ms=np.array(args.early_window, dtype=np.int64),
        late_window_ms=np.array(args.late_window, dtype=np.int64),
        near_max=np.int64(args.near_max),
        dist_max=np.int64(args.dist_max),
        n_shifts=np.int64(args.n_shifts),
        shift_min_words=np.int64(args.shift_min),
        shift_max_words=np.int64(shift_max),
        seed=np.int64(args.seed),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
