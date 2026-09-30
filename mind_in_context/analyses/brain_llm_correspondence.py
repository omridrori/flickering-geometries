"""Element-wise brain × LLM correspondence at a single lag - Figure 4B-C, Figure 6.

correspondence(i, j) = z_brain(i, j) × z_llm(i, j)

Both RDMs are z-scored across their condensed entries (off-diagonal pairs)
before multiplication. Positive cells = brain and LLM agree the pair is
unusually similar (or unusually dissimilar) together; negative cells = they
disagree.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
      --lag-ms <int>          (default: 200)
      --lags-ms <int> ...     (batch: compute and save many lags in one run;
                               builds the LLM RDM once and reuses the brain
                               windowing slab - ~5× faster than looping the
                               script externally)
      --zoom-start <word_idx> (default: 1467)
      --zoom-end   <word_idx> (default: 1507)
      --shift-words <int>     (default: 0; >0 => circularly shift the brain
                               word sequence by N words before building the
                               brain RDM - the same control used in Fig 4A's
                               gray line)
      --permute-llm-seed <int>(if set, randomly permute the LLM word labels
                               with this seed before z-scoring; output is
                               suffixed `_permllm<seed>`. The same operation
                               the permutation test uses.)

Output: results/brain_llm_correspondence/<roi>_<model>_lag<L>ms.npz
  full_matrix : (n_words, n_words) - about 100 MB, so not included in the repository.
  zoom_matrix : (40, 40)
  zoom_start, zoom_end : ints
  z_brain_mean / std, z_llm_mean / std : for reference
Used by: figures/fig4.py, figures/fig6.py
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

LAG_MS = 200
HALF_WINDOW_MS = 100
ZOOM_START = 1467
ZOOM_END = 1507


def _save_correspondence(*, roi: str, model: str, lag_ms: int,
                         shift_words: int, permute_llm_seed: int | None,
                         smooth_half_window_ms: int,
                         random_shifts: int = 0,
                         shift_range: tuple[int, int] = (0, 0),
                         shift_seed: int = 0,
                         correspondence_condensed: np.ndarray,
                         z_brain_mean: float, z_brain_std: float,
                         z_llm_mean: float, z_llm_std: float,
                         zoom_start: int, zoom_end: int) -> None:
    """Save a (already-built) condensed correspondence vector as .npz."""
    full_matrix = squareform(correspondence_condensed.astype(np.float32))
    zoom_matrix = full_matrix[zoom_start:zoom_end,
                              zoom_start:zoom_end].copy()

    if permute_llm_seed is not None:
        suffix = f"_permllm{permute_llm_seed}"
    elif random_shifts > 0:
        suffix = (f"_shiftrand_n{random_shifts}"
                  f"_range{shift_range[0]}-{shift_range[1]}"
                  f"_seed{shift_seed}")
    elif shift_words:
        suffix = f"_shift{shift_words}"
    else:
        suffix = ""
    if smooth_half_window_ms > 0:
        suffix = f"{suffix}_smooth{smooth_half_window_ms}"
    out = save_result(
        f"brain_llm_correspondence/{roi}_{model}_lag{lag_ms}ms{suffix}",
        full_matrix=full_matrix,
        zoom_matrix=zoom_matrix,
        zoom_start=np.int64(zoom_start),
        zoom_end=np.int64(zoom_end),
        lag_ms=np.int64(lag_ms),
        shift_words=np.int64(shift_words),
        permute_llm_seed=np.int64(-1 if permute_llm_seed is None
                                  else permute_llm_seed),
        smooth_half_window_ms=np.int64(smooth_half_window_ms),
        random_shifts=np.int64(random_shifts),
        shift_range_min=np.int64(shift_range[0]),
        shift_range_max=np.int64(shift_range[1]),
        shift_seed=np.int64(shift_seed),
        z_brain_mean=np.float64(z_brain_mean),
        z_brain_std=np.float64(z_brain_std),
        z_llm_mean=np.float64(z_llm_mean),
        z_llm_std=np.float64(z_llm_std),
    )
    print(f"  -> {out}")


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--lag-ms", type=int, default=LAG_MS)
    parser.add_argument("--lags-ms", type=int, nargs="+", default=None,
                        help="Batch: run multiple lags in one process. Overrides --lag-ms.")
    parser.add_argument("--zoom-start", type=int, default=ZOOM_START)
    parser.add_argument("--zoom-end", type=int, default=ZOOM_END)
    parser.add_argument("--shift-words", type=int, default=0,
                        help="Circular shift of brain word sequence (0 = none).")
    parser.add_argument("--permute-llm-seed", type=int, default=None,
                        help="If set, permute LLM word labels with this seed "
                             "before z-scoring (single permutation realization).")
    parser.add_argument("--smooth-half-window-ms", type=int, default=0,
                        help="If > 0, compute correspondence at every lag in "
                             "[center-N, center+N] in 25-ms steps and average. "
                             "Output suffix `_smooth<N>`.")
    parser.add_argument("--random-shifts", type=int, default=0,
                        help="If > 0, sample N random circular shifts and "
                             "average the resulting correspondences. Overrides "
                             "--shift-words.")
    parser.add_argument("--shift-range", type=int, nargs=2,
                        default=(600, 2000), metavar=("MIN", "MAX"),
                        help="Min and max word offsets for random shifts.")
    parser.add_argument("--shift-seed", type=int, default=0,
                        help="RNG seed for --random-shifts.")
    args = parser.parse_args()

    s = load_config()["shared"]
    lags = args.lags_ms if args.lags_ms is not None else [args.lag_ms]
    lag_step = int(s["lag_step_ms"])
    hw = int(args.smooth_half_window_ms)
    if hw and hw % lag_step:
        raise SystemExit(f"--smooth-half-window-ms ({hw}) must be a multiple "
                         f"of the cache step {lag_step} ms.")
    sub_offsets = (list(range(-hw, hw + 1, lag_step)) if hw > 0 else [0])
    all_lags = sorted({c + d for c in lags for d in sub_offsets})
    print(f"  ROI: {args.roi}, model: {args.model}, center lags: {lags} ms, "
          f"smooth ±{hw} ms ({len(sub_offsets)} sub-lags each)")

    # Single BaselineExtractor over ALL requested lags - the brain windowing
    # slab is shared and only the per-lag pdist is recomputed.
    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=all_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    print(f"  electrodes: {extractor.n_electrodes},  words: {extractor.n_words}")

    # LLM RDM is lag-independent - build it once and cache z_llm too.
    layer = load_config()["models"][args.model]["layer"]
    emb_path = llm.full_context_embeddings_path(args.model, layer)
    print(f"  loading LLM embeddings: {emb_path.name}")
    emb = np.load(emb_path).astype(np.float64)
    n_words = min(emb.shape[0], extractor.n_words)
    emb = emb[:n_words]
    print("  building LLM RDM (cosine on centered+L2-normed) ...")
    llm_rdm = llm.cosine_rdm(emb)
    if args.permute_llm_seed is not None:
        rng = np.random.default_rng(args.permute_llm_seed)
        pi = rng.permutation(n_words)
        print(f"  permuting LLM RDM word labels (seed={args.permute_llm_seed})")
        llm_sq = squareform(llm_rdm)
        llm_rdm = squareform(llm_sq[pi][:, pi], checks=False)
    z_llm = (llm_rdm - llm_rdm.mean()) / llm_rdm.std()

    # Build the list of (shift_perm, label) tuples to average over.
    if args.random_shifts > 0:
        rng = np.random.default_rng(args.shift_seed)
        sampled = rng.integers(args.shift_range[0],
                               args.shift_range[1] + 1,
                               size=args.random_shifts)
        shift_perms = [((np.arange(extractor.n_words) + int(s))
                        % extractor.n_words, int(s)) for s in sampled]
        print(f"  random shifts (n={args.random_shifts}, "
              f"range {args.shift_range[0]}..{args.shift_range[1]}, "
              f"seed {args.shift_seed}): {sampled.tolist()}")
    elif args.shift_words:
        sp = (np.arange(extractor.n_words) + args.shift_words
              ) % extractor.n_words
        shift_perms = [(sp, args.shift_words)]
    else:
        shift_perms = [(None, 0)]

    z_llm_mean = float(np.mean(llm_rdm))
    z_llm_std = float(np.std(llm_rdm))
    for center in lags:
        sub_lags = [center + d for d in sub_offsets]
        if hw > 0:
            print(f"  center lag {center} ms  <-  avg over {sub_lags} ms")
        per_shift_corr = []
        zb_means, zb_stds = [], []
        for shift_perm, shift_lbl in shift_perms:
            acc = None
            for sl in sub_lags:
                if shift_perm is not None:
                    brain_rdm = extractor.rdm(int(sl), shift_perm)
                else:
                    brain_rdm = extractor.rdm(int(sl))
                zb_means.append(float(brain_rdm.mean()))
                zb_stds.append(float(brain_rdm.std()))
                z_brain = (brain_rdm - zb_means[-1]) / zb_stds[-1]
                corr = z_brain * z_llm
                acc = corr.copy() if acc is None else acc + corr
            per_shift_corr.append(acc / len(sub_lags))
        correspondence = np.mean(np.stack(per_shift_corr, axis=0), axis=0)
        _save_correspondence(
            roi=args.roi, model=args.model, lag_ms=int(center),
            shift_words=args.shift_words,
            permute_llm_seed=args.permute_llm_seed,
            smooth_half_window_ms=hw,
            random_shifts=args.random_shifts,
            shift_range=tuple(args.shift_range),
            shift_seed=args.shift_seed,
            correspondence_condensed=correspondence,
            z_brain_mean=float(np.mean(zb_means)),
            z_brain_std=float(np.mean(zb_stds)),
            z_llm_mean=z_llm_mean, z_llm_std=z_llm_std,
            zoom_start=args.zoom_start, zoom_end=args.zoom_end,
        )


if __name__ == "__main__":
    main()
