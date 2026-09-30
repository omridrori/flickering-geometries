"""Brain-LLM correspondence by word-pair distance, for a ladder of time windows.

Figure 6 contrasts two windows. This analysis walks a series of consecutive
windows across the post-onset epoch and reports, for each, the mean
correspondence at every word-pair distance. It answers a question the
two-window contrast cannot: does the far-pair contribution grow steadily
through the epoch, or does it rise and then settle?

For each window the z-scored brain RDMs of its lags are averaged, multiplied
element-wise by the z-scored LLM RDM, and the products pooled by the narrative
distance between the two words of each pair - the same decomposition used for
Figures 4B and 6.

Consecutive windows share their endpoint lag by default (150 belongs to both
50-150 and 150-250), which is one lag in five; pass disjoint bounds if that
matters for a given use.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}
      --windows LO-HI [LO-HI ...]   default 50-150 150-250 250-350 350-450
      --dist-max <int>              default 1000
      --near-max <int>              default 100  (near = 1..near-max)

Output: results/brain_llm_distance_by_window/<roi>_<model>.npz
Used by: figures/figS_distance_by_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result

HALF_WINDOW_MS = 100        # matches brain_llm_correspondence.py
WINDOWS = ["50-150", "150-250", "250-350", "350-450"]
DIST_MAX = 1000
NEAR_MAX = 100


def _parse_window(text: str) -> tuple[int, int]:
    try:
        lo, hi = (int(v) for v in text.split("-"))
    except ValueError:
        raise SystemExit(f"window {text!r} is not of the form LO-HI") from None
    if hi <= lo:
        raise SystemExit(f"window {text!r} does not run forwards")
    return lo, hi


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--windows", nargs="+", default=WINDOWS, metavar="LO-HI")
    parser.add_argument("--dist-max", type=int, default=DIST_MAX)
    parser.add_argument("--near-max", type=int, default=NEAR_MAX)
    args = parser.parse_args()

    s = load_config()["shared"]
    step = int(s["lag_step_ms"])
    windows = [_parse_window(w) for w in args.windows]
    per_window = {w: list(range(w[0], w[1] + 1, step)) for w in windows}
    for w, lags in per_window.items():
        if not lags:
            raise SystemExit(f"window {w[0]}-{w[1]} ms contains no {step} ms lag")
        print(f"  {w[0]}-{w[1]} ms: {lags}")
    all_lags = sorted({l for lags in per_window.values() for l in lags})

    extractor = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=all_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    n_words = extractor.n_words
    print(f"  electrodes: {extractor.n_electrodes},  words: {n_words}")

    z_brain = {}
    for i, lag in enumerate(all_lags, 1):
        rdm = extractor.rdm(int(lag))
        z_brain[lag] = ((rdm - rdm.mean()) / rdm.std()).astype(np.float32)
        print(f"  [{i}/{len(all_lags)}] brain RDM at {lag} ms")

    layer = load_config()["models"][args.model]["layer"]
    emb_path = llm.full_context_embeddings_path(args.model, layer)
    print(f"  loading LLM embeddings: {emb_path.name}")
    emb = np.load(emb_path).astype(np.float64)[:n_words]
    llm_rdm = llm.cosine_rdm(emb)
    z_llm = ((llm_rdm - llm_rdm.mean()) / llm_rdm.std()).astype(np.float32)

    # Pool by narrative distance. Pairs beyond dist_max are dropped up front so
    # the per-window product stays a single pass over the condensed vector.
    rows, cols = np.triu_indices(n_words, k=1)
    dist = (cols - rows).astype(np.int32)
    keep = dist <= args.dist_max
    dist = dist[keep]
    z_llm = z_llm[keep]
    counts = np.bincount(dist, minlength=args.dist_max + 1)[1:].astype(np.float64)
    del rows, cols
    print(f"  pairs within distance {args.dist_max}: {dist.size:,}")

    curves = np.empty((len(windows), args.dist_max))
    for k, w in enumerate(windows):
        brain = np.mean([z_brain[l] for l in per_window[w]],
                        axis=0).astype(np.float32)[keep]
        weighted = (brain * z_llm).astype(np.float64)
        curves[k] = np.bincount(dist, weights=weighted,
                                minlength=args.dist_max + 1)[1:] / counts
        print(f"  {w[0]}-{w[1]} ms: near {curves[k][:args.near_max].mean():+.5f}, "
              f"far {curves[k][args.near_max:].mean():+.5f}")
        del brain, weighted

    out = save_result(
        f"brain_llm_distance_by_window/{args.roi}_{args.model}",
        distances=np.arange(1, args.dist_max + 1, dtype=np.int64),
        curves=curves,
        windows_ms=np.array(windows, dtype=np.int64),
        near_means=curves[:, :args.near_max].mean(axis=1),
        far_means=curves[:, args.near_max:].mean(axis=1),
        near_max=np.int64(args.near_max),
        dist_max=np.int64(args.dist_max),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
