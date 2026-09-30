"""Confidence intervals on the brain-LLM peak lag, by subsampling words.

Figure 5 reports the best-matching lag rising from 295 ms at a two-word context
to 360 ms at the full narrative. That shift is an argmax taken from smoothed,
noisy curves, and the paper quotes it without an interval: the monotonicity test
(Spearman rho over context length) shows the ordering is reliable, not that any
individual peak lag is resolvable to a millisecond.

This resamples words and reports the spread of every peak lag, plus the spread
of the 2-word -> full-narrative shift, which is the quantity the claim rests on.

Why subsampling and not the bootstrap
-------------------------------------
The usual bootstrap draws n words from n with replacement. That is invalid here.
A duplicated word sits at distance 0 from its own copy in BOTH the brain RDM and
the model RDM, so every duplicate injects a pair that is perfectly concordant by
construction and inflates rho. Drawing 5,136 from 5,136 duplicates about 37% of
the words, which is not a small perturbation.

Instead this draws a fraction of the words WITHOUT replacement (m-out-of-n
subsampling). No degenerate pairs, and the sampling distribution of the peak lag
is still what we want. Intervals from subsampling are conservative relative to a
valid bootstrap, which is the right direction to err for a robustness claim.

Settings mirror analyses/rsa_peak_by_layer.py, the analysis the published
295 -> 360 numbers come from: layer 25, +-100 ms activation window, 1 ms lag
steps off the 1 ms brain cache, +-10 ms lag smoothing, no winsorization, no
baseline.

Output: results/peak_lag_subsample_ci/<roi>_<model>.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import load_config, make_argparser, save_result
from mind_in_context.lib.rdm import (DEVICE, USE_GPU, compute_rdm_condensed_gpu,
                                     lag_window_average, rank_centered_gpu,
                                     spearman_two_ranked)

LAYER = 25
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 10
# The published peaks span 295-360 ms. The window leaves margin on both sides
# plus room for the smoothing kernel; replicates whose argmax lands on an edge
# are counted and reported rather than silently accepted.
LAG_START_MS, LAG_END_MS, LAG_STEP_MS = 240, 410, 1
N_REPLICATES = 200
FRACTION = 0.8
RNG_SEED = 0


def _cosine_rdm_gpu(emb):
    """lib.llm.cosine_rdm on the GPU: centre columns, L2-normalise rows,
    cosine distance. After normalisation cosine(x, y) = 1 - x.y, so the whole
    RDM is one matmul. scipy's pdist would be far too slow to run thousands of
    times."""
    if not USE_GPU:
        return llm.cosine_rdm(emb)
    import torch
    x = torch.as_tensor(np.ascontiguousarray(emb), dtype=torch.float32,
                        device=DEVICE)
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    gram = x @ x.T
    iu = torch.triu_indices(gram.shape[0], gram.shape[0], offset=1,
                            device=DEVICE)
    return (1.0 - gram[iu[0], iu[1]]).contiguous()


def _check_cosine_helper(emb, n=400, tol=2e-4):
    """The GPU path must agree with the reference before it is trusted."""
    ref = llm.cosine_rdm(emb[:n].astype(np.float64))
    got = _cosine_rdm_gpu(emb[:n])
    got = got.cpu().numpy() if hasattr(got, "cpu") else np.asarray(got)
    err = float(np.max(np.abs(ref - got)))
    print(f"  GPU cosine RDM vs lib.llm.cosine_rdm on {n} words: "
          f"max|diff| = {err:.2e}")
    if err > tol:
        raise SystemExit(f"  GPU cosine RDM disagrees by {err:.2e} (> {tol})")


def _peaks(lags, curves, smooth_pts):
    """argmax per context after lag-axis smoothing. Returns (peaks, hit_edge)."""
    out = np.empty(curves.shape[0], dtype=np.int64)
    edge = False
    for c in range(curves.shape[0]):
        sm = lag_window_average(curves[c], smooth_pts)
        i = int(np.nanargmax(sm))
        edge |= i in (0, len(lags) - 1)
        out[c] = lags[i]
    return out, edge


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n-replicates", type=int, default=N_REPLICATES)
    parser.add_argument("--fraction", type=float, default=FRACTION,
                        help="Fraction of words kept per replicate (no "
                             "replacement). Default 0.8.")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--lag-start", type=int, default=LAG_START_MS)
    parser.add_argument("--lag-end", type=int, default=LAG_END_MS)
    args = parser.parse_args()

    s = load_config()["shared"]
    contexts = list(s["context_lengths"])
    lags = np.arange(args.lag_start, args.lag_end + 1, LAG_STEP_MS, dtype=int)
    smooth_pts = max(1, SMOOTH_HALF_WINDOW_MS // LAG_STEP_MS)

    extractor = BaselineExtractor(
        roi_key=args.roi, eval_lags_ms=lags, half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=None, baseline_window_ms=None, cache="1ms")
    n_words = extractor.n_words
    m = int(round(args.fraction * n_words))
    print(f"  ROI {args.roi}: {extractor.n_electrodes} contacts, "
          f"{n_words} words, layer {args.layer}")
    print(f"  lags {lags[0]}..{lags[-1]} step {LAG_STEP_MS} ms (n={len(lags)}), "
          f"smoothing +-{SMOOTH_HALF_WINDOW_MS} ms")
    print(f"  {args.n_replicates} replicates, {m} of {n_words} words kept "
          f"({100 * args.fraction:.0f}%, no replacement), seed {args.seed}")

    # ---- Embeddings held once; each replicate only re-centres its own subset.
    print(f"\n  loading {len(contexts)} context embedding sets ...")
    emb = {}
    for c in tqdm(contexts, desc="    embeddings", leave=False):
        e = np.load(llm.embeddings_path(args.model, args.layer, c))
        emb[c] = np.ascontiguousarray(e[:n_words], dtype=np.float32)
    total_mb = sum(v.nbytes for v in emb.values()) / 1e6
    print(f"    held in RAM: {total_mb:.0f} MB")
    _check_cosine_helper(emb[contexts[-1]])

    def one_pass(word_idx):
        """Peak lag per context for one word set."""
        ranks = {c: rank_centered_gpu(_cosine_rdm_gpu(emb[c][word_idx]))
                 for c in contexts}
        curves = np.full((len(contexts), len(lags)), np.nan)
        for k, lag in enumerate(lags):
            act = extractor.activity(int(lag))[word_idx]
            rb = rank_centered_gpu(compute_rdm_condensed_gpu(act))
            for ci, c in enumerate(contexts):
                curves[ci, k] = spearman_two_ranked(rb, ranks[c])
            del rb
        del ranks
        return _peaks(lags, curves, smooth_pts)

    # ---- Full sample first, so the point estimate uses this exact pipeline.
    print("\n  full sample ...")
    full_peaks, full_edge = one_pass(np.arange(n_words))
    for c, p in zip(contexts, full_peaks):
        print(f"    context {str(c):>5}: {p} ms")
    if full_edge:
        print("  [warn] a full-sample peak sits on the lag-window edge")

    # ---- Replicates.
    rng = np.random.default_rng(args.seed)
    peaks = np.full((args.n_replicates, len(contexts)), -1, dtype=np.int64)
    n_edge = 0
    out_path = None
    for b in tqdm(range(args.n_replicates), desc="  replicates"):
        idx = np.sort(rng.choice(n_words, size=m, replace=False))
        peaks[b], edge = one_pass(idx)
        n_edge += int(edge)
        if (b + 1) % 25 == 0 or b + 1 == args.n_replicates:
            out_path = save_result(
                f"peak_lag_subsample_ci/{args.roi}_{args.model}",
                context_words=np.array([-1 if c == "full" else int(c)
                                        for c in contexts], dtype=np.int64),
                is_full=np.array([c == "full" for c in contexts]),
                eval_lags_ms=lags,
                peaks=peaks[:b + 1],
                peaks_full_sample=full_peaks,
                n_replicates_done=np.int64(b + 1),
                n_edge_hits=np.int64(n_edge),
                fraction=np.float64(args.fraction),
                n_words=np.int64(n_words),
                n_words_kept=np.int64(m),
                layer=np.int64(args.layer),
                half_window_ms=np.int64(HALF_WINDOW_MS),
                smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
                rng_seed=np.int64(args.seed),
            )

    # ---- Report.
    lo = np.percentile(peaks, 2.5, axis=0)
    hi = np.percentile(peaks, 97.5, axis=0)
    med = np.median(peaks, axis=0)
    print(f"\n  {'context':>8}{'full sample':>13}{'median':>9}"
          f"{'95% interval':>18}{'width':>8}")
    for ci, c in enumerate(contexts):
        print(f"  {str(c):>8}{full_peaks[ci]:>10} ms{med[ci]:>9.0f}"
              f"{lo[ci]:>11.0f}-{hi[ci]:<6.0f}{hi[ci] - lo[ci]:>6.0f} ms")

    shift = peaks[:, -1] - peaks[:, 0]          # full narrative minus 2 words
    s_lo, s_hi = np.percentile(shift, [2.5, 97.5])
    print(f"\n  shift (2 words -> full): point estimate "
          f"{full_peaks[-1] - full_peaks[0]} ms, "
          f"median {np.median(shift):.0f} ms, 95% interval "
          f"{s_lo:.0f}-{s_hi:.0f} ms")
    print(f"  replicates whose shift is > 0: "
          f"{100 * np.mean(shift > 0):.1f}%")
    if n_edge:
        print(f"  [warn] {n_edge} of {args.n_replicates} replicates had a peak "
              f"on the lag-window edge; widen --lag-start/--lag-end")
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
