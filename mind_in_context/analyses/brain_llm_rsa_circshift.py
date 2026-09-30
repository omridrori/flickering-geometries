"""Circular-shift null for the brain x LLM temporal RSA - Fig 4A control.

`brain_llm_rsa_permutation.py` tests a different null: it scrambles the LLM word
labels completely, which destroys every trace of temporal structure and is
therefore easy to beat. This is the stronger control. Each shuffle slides the
LLM word axis around the narrative by a random signed offset, so word order,
the autocorrelation of the brain signal and the autocorrelation of the
embeddings all survive intact - only the pairing between them is broken.

`brain_llm_rsa.py` already reports one such shift (SHIFT_WORDS = 600) as
`rsa_shifted`. This repeats it N times with random offsets and keeps every
curve, so the null becomes a distribution with a mean and a SEM rather than a
single line.

Settings mirror brain_llm_rsa.py exactly: +-100 ms activity window, winsorize
3 sigma, no baseline correction, the full-context embeddings, cosine
RDM via lib.llm.cosine_rdm, and +-50 ms smoothing along the lag axis.

Two implementation notes, both of which are what make 1000 shuffles cheap:

  Loop order. Lags are the outer loop and shuffles the inner one, so each brain
  RDM is built once, used by every shuffle, then dropped. BaselineExtractor.rdm
  memoises every RDM it returns, which at 321 lags would hold ~17 GB, so RDMs
  are built from extractor.activity() via compute_rdm_condensed_gpu instead -
  the same pattern the other wide-lag analyses use. Memory stays flat.

  Ranking. A circular shift permutes the entries of the RDM without changing
  their multiset, so rank(shift(L)) == shift(rank(L)). The LLM RDM is therefore
  rank-centred ONCE and merely rolled per shuffle, instead of re-ranking 13.2
  million values 1000 times. Because the shift is applied to both axes, the
  diagonal maps onto the diagonal and stays zero, so the condensed (off
  diagonal) Spearman is recovered exactly from the square form.

Shifts are drawn without replacement from the signed offsets whose circular
distance is at least --min-shift, so no shuffle sits near the true alignment.
Note that -k and +(N-k) are the same rotation; the sign is recorded as drawn but
adds no coverage the magnitude alone would not.

Everything needed for later statistics and plotting is saved: the raw and the
smoothed curve for every shuffle, the signed shift each one used, the real
curve on the same lag grid, and all settings.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}  [--n-shuffles 1000]
      [--min-shift 600] [--seed 0] [--lag-start -4000] [--lag-end 4000]
Output: results/brain_llm_rsa_circshift/<roi>_<model>.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (RESULTS_DIR, load_config, load_result,
                                    make_argparser, save_result)
from mind_in_context.lib.rdm import (DEVICE, USE_GPU, compute_rdm_condensed_gpu,
                                     lag_window_average, rank_centered_gpu)

LAG_START_MS = -4000
LAG_END_MS = 4000
LAG_STEP_MS = 25
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 50
N_SHUFFLES = 1000
MIN_SHIFT_WORDS = 600


def _draw_shifts(n_words: int, n_shuffles: int, min_shift: int,
                 seed: int) -> np.ndarray:
    """Signed offsets with circular distance >= min_shift, without replacement.

    Magnitudes run from min_shift to n_words // 2 and carry either sign, which
    covers exactly the residues [min_shift, n_words - min_shift].
    """
    hi = n_words // 2
    if min_shift < 1 or min_shift > hi:
        raise ValueError(f"min_shift must be in [1, {hi}], got {min_shift}")
    mags = np.arange(min_shift, hi + 1, dtype=np.int64)
    pool = np.concatenate([mags, -mags])
    pool = np.unique(pool)
    if n_shuffles > pool.size:
        raise ValueError(
            f"asked for {n_shuffles} shuffles but only {pool.size} distinct "
            f"offsets have circular distance >= {min_shift}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n_shuffles, replace=False))


def _square_ranked(condensed, n: int, iu):
    """Rank-centre a condensed RDM and expand to a symmetric square, diag 0."""
    r = rank_centered_gpu(condensed)
    if USE_GPU:
        import torch
        sq = torch.zeros((n, n), dtype=r.dtype, device=DEVICE)
        sq[iu[0], iu[1]] = r
        sq[iu[1], iu[0]] = r
        return sq, float(torch.sqrt((r ** 2).sum()).item())
    sq = np.zeros((n, n), dtype=np.float64)
    sq[iu[0], iu[1]] = r
    sq[iu[1], iu[0]] = r
    return sq, float(np.sqrt((r ** 2).sum()))


def _roll_both(sq, k: int):
    """Rotate both axes by k - the square form of shifting the word axis."""
    if USE_GPU:
        import torch
        return torch.roll(sq, shifts=(int(k), int(k)), dims=(0, 1))
    return np.roll(np.roll(sq, k, axis=0), k, axis=1)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    parser.add_argument("--min-shift", type=int, default=MIN_SHIFT_WORDS,
                        help="Minimum circular distance in words (default 600, "
                             "matching SHIFT_WORDS in brain_llm_rsa.py).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lag-start", type=int, default=LAG_START_MS)
    parser.add_argument("--lag-end", type=int, default=LAG_END_MS)
    parser.add_argument("--lag-step", type=int, default=LAG_STEP_MS)
    args = parser.parse_args()

    global PARTIAL
    PARTIAL = (RESULTS_DIR / "brain_llm_rsa_circshift" /
               f"{args.roi}_{args.model}_partial.npz")
    PARTIAL.parent.mkdir(parents=True, exist_ok=True)

    s = load_config()["shared"]
    eval_lags = np.arange(args.lag_start, args.lag_end + 1, args.lag_step,
                          dtype=int)
    n_lags = len(eval_lags)
    print(f"  ROI: {args.roi},  model: {args.model},  GPU: {USE_GPU}")
    print(f"  lags [{args.lag_start}, {args.lag_end}] ms step {args.lag_step} "
          f"({n_lags} lags),  half-window +-{HALF_WINDOW_MS} ms")

    ext = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )

    layer = load_config()["models"][args.model]["layer"]
    emb_path = llm.full_context_embeddings_path(args.model, layer)
    print(f"  loading LLM embeddings: {emb_path.name}")
    emb = np.load(emb_path).astype(np.float64)
    n_words = int(min(emb.shape[0], ext.n_words))
    emb = emb[:n_words]
    word_subset = (np.arange(n_words, dtype=np.int64)
                   if n_words != ext.n_words else None)
    print(f"  electrodes: {ext.n_electrodes},  words: {n_words}")

    shifts = _draw_shifts(n_words, args.n_shuffles, args.min_shift, args.seed)
    print(f"  shuffles: {args.n_shuffles} drawn without replacement from "
          f"{2 * (n_words // 2 - args.min_shift + 1)} offsets with circular "
          f"distance >= {args.min_shift}")
    print(f"  |shift| range: {np.abs(shifts).min()}..{np.abs(shifts).max()} "
          f"words  ({int((shifts < 0).sum())} negative, "
          f"{int((shifts > 0).sum())} positive)")

    if USE_GPU:
        import torch
        iu = torch.triu_indices(n_words, n_words, offset=1, device=DEVICE)
    else:
        iu = np.triu_indices(n_words, k=1)

    print(f"  building and ranking the LLM RDM once ...")
    llm_sq, llm_norm = _square_ranked(llm.cosine_rdm(emb), n_words, iu)
    del emb

    def finish(raw_):
        """Smooth every shuffle along the lag axis. NaN lags stay NaN."""
        pts = max(1, SMOOTH_HALF_WINDOW_MS // args.lag_step)
        return np.stack([lag_window_average(raw_[j], pts)
                         for j in range(args.n_shuffles)], axis=0)

    raw = np.full((args.n_shuffles, n_lags), np.nan, dtype=np.float64)
    for t, lag in enumerate(tqdm(eval_lags, desc="    lag")):
        act = ext.activity(int(lag))
        if word_subset is not None:
            act = act[word_subset]
        brain_sq, brain_norm = _square_ranked(
            compute_rdm_condensed_gpu(act), n_words, iu)
        den = 2.0 * brain_norm * llm_norm          # both squares double-count
        for j, k in enumerate(shifts):
            rolled = _roll_both(llm_sq, int(k))
            if USE_GPU:
                num = float((brain_sq * rolled).sum().item())
            else:
                num = float((brain_sq * rolled).sum())
            raw[j, t] = num / den
            del rolled
        del brain_sq
        # Partial dump every 25 lags, so a crash costs minutes not the run.
        # Lags not yet reached stay NaN and are obvious in the saved file.
        if (t + 1) % 25 == 0 and (t + 1) < n_lags:
            np.savez(PARTIAL, eval_lags_ms=eval_lags, shift_words=shifts,
                     curves_raw=raw.astype(np.float32),
                     lags_done=np.int64(t + 1))

    smoothed = finish(raw)

    # Real curve on the same grid, for convenience when plotting later.
    real_curve = np.full(n_lags, np.nan)
    try:
        real = load_result(f"brain_llm_rsa/{args.roi}_{args.model}")
        rl, rv = np.asarray(real["eval_lags_ms"]), np.asarray(real["rsa_all"])
        idx = np.searchsorted(rl, eval_lags)
        ok = (idx < len(rl)) & (rl[np.clip(idx, 0, len(rl) - 1)] == eval_lags)
        real_curve[ok] = rv[idx[ok]]
    except FileNotFoundError:
        print("  [warn] real curve not found - saving NaNs for real_curve")

    mean = smoothed.mean(axis=0)
    sem = smoothed.std(axis=0, ddof=1) / np.sqrt(args.n_shuffles)

    out = save_result(
        f"brain_llm_rsa_circshift/{args.roi}_{args.model}",
        eval_lags_ms=eval_lags,
        shift_words=shifts,
        curves_raw=raw.astype(np.float32),
        curves=smoothed.astype(np.float32),
        mean=mean, sem=sem,
        real_curve=real_curve,
        n_shuffles=np.int64(args.n_shuffles),
        min_shift_words=np.int64(args.min_shift),
        seed=np.int64(args.seed),
        n_words=np.int64(n_words),
        n_electrodes=np.int64(ext.n_electrodes),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
        lag_step_ms=np.int64(args.lag_step),
    )
    print(f"  -> {out}")

    post = (eval_lags >= 0) & (eval_lags <= 1000)
    print(f"\n  null over {args.n_shuffles} shuffles, 0..1000 ms:")
    print(f"    mean {mean[post].min():+.4f}..{mean[post].max():+.4f}   "
          f"SEM {sem[post].mean():.5f}")
    if np.isfinite(real_curve).any():
        j = int(np.nanargmax(np.where(post, real_curve, np.nan)))
        above = int((smoothed[:, j] >= real_curve[j]).sum())
        print(f"    real peak {real_curve[j]:.4f} at {eval_lags[j]} ms; "
              f"{above}/{args.n_shuffles} shuffles reach it "
              f"(p = {(above + 1) / (args.n_shuffles + 1):.4g})")


if __name__ == "__main__":
    main()
