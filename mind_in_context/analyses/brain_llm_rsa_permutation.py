"""Permutation test for brain × LLM temporal RSA (Fig 4 panel A).

Null hypothesis: there is no meaningful correspondence between the brain RDM
and the LLM RDM at any post-onset lag. Under H0, randomly permuting the LLM's
word labels should yield the same RSA peak as the unpermuted curve.

Procedure (matches `brain_llm_rsa.py` parameters exactly except for the
permutation of LLM word labels):
  - For each permutation k of 1..N:
      * Sample a random permutation π_k of the word indices.
      * Permute rows + cols of the LLM RDM squareform by π_k -> permuted LLM RDM.
      * Pre-rank-centre the permuted LLM RDM on the device (GPU if available).
      * For each lag t in -4000..+4000 ms (step 25):
          - brain RDM at lag t (cached across permutations by BaselineExtractor).
          - Spearman ρ via the device-side pre-ranked LLM RDM.
      * Smooth the curve with the same ±50 ms moving average used by the
        reference analysis.
      * Write per-permutation curve to disk -> resumable.

Resumability: each completed permutation is written as
  results/brain_llm_rsa_permutation/<roi>_<model>/perm_<k>.npy
On restart, existing files are skipped. Aggregate `.npz` is rewritten at the
end (and every `--aggregate-every` permutations) so partial results are
inspectable mid-run.

Statistics (computed once all permutations are done):
  (a) Per-lag p-value: fraction of perms whose ρ exceeds the real ρ at lag t.
  (b) Peak-lag p-value: max(real ρ) within POST-ONSET window vs distribution
      of max(perm ρ) within the same window.
  (c) Margin p-value: mean(real ρ) within MARGIN window vs distribution of
      mean(perm ρ) within the same window.

The real curve is loaded from `results/brain_llm_rsa/<roi>_<model>.npz`.

CLI:  --roi {lang,aud}  --model {llama3,mistral7b}  [--n-perms 100] [--seed 0]
      [--peak-window 0 500]  [--margin-window 500 1500]
      [--aggregate-every 10]
Output: results/brain_llm_rsa_permutation/<roi>_<model>.npz
        results/brain_llm_rsa_permutation/<roi>_<model>/perm_<k>.npy  (partial)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.distance import squareform
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (
    RESULTS_DIR, load_config, load_result, make_argparser,
)
from mind_in_context.lib.rdm import (
    USE_GPU, lag_window_average, rank_centered_gpu, spearman_from_ranked,
)

# Mirror brain_llm_rsa.py parameters. Lag range is restricted by default to
# the post-onset RSA peak region for runtime; the real curve covers ±4000 ms
# and can still be compared on this sub-range without recomputation.
LAG_START_MS = -1000
LAG_END_MS = 1000
LAG_STEP_MS = 25
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 50


def _outdir(roi: str, model: str) -> Path:
    return RESULTS_DIR / "brain_llm_rsa_permutation" / f"{roi}_{model}"


def _perm_path(roi: str, model: str, k: int) -> Path:
    return _outdir(roi, model) / f"perm_{k:04d}.npy"


def _load_completed_perms(roi: str, model: str, n_lags: int
                          ) -> tuple[list[int], np.ndarray | None]:
    """Return (sorted indices of completed perms, stacked curves) or ([], None)."""
    od = _outdir(roi, model)
    if not od.exists():
        return [], None
    files = sorted(od.glob("perm_*.npy"))
    if not files:
        return [], None
    completed: list[int] = []
    curves = []
    for f in files:
        try:
            k = int(f.stem.split("_")[1])
            arr = np.load(f)
            if arr.shape != (n_lags,):
                continue
            completed.append(k)
            curves.append(arr)
        except (ValueError, OSError):
            continue
    if not completed:
        return [], None
    order = np.argsort(completed)
    completed_sorted = [completed[i] for i in order]
    curves_sorted = np.stack([curves[i] for i in order], axis=0)
    return completed_sorted, curves_sorted


def _build_llm_rdm_square(model_key: str, n_words: int) -> np.ndarray:
    layer = load_config()["models"][model_key]["layer"]
    emb_path = llm.full_context_embeddings_path(model_key, layer)
    print(f"  loading LLM embeddings: {emb_path.name}")
    emb = np.load(emb_path).astype(np.float64)
    n = min(emb.shape[0], n_words)
    emb = emb[:n]
    print(f"  building LLM RDM (cosine_rdm, n_words={n})...")
    rdm_condensed = llm.cosine_rdm(emb)
    return squareform(rdm_condensed).astype(np.float32)


def _aggregate_and_save(roi: str, model: str, *, eval_lags: np.ndarray,
                        completed: list[int], curves: np.ndarray,
                        real: dict, peak_window: tuple[int, int],
                        margin_window: tuple[int, int]) -> Path:
    """Compute mean / SEM / p-values, save aggregate .npz."""
    n_perms = len(completed)
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(n_perms) if n_perms > 1 else np.zeros_like(mean)

    real_lags = np.asarray(real["eval_lags_ms"])
    real_full = np.asarray(real["rsa_all"])
    # Align real curve to the permutation lag grid.
    if real_lags.shape == eval_lags.shape and np.array_equal(real_lags, eval_lags):
        real_curve = real_full
    else:
        idx = np.searchsorted(real_lags, eval_lags)
        if (idx >= len(real_lags)).any() or (real_lags[idx] != eval_lags).any():
            raise ValueError(
                f"Real curve lags do not contain all permutation lags. "
                f"Real range [{real_lags[0]}, {real_lags[-1]}], "
                f"perm range [{eval_lags[0]}, {eval_lags[-1]}]"
            )
        real_curve = real_full[idx]
    # Per-lag p-value: fraction of perms >= real (one-tailed).
    p_per_lag = (np.sum(curves >= real_curve[None, :], axis=0) + 1) / (n_perms + 1)

    # Peak window: max of curve within [peak_lo, peak_hi].
    in_peak = (eval_lags >= peak_window[0]) & (eval_lags <= peak_window[1])
    real_peak = float(real_curve[in_peak].max())
    perm_peaks = curves[:, in_peak].max(axis=1)
    p_peak = (np.sum(perm_peaks >= real_peak) + 1) / (n_perms + 1)

    # Margin window: mean of curve within [margin_lo, margin_hi].
    in_margin = (eval_lags >= margin_window[0]) & (eval_lags <= margin_window[1])
    real_margin = float(real_curve[in_margin].mean())
    perm_margins = curves[:, in_margin].mean(axis=1)
    p_margin = (np.sum(perm_margins >= real_margin) + 1) / (n_perms + 1)

    out = RESULTS_DIR / "brain_llm_rsa_permutation" / f"{roi}_{model}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        eval_lags_ms=eval_lags.astype(np.int64),
        perm_curves=curves.astype(np.float32),
        mean=mean.astype(np.float32),
        sem=sem.astype(np.float32),
        completed_perm_seeds=np.array(completed, dtype=np.int64),
        n_perms_completed=np.int64(n_perms),
        real_curve=real_curve.astype(np.float32),
        p_per_lag=p_per_lag.astype(np.float32),
        peak_window_ms=np.array(peak_window, dtype=np.int64),
        real_peak=np.float64(real_peak),
        perm_peaks=perm_peaks.astype(np.float32),
        p_peak=np.float64(p_peak),
        margin_window_ms=np.array(margin_window, dtype=np.int64),
        real_margin=np.float64(real_margin),
        perm_margins=perm_margins.astype(np.float32),
        p_margin=np.float64(p_margin),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
    )
    return out


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--lag-start", type=int, default=LAG_START_MS)
    parser.add_argument("--lag-end", type=int, default=LAG_END_MS)
    parser.add_argument("--lag-step", type=int, default=LAG_STEP_MS)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed; perm k uses seed + k.")
    parser.add_argument("--peak-window", type=int, nargs=2, default=(0, 500),
                        metavar=("LO", "HI"),
                        help="Lag window (ms) for the peak statistic.")
    parser.add_argument("--margin-window", type=int, nargs=2, default=(500, 1500),
                        metavar=("LO", "HI"),
                        help="Lag window (ms) for the margin statistic.")
    parser.add_argument("--aggregate-every", type=int, default=10,
                        help="Rewrite aggregate .npz every N perms.")
    args = parser.parse_args()

    s = load_config()["shared"]
    eval_lags = np.arange(args.lag_start, args.lag_end + 1, args.lag_step,
                          dtype=int)
    n_lags = len(eval_lags)
    print(f"  ROI: {args.roi},  model: {args.model},  GPU: {USE_GPU}")
    print(f"  lags [{args.lag_start}, {args.lag_end}] ms step {args.lag_step}  "
          f"({n_lags} lags)")
    print(f"  perms: {args.n_perms},  base seed: {args.seed}")

    # Real curve (for stats at the end). Errors loudly if missing.
    try:
        real = dict(load_result(f"brain_llm_rsa/{args.roi}_{args.model}").items())
    except FileNotFoundError as e:
        raise SystemExit(
            f"Need the real curve first: run analyses/brain_llm_rsa.py "
            f"--roi {args.roi} --model {args.model}. ({e})"
        )

    # Brain extractor (same params as brain_llm_rsa.py).
    ext = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    print(f"  electrodes: {ext.n_electrodes},  words: {ext.n_words}")

    # LLM RDM as a square (n_words × n_words).
    llm_sq = _build_llm_rdm_square(args.model, ext.n_words)
    n_words = llm_sq.shape[0]
    if n_words != ext.n_words:
        # Brain has more / fewer words than LLM - restrict to overlap.
        print(f"  word count mismatch: brain={ext.n_words}, LLM={n_words}; "
              f"using first {n_words} words from both.")
        # We'll feed a fixed subset to extractor.rdm via word_subset.
        word_subset = np.arange(n_words, dtype=np.int64)
    else:
        word_subset = None

    # Resume.
    completed, curves = _load_completed_perms(args.roi, args.model, n_lags)
    completed_set = set(completed)
    if completed_set:
        print(f"  resuming: {len(completed)} completed perms already on disk")

    smooth_half_pts = max(1, SMOOTH_HALF_WINDOW_MS // LAG_STEP_MS)
    rng_base = args.seed

    pbar_outer = tqdm(range(1, args.n_perms + 1), desc="perm", position=0,
                      leave=True)
    for k in pbar_outer:
        if k in completed_set:
            continue
        rng = np.random.default_rng(rng_base + k)
        pi = rng.permutation(n_words)
        # Permute LLM RDM by π - equivalent to permuting word labels.
        llm_perm_sq = llm_sq[pi][:, pi]
        llm_perm_condensed = squareform(llm_perm_sq, checks=False)
        # Pre-rank-centre once per perm (GPU when available).
        llm_rank = rank_centered_gpu(llm_perm_condensed)

        raw = np.full(n_lags, np.nan, dtype=np.float64)
        pbar_inner = tqdm(range(n_lags), desc=f"  lag (perm {k})", position=1,
                          leave=False)
        for t in pbar_inner:
            brain_rdm = ext.rdm(int(eval_lags[t]), word_subset=word_subset)
            raw[t] = spearman_from_ranked(llm_rank, brain_rdm)
        pbar_inner.close()

        smoothed = lag_window_average(raw, smooth_half_pts).astype(np.float32)
        p = _perm_path(args.roi, args.model, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, smoothed)

        # Update in-memory aggregate and periodically rewrite .npz.
        if curves is None:
            curves = smoothed[None, :].copy()
        else:
            curves = np.concatenate([curves, smoothed[None, :]], axis=0)
        completed.append(k)
        completed_set.add(k)

        if (len(completed) % args.aggregate_every) == 0:
            _aggregate_and_save(
                args.roi, args.model, eval_lags=eval_lags,
                completed=completed, curves=curves, real=real,
                peak_window=tuple(args.peak_window),
                margin_window=tuple(args.margin_window),
            )
        pbar_outer.set_postfix(done=len(completed))

    pbar_outer.close()

    out = _aggregate_and_save(
        args.roi, args.model, eval_lags=eval_lags,
        completed=completed, curves=curves, real=real,
        peak_window=tuple(args.peak_window),
        margin_window=tuple(args.margin_window),
    )
    print(f"  -> {out}")

    # Echo headline stats.
    print()
    d = np.load(out)
    print(f"  RESULTS  (n_perms = {int(d['n_perms_completed'])})")
    print(f"  peak window  {tuple(d['peak_window_ms'].tolist())} ms:  "
          f"real = {float(d['real_peak']):.4f},  p = {float(d['p_peak']):.4g}")
    print(f"  margin window  {tuple(d['margin_window_ms'].tolist())} ms:  "
          f"real = {float(d['real_margin']):.4f},  p = {float(d['p_margin']):.4g}")


if __name__ == "__main__":
    main()
