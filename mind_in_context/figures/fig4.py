"""Figure 4 - Brain × LLM RSA control analyses + RDM correspondence.

Layout (RSA on top spanning both columns; two correspondence matrices below):

  +-----------------------------------------------------------+
  |  A  Temporal RSA control analyses                         |
  |     - all 73 lang electrodes                              |
  |     - circular-shift null (1000 shuffles, mean ± s.e.m.)  |
  |     - permuted null (±1 SD)                               |
  +----------------------------+------------------------------+
  |  B  Original               |  C  Random circular shifts   |
  |     correspondence         |     (off-diagonal destroyed) |
  |     (lag 350 ms, ±50 ms)   |                              |
  +----------------------------+------------------------------+

Both matrices are 50×50 block-averaged and share one symmetric color limit
(taken from the Original off-diagonal). Their titles name the post-onset lag
of the brain RDM, since the LLM RDM is static and the lag is what the panel
is actually selecting.

Panel A's yellow band marks where the real curve clears the circular-shift
null with family-wise correction across lags: the threshold is the 95th
percentile of the per-shuffle MAXIMUM over all lags (max-statistic
correction). Uncorrected per-lag thresholds are not usable here - the real
curve exceeds them at every one of the 321 lags, because a rotated pairing
loses a small amount of correspondence at all lags, not only near the peak.

Loads:
  results/brain_llm_rsa/lang_<model>.npz
  results/brain_llm_rsa_permutation/lang_<model>.npz
  results/brain_llm_rsa_circshift/lang_<model>.npz
  results/brain_llm_correspondence/lang_<model>_lag<L>ms_smooth50.npz
  results/brain_llm_correspondence/lang_<model>_lag<L>ms<shiftkey>_smooth50.npz

Reproduce the two correspondence inputs (not included, about 100 MB each) with:
  python -m mind_in_context.analyses.brain_llm_correspondence \
      --roi lang --model llama3 --lag-ms 350 --smooth-half-window-ms 50
  python -m mind_in_context.analyses.brain_llm_correspondence \
      --roi lang --model llama3 --lag-ms 350 --smooth-half-window-ms 50 \
      --random-shifts 10 --shift-range 600 2568 --shift-seed 0

CLI:  --model {llama3,mistral7b}
      --lag-ms <int>   (default 350; must match saved correspondence files)
Output: plots/fig4.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


from mind_in_context.lib import style

# Authored width drives the type scale: this figure is wide, so its points scale
# up to land at the same on-page size as the rest. K also scales the sizes passed
# explicitly below.
FIG_W_IN = 13.0
K = FIG_W_IN / style.REFERENCE_FIGURE_WIDTH_IN
from mind_in_context.lib.io import load_result, make_argparser, primary_model

CORRESPONDENCE_LAG_MS = 350         # RSA peak; matches the controls supplement
CORR_SMOOTH_HW_MS = 50              # load the `_smooth50` block matrices
CORR_BLOCK = 50                    # 50x50 block-averaging (-> 102x102)
CORR_VLIM_PERCENTILE = 99.0        # shared color limit from Original off-diagonal
# Random-shift control (panel C) - same file the controls supplement uses.
# Same offset range as the panel-A null (600 words to half the narrative), so
# the Results can say "as in the control" truthfully. It was 600-2000 before,
# with no recorded reason for the narrower range.
CORR_SHIFT_KEY = "_shiftrand_n10_range600-2568_seed0"
CORR_SHIFT_TITLE = "Random circular shifts"


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    parser.add_argument("--lag-ms", type=int, default=CORRESPONDENCE_LAG_MS)
    args = parser.parse_args()

    sm = f"_smooth{CORR_SMOOTH_HW_MS}"
    rsa = load_result(f"brain_llm_rsa/lang_{args.model}")
    perm = load_result(f"brain_llm_rsa_permutation/lang_{args.model}")
    circ = load_result(f"brain_llm_rsa_circshift/lang_{args.model}")
    orig = load_result(
        f"brain_llm_correspondence/lang_{args.model}_lag{args.lag_ms}ms{sm}")
    shift = load_result(
        f"brain_llm_correspondence/lang_{args.model}_lag{args.lag_ms}ms"
        f"{CORR_SHIFT_KEY}{sm}")

    style.apply_rcparams()
    import matplotlib as mpl
    mpl.rcParams.update(style.figure_fonts_for_width(FIG_W_IN))
    fig = plt.figure(figsize=(FIG_W_IN, 8.6), constrained_layout=True)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.85, 1.0])

    ax_rsa = fig.add_subplot(outer[0, 0])
    # One flat row for the correspondence block so both matrices get identical
    # widths: [B-mat, gap, C-mat, cbar, right-pad]. B-mat is the first column
    # so its left edge lines up with A; the right pad shrinks the matrices
    # without shifting B.
    bottom = outer[1, 0].subgridspec(
        1, 5, width_ratios=[1.0, 0.16, 1.0, 0.035, 0.30], wspace=0.05)
    ax_bm = fig.add_subplot(bottom[0, 0])
    ax_cm = fig.add_subplot(bottom[0, 2])
    cax = fig.add_subplot(bottom[0, 3])

    _draw_rsa(ax_rsa, rsa, perm, circ)
    _draw_correspondence_pair(fig, ax_bm, ax_cm, cax, orig, shift,
                              lag_ms=args.lag_ms)

    # Freeze the layout, then shift the whole bottom block so panel B's left
    # edge lines up exactly under panel A's plot area (auto-layout leaves them
    # misaligned because A and B have different y-axis label widths).
    fig.canvas.draw()
    fig.set_layout_engine("none")
    dx = ax_rsa.get_position().x0 - ax_bm.get_position().x0
    for a in (ax_bm, ax_cm, cax):
        p = a.get_position()
        a.set_position([p.x0 + dx, p.y0, p.width, p.height])

    out = style.save_svg(fig, "fig4")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def _draw_rsa(ax, rsa, perm, circ) -> None:
    lags = rsa["eval_lags_ms"]
    rsa_all = rsa["rsa_all"]
    n_all = int(rsa["n_all"])

    perm_lags = perm["eval_lags_ms"]
    perm_mean = perm["mean"]
    # Spread of the null across shuffles. SD (not SEM) is the meaningful width:
    # SEM = SD/sqrt(n) is ~10x smaller and invisible at n=100.
    perm_sd = perm["perm_curves"].std(axis=0, ddof=1)
    n_perms = int(perm["n_perms_completed"])

    # Circular-shift null: each shuffle rotates the LLM word axis by a random
    # signed offset, preserving word order and both autocorrelations.
    circ_lags = circ["eval_lags_ms"]
    circ_curves = circ["curves"]
    circ_mean = circ["mean"]
    circ_sem = circ["sem"]
    n_circ = int(circ["n_shuffles"])
    min_shift = int(circ["min_shift_words"])
    circ_lo = np.percentile(circ_curves, 2.5, axis=0)
    circ_hi = np.percentile(circ_curves, 97.5, axis=0)
    # Family-wise threshold: 95th percentile of each shuffle's max over lags.
    thr = float(np.percentile(circ_curves.max(axis=1), 95))
    if not np.array_equal(np.asarray(circ_lags), np.asarray(lags)):
        raise ValueError("circular-shift null is on a different lag grid")

    # Fill from the threshold, not from the null mean: filling from the mean
    # overlaps the percentile envelope and the two translucent fills blend to
    # a third colour. The threshold is also the exact quantity being tested.
    sig = rsa_all > thr
    ax.fill_between(lags, np.full_like(rsa_all, thr), rsa_all, where=sig,
                    color="#FFFF66", alpha=0.55, linewidth=0, zorder=0,
                    label="Above circular-shift null (p < 0.05, FWE)")
    # The threshold itself is not drawn: the edge of the yellow fill already
    # marks it, and a dashed line spanning the whole width read as a data
    # feature of its own.
    ax.plot(lags, rsa_all, color=style.ROI_COLORS["lang"], linewidth=3.3,
            label=f"All electrodes (n={n_all})")
    # The audio-envelope control lives in Supplementary Fig. 1, against a
    # size-matched null. Drawn here beside the full ROI it invited a comparison
    # that confounds audio-responsiveness with having a third as many contacts.
    # Taupe, not gold: the bright yellow fill above means "significant", and a
    # gold envelope right beneath it reads as a second, weaker yellow.
    ax.fill_between(circ_lags, circ_lo, circ_hi, color="#6B5416", alpha=0.20,
                    linewidth=0, zorder=1,
                    label="Circular-shift null (2.5–97.5th pct of shuffles)")
    # The s.e.m. band is drawn to scale and is therefore invisible: at n=1000
    # it is ±0.00008, roughly 1/1250 of the y-range and thinner than the mean
    # line. It is quoted numerically in the legend instead of being inflated.
    ax.fill_between(circ_lags, circ_mean - circ_sem, circ_mean + circ_sem,
                    color="#8A6D1F", alpha=0.9, linewidth=0, zorder=2)
    ax.plot(circ_lags, circ_mean, color="#6B5416", linewidth=1.6,
            linestyle="-", zorder=2,
            label=f"Circular shifts ≥{min_shift} words (n={n_circ}, "
                  f"mean ± s.e.m. {circ_sem.mean():.5f})")
    ax.fill_between(perm_lags, perm_mean - perm_sd, perm_mean + perm_sd,
                    color="#888888", alpha=0.35, linewidth=0,
                    label=f"Permuted null (mean ±1 SD, n={n_perms})")
    ax.plot(perm_lags, perm_mean, color="#555555", linewidth=1.2)
    ax.axvline(0, color=style.WORD_ONSET_COLOR, linestyle="--",
               linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel("Spearman ρ")
    # Centred like panels B and C; left-aligned it collided with the panel letter.
    # The pad keeps it clear of the panel letter, which sits higher still.
    ax.set_title("Brain–LLM Temporal RSA", loc="center", pad=16)
    ax.legend(loc="upper right", fontsize=11)

    y_min = float(min(rsa_all.min(), circ_lo.min(),
                      (perm_mean - perm_sd).min()))
    y_max = float(max(rsa_all.max(), circ_hi.max(),
                      (perm_mean + perm_sd).max()))
    pad = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - pad, y_max + pad * 4)

    pk = int(np.argmax(rsa_all))
    ax.annotate("***", xy=(lags[pk], rsa_all[pk]), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    run = lags[sig]
    print(f"  circular-shift null: mean {circ_mean.mean():.4f}, "
          f"FWE threshold {thr:.4f}, real peak {rsa_all[pk]:.4f} at "
          f"{lags[pk]} ms")
    print(f"  real above threshold at {int(sig.sum())}/{len(lags)} lags "
          f"({run.min()} to {run.max()} ms)")
    # Raised above the title, which this panel has and B/C carry lower down.
    style.panel_label(ax, "A", offset_pt=(-26.0, 26.0))


def _block_average(M: np.ndarray, block: int) -> np.ndarray:
    """Down-sample a square word×word matrix into block×block averages."""
    n = M.shape[0]
    nb = n // block
    Mt = M[:nb * block, :nb * block]
    return Mt.reshape(nb, block, nb, block).mean(axis=(1, 3))


def _corr_panel(ax, M, *, vlim, block, letter, title) -> "object":
    """Draw one block-averaged correspondence matrix."""
    n = M.shape[0]
    extent = [0, n * block, n * block, 0]        # word-index coords, origin top-left
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   aspect="equal", interpolation="nearest", extent=extent)
    ax.set_xlabel("Word index", fontsize=13)
    ax.set_ylabel("Word index", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title(title, fontsize=12, loc="center", pad=5)
    style.panel_label(ax, letter)
    return im


def _draw_correspondence_pair(fig, ax_bm, ax_cm, cax, orig, shift, *,
                              lag_ms) -> None:
    Mo = _block_average(np.asarray(orig["full_matrix"]), CORR_BLOCK)
    Ms = _block_average(np.asarray(shift["full_matrix"]), CORR_BLOCK)

    # Shared symmetric color limit from the Original off-diagonal blocks.
    n = Mo.shape[0]
    off = ~np.eye(n, dtype=bool)
    vlim = float(np.percentile(np.abs(Mo[off]), CORR_VLIM_PERCENTILE))

    # The lag is named in the title: the matrix is one brain RDM (at that lag)
    # against the single static LLM RDM, so the lag is part of what it shows.
    _corr_panel(ax_bm, Mo, vlim=vlim, block=CORR_BLOCK, letter="B",
                title=f"Brain (lag {lag_ms} ms)–LLM contribution")
    im = _corr_panel(ax_cm, Ms, vlim=vlim, block=CORR_BLOCK, letter="C",
                     title=f"{CORR_SHIFT_TITLE} (lag {lag_ms} ms)")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Brain–LLM correspondence (A.U.)", labelpad=4, fontsize=13)
    cbar.ax.tick_params(labelsize=11)


if __name__ == "__main__":
    main()
