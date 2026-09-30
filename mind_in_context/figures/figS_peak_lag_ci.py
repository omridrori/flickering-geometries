"""Extended Data Fig. 8 - how well each peak lag is actually resolved.

The best-matching lag rises with context length, but each individual peak is an
argmax taken from a smooth, noisy curve. The whole analysis was recomputed on
random 80% subsets of the words; this shows what that produced.

  A  the full distribution per context     - every subset, not a summary of them
  B  the two-word -> full-narrative shift  - the quantity the claim rests on

A percentile interval is deliberately not plotted. It would summarise these
distributions as if they were single-peaked, and beyond roughly twelve words
they are not: a second mode appears near 360 ms, meaning the argmax flips
between two nearly equal humps from one subset to the next. Panel A shows that
directly; an interval would have hidden it.

Loads:  results/peak_lag_subsample_ci/<roi>_<model>.npz
Output: plots/figS_peak_lag_ci.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

BIN_MS = 2


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    args = parser.parse_args()

    d = load_result(f"peak_lag_subsample_ci/{args.roi}_{args.model}")
    peaks = np.asarray(d["peaks"], dtype=float)          # (n_rep, n_context)
    point = np.asarray(d["peaks_full_sample"], dtype=float)
    is_full = np.asarray(d["is_full"])
    n_rep, n_ctx = peaks.shape
    color = style.roi_color(args.roi)
    frac = int(round(100 * float(d["fraction"])))

    x = np.arange(n_ctx, dtype=float)
    k = int((~is_full).sum())                 # index of the "full" column
    tick_labels = ["2", "5", "10", "15", "20", "full"]

    fig, axes = style.fig_layout(1, 2, panel_size=(4.7, 3.6))

    # ------------------------------------------ A  the distributions in full
    ax = axes[0]
    edges = np.arange(np.floor(peaks.min() / BIN_MS) * BIN_MS,
                      np.ceil(peaks.max() / BIN_MS) * BIN_MS + BIN_MS, BIN_MS)
    dens = np.stack([np.histogram(peaks[:, i], bins=edges)[0] for i in range(n_ctx)])
    dens = dens / dens.max(axis=1, keepdims=True)       # per-context, so short
    im = ax.imshow(dens.T, origin="lower", aspect="auto", cmap="Reds",
                   extent=[-0.5, n_ctx - 0.5, edges[0], edges[-1]],
                   interpolation="nearest", vmin=0, vmax=1)
    ax.plot(x, point, color="#222222", linewidth=1.0, marker="o", ms=2.5,
            markerfacecolor="white", markeredgewidth=0.6)
    ax.axvline(k - 0.5, color="#666666", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_xticks([0, 3, 8, 13, 18, n_ctx - 1])
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Context length (words)")
    ax.set_ylabel("Best-matching lag (ms)")
    ax.set_title(f"Distribution over {n_rep} subsets", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Subsets (peak per context)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

    # ------------------------------------------------------------ B  the shift
    ax = axes[1]
    shift = peaks[:, -1] - peaks[:, 0]
    s_lo, s_hi = np.percentile(shift, [2.5, 97.5])
    ax.hist(shift, bins=np.arange(shift.min() - 2, shift.max() + 4, 3),
            color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(point[-1] - point[0], color="#222222", linewidth=1.6,
               label=f"All words: {point[-1] - point[0]:.0f} ms")
    ax.axvspan(s_lo, s_hi, color="#999999", alpha=0.18, linewidth=0,
               label=f"95% CI {s_lo:.0f}–{s_hi:.0f} ms")
    ax.axvline(0, color=style.ZERO_LINE_COLOR, linewidth=0.9,
               linestyle=(0, (3, 3)))
    ax.set_xlabel("Shift, two words → full narrative (ms)")
    ax.set_ylabel(f"Subsets (of {n_rep})")
    ax.set_title(f"{np.mean(shift > 0) * 100:.0f}% of subsets positive",
                 fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.94)
    ax.grid(alpha=0.20, linewidth=0.5)

    for ax, letter in zip(axes, "AB"):
        style.panel_label(ax, letter)

    fig.suptitle(
        f"{style.roi_display_name(args.roi)} — peak lag re-estimated on "
        f"{n_rep} random {frac}% word subsets (without replacement)",
        fontsize=10)

    out = style.save_svg(fig, "figS_peak_lag_ci")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
