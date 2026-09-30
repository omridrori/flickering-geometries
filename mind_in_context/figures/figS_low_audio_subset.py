"""Low-audio control against a size-matched null - Supplementary Fig. 1.

Reads results/low_audio_random_subsets/<roi>_<model>.npz and draws the
low-audio curve over the band produced by random contact subsets of the same
size. One panel: the comparison is a single claim, and the per-subset peak
distribution it rests on is quoted in the caption rather than plotted.

The legend sits low-left. The curves occupy the upper half of the axes and the
lower half is empty, so anywhere higher covers the peak that the figure exists
to show.

Usage
-----
    python figS_low_audio_subset.py --roi lang --model llama3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

C_RANDOM = "#8D99AE"
XLIM = (-4000, 4000)
POST_WINDOW = (0, 1000)
PANEL_IN = (7.2, 4.2)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    args = parser.parse_args()

    d = load_result(f"low_audio_random_subsets/{args.roi}_{args.model}")
    lags = d["eval_lags_ms"]
    low = d["low_curve"]
    curves = d["curves"]
    n_sub = int(d["n_subset"])
    n_roi = int(d["n_roi"])
    n_draws = curves.shape[0]

    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(n_draws)
    post = (lags >= POST_WINDOW[0]) & (lags <= POST_WINDOW[1])
    peaks = np.nanmax(curves[:, post], axis=1)
    low_peak = float(np.nanmax(low[post]))
    n_below = int((peaks <= low_peak).sum())
    color = style.ROI_COLORS[args.roi]

    fonts = style.figure_fonts_for_width(PANEL_IN[0])
    fig, axes = style.fig_layout(1, 1, panel_size=PANEL_IN, fonts=fonts)
    ax = np.atleast_1d(axes).ravel()[0]

    m = (lags >= XLIM[0]) & (lags <= XLIM[1])
    ax.axhline(0.0, lw=0.6, color="0.85", zorder=0)
    ax.axvline(0.0, lw=0.6, color="0.85", zorder=0)
    ax.fill_between(lags[m], (mean - sem)[m], (mean + sem)[m],
                    color=C_RANDOM, alpha=0.35, lw=0, zorder=1)
    ax.plot(lags[m], mean[m], color=C_RANDOM, lw=1.8, zorder=2)
    ax.plot(lags[m], low[m], color=color, lw=2.4, zorder=3)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel("Brain–LLM correlation (Spearman ρ)")

    # Headroom under the curves for the legend, so it never crosses the peak.
    lo = float(min((mean - sem)[m].min(), low[m].min(), 0.0))
    hi = float(max((mean + sem)[m].max(), low[m].max()))
    span = hi - lo
    ax.set_ylim(lo - 0.62 * span, hi + 0.06 * span)

    handles = [
        Line2D([], [], color=color, lw=2.4,
               label=f"Least audio-responsive third (n={n_sub})"),
        Line2D([], [], color=C_RANDOM, lw=1.8,
               label=f"Random {n_sub} of {n_roi} contacts (mean of {n_draws})"),
        Patch(facecolor=C_RANDOM, alpha=0.35,
              label="± s.e.m. across random subsets"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left",
              borderaxespad=0.6, handlelength=1.8)

    out = style.save_svg(fig, "figS_low_audio_subset")
    print(f"  low-audio peak {low_peak:.4f}; random {peaks.mean():.4f} "
          f"± {peaks.std(ddof=1):.4f}; {n_below}/{n_draws} at or below")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
