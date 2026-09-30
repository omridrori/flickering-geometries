"""Figure 3 - Informational selectivity of the ultra-fast transformation.

Three panels, language ROI:

  +---------------------------------------------+
  | A  ρ_ERG curves, -500..1000 ms              |
  |    all words / low (Q1) / high (Q5)         |
  +----------------------+----------------------+
  | B  Q1 low surprisal  | C  Q5 high surprisal |
  |    2D RDM-similarity | 2D RDM-similarity    |
  +----------------------+----------------------+

Panels B/C show the lag×lag RDM-similarity matrix (mean-RDM subtracted) for
predictable (Q1) and surprising (Q5) words separately. Their frames carry the
surprisal colours, they share one y-axis and one colourbar, and they are placed
side by side so together they span panel A's width.

Loads:
  results/surprisal_split/lang.npz
  results/rdm_similarity_2d_surprisal/lang_diag.npz

Output: plots/fig3.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

ZOOM_LAG_MIN_MS = -500
ZOOM_LAG_MAX_MS = 1000

_FONTS = {"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 17,
          "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 15}


def main() -> None:
    parser = make_argparser(__doc__)
    parser.parse_args()

    d = load_result("surprisal_split/lang")
    lags = d["eval_lags_ms"]
    rho_all, rho_q1, rho_q5 = d["rho_all"], d["rho_q1_low"], d["rho_q5_high"]

    m = load_result("rdm_similarity_2d_surprisal/lang_diag")
    S_q1, S_q5, mat_lags = m["S_q1"], m["S_q5"], m["lag_ms"]

    style.apply_rcparams()
    mpl.rcParams.update(_FONTS)
    # Manual layout (no constrained_layout) so the two matrices sit close
    # together, with a dedicated thin column for the shared colourbar.
    fig = plt.figure(figsize=(8.6, 8.4))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.045],
                  height_ratios=[1.0, 1.0], left=0.10, right=0.90,
                  top=0.93, bottom=0.09, hspace=0.42, wspace=0.13)
    ax_a = fig.add_subplot(gs[0, :2])            # spans the two matrix columns
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)
    cax = fig.add_subplot(gs[1, 2])

    zoom = (lags >= ZOOM_LAG_MIN_MS) & (lags <= ZOOM_LAG_MAX_MS)
    _draw_curves(ax_a, lags, rho_all, rho_q1, rho_q5, mask=zoom,
                 title=f"{style.roi_display_name('lang')} — Low vs High surprisal")
    style.panel_label(ax_a, "A")

    abs_max = max(float(np.nanmax(np.abs(S_q1))), float(np.nanmax(np.abs(S_q5))))
    _draw_matrix(ax_b, S_q1, mat_lags, letter="B",
                 title=f"Q1 low surprisal  (n={int(m['n_q1'])})",
                 vmin=-abs_max, vmax=abs_max,
                 frame_color=style.SURPRISAL_COLORS["Q1"])
    im = _draw_matrix(ax_c, S_q5, mat_lags, letter="C",
                      title=f"Q5 high surprisal  (n={int(m['n_q5'])})",
                      vmin=-abs_max, vmax=abs_max,
                      frame_color=style.SURPRISAL_COLORS["Q5"],
                      show_ylabel=False)

    cbar = fig.colorbar(im, cax=cax)             # shared by panels B and C
    cbar.set_label("Residual ρ", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.5)

    out = style.save_svg(fig, "fig3")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def _draw_curves(ax, lags, rho_all, rho_q1, rho_q5, *, mask, title) -> None:
    if mask is not None:
        lags, rho_all = lags[mask], rho_all[mask]
        rho_q1, rho_q5 = rho_q1[mask], rho_q5[mask]

    ax.plot(lags, rho_all, color="#7A7A7A", linewidth=3.0, alpha=0.85,
            label="All words")
    ax.plot(lags, rho_q1, color=style.SURPRISAL_COLORS["Q1"], linewidth=3.0,
            label="Low surpris.")
    ax.plot(lags, rho_q5, color=style.SURPRISAL_COLORS["Q5"], linewidth=3.0,
            label="High surpris.")

    ax.axvline(0, color=style.WORD_ONSET_COLOR, linestyle="--",
               linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel("ERG stability  ρ")
    ax.set_title(title)
    legend = ax.legend(loc="best")
    for line in legend.get_lines():              # thicker legend swatches
        line.set_linewidth(5.0)


def _draw_matrix(ax, S, lags, *, letter: str, title: str, vmin: float,
                 vmax: float, frame_color: str, show_ylabel: bool = True):
    im = ax.imshow(S, origin="lower", aspect="auto",
                   extent=[lags[0], lags[-1], lags[0], lags[-1]],
                   cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Lag t2 (ms)")
    if show_ylabel:
        ax.set_ylabel("Lag t1 (ms)")
    else:                                        # shares the left matrix's axis
        ax.tick_params(labelleft=False, left=False)
    ax.set_title(title, fontsize=15)
    for spine in ax.spines.values():             # thick surprisal-coloured frame
        spine.set_visible(True)
        spine.set_color(frame_color)
        spine.set_linewidth(4.0)
    style.panel_label(ax, letter)
    return im


if __name__ == "__main__":
    main()
