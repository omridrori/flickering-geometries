"""Figure 6 - Brain × LLM correspondence vs word-pair distance, per window.

Single panel (language ROI): the mean element-wise correspondence
z_brain × z_model, pooled by word-pair distance |i − j|, for an early window
(50–150 ms) and a late one spanning the RSA peak (350–450 ms). Curves are the
mean over each distance's diagonal, smoothed along the distance axis, with
±1 s.e.m. shading computed within each distance.

Windows rather than single lags: a single lag is one sample of a signal that is
already integrated over 100 ms, so averaging the neighbouring lags costs no
temporal detail and removes the question of why those two lags in particular.

The early-to-late gain over far word pairs (101–1000 apart) is tested against
the circular-shift null in analyses/brain_llm_far_gain_circshift.py and
reported in the text; it is not drawn here.

Loads:
  results/brain_llm_correspondence/lang_<model>_lag<centre>ms_smooth<half>.npz

Reproduce the inputs with:
  python -m mind_in_context.analyses.brain_llm_correspondence \
      --roi lang --model llama3 --lags-ms 100 400 --smooth-half-window-ms 50

CLI:  --model {llama3,mistral7b}
Output: plots/fig6.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.ndimage import uniform_filter1d

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

EARLY_WINDOW_MS = (50, 150)     # early post-onset window
LATE_WINDOW_MS = (350, 450)     # window spanning the RSA peak
DIST_MAX = 1000                 # largest word-pair distance shown
SMOOTH_WINDOW = 20              # moving average along the distance axis
Y_CLIP = (1e-2, 0.5)            # log-y window (near-diagonal peak is off-chart)
NEAR_RANGE = (1, 100)           # near word pairs (K = 0 is the diagonal)


def _window_key(window: tuple[int, int]) -> str:
    """Result-file suffix for a window, as brain_llm_correspondence names it."""
    centre = (window[0] + window[1]) // 2
    half = (window[1] - window[0]) // 2
    return f"lag{centre}ms_smooth{half}"


def _window_label(name: str, window: tuple[int, int]) -> str:
    """Legend entry: the name first, the range in parentheses after it.

    The two windows are referred to as "early" and "late" throughout, so that
    their difference reads as "late − early" rather than as a subtraction of
    two hyphenated ranges, which puts three dashes in one label.
    """
    return f"{name} ({window[0]}–{window[1]} ms)"

# Colour-blind-safe (Wong 2011): blue = early lag, vermillion = late lag.
COLOR_EARLY = "#0072B2"
COLOR_LATE = "#D55E00"

PANEL_IN = 6.5                  # authored figure size; drives the type scale
_FONTS = None                   # filled in main(), scaled for PANEL_IN


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    args = parser.parse_args()

    early = load_result(f"brain_llm_correspondence/lang_{args.model}_"
                        f"{_window_key(EARLY_WINDOW_MS)}")
    late = load_result(f"brain_llm_correspondence/lang_{args.model}_"
                       f"{_window_key(LATE_WINDOW_MS)}")
    M_early = np.asarray(early["full_matrix"], dtype=np.float64)
    M_late = np.asarray(late["full_matrix"], dtype=np.float64)

    global _FONTS
    _FONTS = style.figure_fonts_for_width(PANEL_IN)
    fig, axes = style.fig_layout(rows=1, cols=1,
                                 panel_size=(PANEL_IN, PANEL_IN), fonts=_FONTS)
    ax = axes[0]
    ax.set_box_aspect(1.0)

    _draw_curves(ax, M_early, M_late)

    out = style.save_svg(fig, "fig6")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def _distance_means(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and s.e.m. per word-pair distance K = 1..DIST_MAX (smoothed).

    The s.e.m. is taken within each distance - over the N − K pairs on that
    diagonal - and smoothed with the same kernel as the mean, so the band
    tracks the curve it belongs to. It describes the spread of the pairs at
    one distance; it is not the error term of the far-pair test.
    """
    diags = [np.diagonal(M, offset=K) for K in range(1, DIST_MAX + 1)]
    means = np.array([d.mean() for d in diags])
    sems = np.array([d.std(ddof=1) / np.sqrt(d.size) for d in diags])
    if SMOOTH_WINDOW > 1:
        means = uniform_filter1d(means, size=SMOOTH_WINDOW, mode="nearest")
        sems = uniform_filter1d(sems, size=SMOOTH_WINDOW, mode="nearest")
    return np.arange(1, DIST_MAX + 1, dtype=float), means, sems


def _draw_curves(ax, M_early, M_late) -> None:
    dist, mean_early, sem_early = _distance_means(M_early)
    _, mean_late, sem_late = _distance_means(M_late)

    # Faint divider between near (1-100) and far word pairs.
    ax.axvline(NEAR_RANGE[1], color="#CFCFCF", lw=0.6, ls=(0, (3, 2)), zorder=0)

    for mean, sem, color, name, window in ((mean_early, sem_early, COLOR_EARLY,
                                            "early", EARLY_WINDOW_MS),
                                           (mean_late, sem_late, COLOR_LATE,
                                            "late", LATE_WINDOW_MS)):
        # Clip the lower edge to the axis floor: the band is drawn on a log
        # axis, where mean − sem can fall at or below zero near the far end.
        ax.fill_between(dist, np.maximum(mean - sem, Y_CLIP[0]), mean + sem,
                        color=color, alpha=0.25, linewidth=0, zorder=1)
        ax.plot(dist, mean, color=color, lw=1.6, solid_capstyle="round",
                label=_window_label(name, window), zorder=2)

    ax.set_xlim(0, DIST_MAX)
    ax.set_yscale("log")
    ax.set_ylim(*Y_CLIP)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="both", length=2.5, width=0.6)

    ax.set_xlabel("Word-pair distance")
    ax.set_ylabel("Brain–LLM correspondence (a.u.)")
    legend = ax.legend(loc="lower left", frameon=False, title="Window",
                       handlelength=1.4, borderaxespad=0.6)
    legend.get_title().set_fontsize(style.FIGURE_FONTS["legend.fontsize"])
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    # No panel letter: this figure is a single panel, so a letter labels nothing.
    # (It previously carried a hand-placed lowercase "a".)


if __name__ == "__main__":
    main()
