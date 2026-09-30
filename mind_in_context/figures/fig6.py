"""Figure 6 - Brain × LLM correspondence vs word-pair distance, per window.

Single panel (language ROI): the mean element-wise correspondence
z_brain × z_model, pooled by word-pair distance |i − j|, for an early window
(50–150 ms) and a late one spanning the RSA peak (350–450 ms). Curves are the
mean over each distance's diagonal, smoothed along the distance axis, with
±1 s.e.m. shading computed within each distance.

Windows rather than single lags: a single lag is one sample of a signal that is
already integrated over 100 ms, so averaging the neighbouring lags costs no
temporal detail and removes the question of why those two lags in particular.

The inset gives the significance. Its statistic is the mean early-to-late gain
over FAR word pairs (101–1000 apart) - the pairs that a rotation of the word
labels cannot preserve - shown against the circular-shift null. A t-test is not
usable on these values: each word enters thousands of pairs, so RDM entries are
not independent observations. See analyses/brain_llm_far_gain_circshift.py.

Loads:
  results/brain_llm_correspondence/lang_<model>_lag<centre>ms_smooth<half>.npz
  results/brain_llm_far_gain_circshift/lang_<model>.npz

Reproduce the inputs with:
  python -m mind_in_context.analyses.brain_llm_correspondence \
      --roi lang --model llama3 --lags-ms 100 400 --smooth-half-window-ms 50
  python -m mind_in_context.analyses.brain_llm_far_gain_circshift \
      --roi lang --model llama3

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
FAR_RANGE = (101, 1000)         # inset: the far pairs the statistic uses


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
BAR_NEAR = "#BDBDBD"            # neutral greys: the bars show a difference,
BAR_FAR = "#525252"             # not the lag curves themselves

PANEL_IN = 6.5                  # authored figure size; drives the type scale
_FONTS = None                   # filled in main(), scaled for PANEL_IN

# Inset type, kept a clear step BELOW the main axes so they read as the primary
# ones.
_INSET_SCALE = 0.72


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    args = parser.parse_args()

    early = load_result(f"brain_llm_correspondence/lang_{args.model}_"
                        f"{_window_key(EARLY_WINDOW_MS)}")
    late = load_result(f"brain_llm_correspondence/lang_{args.model}_"
                       f"{_window_key(LATE_WINDOW_MS)}")
    M_early = np.asarray(early["full_matrix"], dtype=np.float64)
    M_late = np.asarray(late["full_matrix"], dtype=np.float64)
    null = load_result(f"brain_llm_far_gain_circshift/lang_{args.model}")

    global _FONTS
    _FONTS = style.figure_fonts_for_width(PANEL_IN)
    fig, axes = style.fig_layout(rows=1, cols=1,
                                 panel_size=(PANEL_IN, PANEL_IN), fonts=_FONTS)
    ax = axes[0]
    ax.set_box_aspect(1.0)

    _draw_curves(ax, M_early, M_late)
    _draw_inset(ax, null)

    out = style.save_svg(fig, "fig6")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def _distance_means(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and s.e.m. per word-pair distance K = 1..DIST_MAX (smoothed).

    The s.e.m. is taken within each distance - over the N − K pairs on that
    diagonal - and smoothed with the same kernel as the mean, so the band
    tracks the curve it belongs to. It describes the spread of the pairs at
    one distance; it is not the error term of the inset's test.
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

    # Faint divider where the inset's far range begins.
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


def _draw_inset(ax, null) -> None:
    """Far-pair gain, real against the circular-shift null.

    Not near-versus-far: a rotation of the word labels reproduces roughly half
    of that contrast on its own, because the near-diagonal contribution shrinks
    from the early to the late window under rotation while the far contribution
    stays at zero. The far gain alone has a null centred on zero, so it is the
    quantity that actually separates.
    """
    far_real = float(null["far_real"])
    null_far = np.asarray(null["null_far"], dtype=np.float64)
    p_value = float(null["p_far"])
    n_shifts = int(null["n_shifts"])
    means = [far_real, float(null_far.mean())]
    sems = [0.0, float(null_far.std(ddof=1))]
    print(f"  far {FAR_RANGE} gain: real={means[0]:+.5f}")
    print(f"  circular-shift null (n={n_shifts}): mean={means[1]:+.5f} "
          f"± {sems[1]:.5f}, max={null_far.max():+.5f}")
    print(f"  far gain vs circular shift: p={p_value:.4f}")

    axin = ax.inset_axes([0.66, 0.66, 0.30, 0.28])
    x = np.arange(2)
    axin.bar(x, means, yerr=sems, width=0.62, color=[BAR_FAR, BAR_NEAR],
             edgecolor="black", linewidth=0.5,
             error_kw=dict(ecolor="black", elinewidth=0.7, capsize=2,
                           capthick=0.7))
    axin.axhline(0, color="#888888", lw=0.5)
    axin.grid(False)
    axin.set_xticks(x)
    axin.set_xticklabels(["real", "shifted"],
                         fontsize=_FONTS["xtick.labelsize"] * _INSET_SCALE)
    axin.set_xlabel(f"far pairs ({FAR_RANGE[0] - 1}–{FAR_RANGE[1]})",
                    fontsize=_FONTS["axes.labelsize"] * _INSET_SCALE)
    # The inset's y axis had no quantity on it, only the lag difference. It is
    # the same measure as the main y axis, differenced across the two windows.
    axin.set_ylabel("Δ Brain–LLM corr (a.u.)\n(late − early)",
                    fontsize=_FONTS["axes.labelsize"] * _INSET_SCALE,
                    linespacing=1.15)
    # p as plain text over the real bar. No bracket and no stars: those mark a
    # comparison between the two x categories, and the categories here are the
    # measurement and its null, not two ranges of word distance.
    axin.annotate(f"p = {p_value:.3f}" if p_value > 1.0 / n_shifts
                  else f"p < {1.0 / n_shifts:.3f}",
                  xy=(0, means[0]), xytext=(0, 4), textcoords="offset points",
                  ha="center", va="bottom",
                  fontsize=_FONTS["xtick.labelsize"] * _INSET_SCALE)
    axin.tick_params(axis="y", labelsize=_FONTS["ytick.labelsize"] * _INSET_SCALE,
                     length=2, width=0.5)
    axin.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        axin.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axin.spines[side].set_linewidth(0.5)

    # Headroom for the p annotation above the taller bar. No bracket and no
    # stars: a bracket spans two conditions being contrasted, and these two
    # bars are a measurement and its own null rather than two conditions.
    axin.set_ylim(0, max(m + s for m, s in zip(means, sems)) * 1.35)


if __name__ == "__main__":
    main()
