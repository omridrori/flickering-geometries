"""Extended Data Fig. 10 - correspondence by word-pair distance, window by window.

One curve per 100 ms post-onset window (50-150, 150-250, 250-350, 350-450 ms),
each showing the mean brain-LLM correspondence at every narrative distance
between the two words of a pair. Colours run cold to warm with latency, using
the same coolwarm progression the paper uses for ordered series, so the
ordering reads without tracing the legend.

What the figure is for. Figure 6 contrasts two windows and shows that the gain
between them is carried by distant pairs. Four windows show the shape of that
gain over time: the near-diagonal contribution is unchanged across windows,
while the far-pair contribution rises and then settles, so the geometry becomes
contextual over the first few hundred milliseconds rather than drifting for the
whole epoch.

No s.e.m. shading: with four overlapping curves the bands obscure the ordering,
which is the thing being shown. Figure 6 carries the spread for the two windows
the statistics are computed on.

Loads:
  results/brain_llm_distance_by_window/lang_<model>.npz

Reproduce the input with:
  python -m mind_in_context.analyses.brain_llm_distance_by_window \
      --roi lang --model llama3

CLI:  --model {llama3,mistral7b}
Output: plots/figS_distance_by_window.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.ndimage import uniform_filter1d

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

PANEL_IN = 6.5                  # authored figure size; drives the type scale
SMOOTH_WINDOW = 20              # moving average along the distance axis
Y_CLIP = (1e-2, 0.5)            # log-y window (near-diagonal peak is off-chart)


def _draw(ax, distances, curves, windows, near_max) -> None:
    for k, (curve, window) in enumerate(zip(curves, windows)):
        ax.plot(distances, uniform_filter1d(curve, size=SMOOTH_WINDOW,
                                            mode="nearest"),
                color=style.segment_color(k, len(windows)), lw=1.7,
                solid_capstyle="round",
                label=f"{int(window[0])}–{int(window[1])} ms")
    # Faint divider where Figure 6's far range begins, so the two figures can
    # be read against each other.
    ax.axvline(near_max, color="#CFCFCF", lw=0.7, ls=(0, (3, 2)), zorder=0)

    ax.set_yscale("log")
    ax.set_xlim(0, float(distances[-1]))
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


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    args = parser.parse_args()

    data = load_result(f"brain_llm_distance_by_window/lang_{args.model}")
    distances = np.asarray(data["distances"], dtype=float)
    curves = np.asarray(data["curves"], dtype=float)
    windows = np.asarray(data["windows_ms"], dtype=int)
    near_max = int(data["near_max"])

    fonts = style.figure_fonts_for_width(PANEL_IN)
    fig, axes = style.fig_layout(rows=1, cols=1,
                                 panel_size=(PANEL_IN, PANEL_IN * 0.78),
                                 fonts=fonts)
    ax = axes[0]
    _draw(ax, distances, curves, windows, near_max)

    for window, near, far in zip(windows, data["near_means"], data["far_means"]):
        print(f"  {int(window[0])}–{int(window[1])} ms: near={near:+.5f}, "
              f"far={far:+.5f}")

    out = style.save_svg(fig, "figS_distance_by_window")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
