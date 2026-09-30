"""Plot style - single source of truth for colors, fonts, save_svg.

Figure scripts must not hard-code colors or fonts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .io import PLOTS_DIR, load_config

# ---------------------------------------------------------------------------
# Palette (mirrors config.yaml).
# ---------------------------------------------------------------------------
ROI_COLORS = {
    "aud": "#0A9396",
    "lang": "#E63946",
}

SURPRISAL_COLORS = {
    "Q1": "#4361EE",
    "Q5": "#E63946",
}

WORD_ONSET_COLOR = "#9D0208"
ZERO_LINE_COLOR = "#888888"
SEGMENT_GRADIENT_CMAP = "coolwarm"


# ---------------------------------------------------------------------------
# Project-standard rcParams.
# ---------------------------------------------------------------------------
def apply_rcparams() -> None:
    """Apply project-wide matplotlib defaults. Idempotent."""
    mpl.rcParams.update({
        # Fonts: humanist sans, with safe fallbacks. svg.fonttype="none"
        # keeps text editable; the consumer (LaTeX/Illustrator) resolves it.
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans",
                            "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Math text in the same family as body text - no thin italic Computer Modern.
        "mathtext.fontset": "stixsans",
        "mathtext.default": "regular",

        # Titles: panel letters are added manually; keep title small & non-bold.
        "axes.titlesize": 9.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,

        "axes.labelsize": 9,
        "axes.labelpad": 3.0,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "legend.fontsize": 8,
        "legend.frameon": False,

        # Lighter, less intrusive grid - reads as background, not pattern.
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "grid.color": "#D9D9D9",
        "grid.alpha": 1.0,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# Main-figure type scale. Figures 3, 4 and 6 each carried their own private
# copy of these numbers, which is why Figure 6 read as a different style from
# the rest. One definition, used by all of them.
FIGURE_FONTS = {
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
}


# Every main figure is placed in the manuscript at the same 6.0 in column width,
# so the type size a reader actually sees is (point size) x 6.0 / (authored
# figure width). A figure authored narrow therefore needs SMALLER points to look
# the same on the page, not larger. FIGURE_FONTS is calibrated against Figure 3,
# which is 8.36 in wide; anything of a different width scales from there.
REFERENCE_FIGURE_WIDTH_IN = 8.36


def figure_fonts_for_width(width_in: float) -> dict:
    """FIGURE_FONTS rescaled so a figure of `width_in` matches on the page."""
    k = width_in / REFERENCE_FIGURE_WIDTH_IN
    return {key: round(v * k, 1) for key, v in FIGURE_FONTS.items()}


def apply_figure_fonts(width_in: float | None = None) -> None:
    """Project rcParams plus the main-figure type scale.

    NOTE: fig_layout() calls apply_rcparams() itself, which resets these. Pass
    the fonts to fig_layout(fonts=...) instead of calling this before it.
    """
    apply_rcparams()
    mpl.rcParams.update(FIGURE_FONTS if width_in is None
                        else figure_fonts_for_width(width_in))


# ---------------------------------------------------------------------------
# Layout helpers.
# ---------------------------------------------------------------------------
def fig_layout(rows: int, cols: int,
               panel_size: tuple[float, float] = (4.0, 3.0),
               *, constrained: bool = True, fonts: dict | None = None
               ) -> tuple[Figure, "list"]:
    """Create a Figure + flat list of axes with project-standard sizing.

    Uses constrained_layout by default - handles colorbars and outside-axes
    annotations (panel letters, suptitle) gracefully without manual padding.

    `fonts` is applied AFTER the project defaults. Callers used to set their own
    rcParams and then call this, which silently reset every one of them back to
    the 9 pt default - the reason Figure 6's axis labels came out far smaller
    than the rest of the paper's.
    """
    apply_rcparams()
    if fonts:
        mpl.rcParams.update(fonts)
    w = panel_size[0] * cols
    h = panel_size[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(w, h),
                             constrained_layout=constrained)
    if rows == 1 and cols == 1:
        return fig, [axes]
    return fig, list(axes.ravel())


# ---------------------------------------------------------------------------
# Common annotations.
# ---------------------------------------------------------------------------
def add_word_onset(ax) -> None:
    """Dashed vertical line at lag = 0."""
    ax.axvline(0, color=WORD_ONSET_COLOR, linestyle="--", linewidth=1.5)


PANEL_LABEL_OFFSET_PT = (-26.0, 8.0)   # left, up from the axes' top-left corner


def panel_label(ax, letter: str, *, fontsize: float | None = None,
                offset_pt: tuple[float, float] | None = None) -> None:
    """Bold panel letter (A, B, ...) above and left of the axes' top-left corner.

    Two things this gets right that the previous version did not.

    Size scales with the figure's base font. It used to be pinned at 11 pt while
    the figure scripts raise font.size to 14-15, so the letters came out smaller
    than the axis labels they were meant to head.

    Position is a constant offset in POINTS, not in axes fractions. An axes
    fraction is a fraction of that panel's own width, so the same value put the
    letter a different physical distance from a wide time-course panel than from
    a square matrix - which is what made the letters look unaligned across a
    figure with mixed panel shapes.
    """
    size = fontsize if fontsize is not None else max(12.0, mpl.rcParams["font.size"] * 1.5)
    dx, dy = offset_pt if offset_pt is not None else PANEL_LABEL_OFFSET_PT
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=size, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)


BAR_BASELINE = -5  # bar bottom sits this far below zero so a 0 bar is visible


def draw_segment_bar(ax, values, *, ylabel: str, title: str,
                     yerr=None, ylim=None) -> None:
    """Standard 5-segment bar chart used by Fig 2 (and supplementary)."""
    import numpy as np
    n = len(values)
    colors = [segment_color(k, n) for k in range(n)]
    x = np.arange(1, n + 1)
    ax.bar(x, np.asarray(values) - BAR_BASELINE, bottom=BAR_BASELINE,
           color=colors, edgecolor="black", linewidth=0.6, width=0.78,
           yerr=yerr,
           error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2.5,
                         capthick=0.8) if yerr is not None else None)
    ax.axhline(0, color="#888888", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Segment")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.margins(x=0.01)
    if ylim is not None:
        ax.set_ylim(*ylim)


def draw_segment_scatter(ax, values, *, ylabel: str, title: str,
                         show_r: bool = True) -> None:
    """Per-segment scatter + linear-regression line (used by Fig 2 panels C-F).

    Dots colored along the segment gradient (early=blue -> late=red), dashed
    black regression line. Inset shows Pearson r. The y-axis is auto-scaled
    to the data with a small pad, never anchored at 0.
    """
    import numpy as np
    n = len(values)
    x = np.arange(1, n + 1)
    vals = np.asarray(values, dtype=float)

    slope, intercept = np.polyfit(x, vals, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    r_obs = float(np.corrcoef(x, vals)[0, 1])

    for k in range(n):
        ax.scatter(x[k], vals[k], color=segment_color(k, n),
                   s=42, edgecolor="black", linewidth=0.5, zorder=3)
    ax.plot(x_line, y_line, color="#222222", linewidth=1.5,
            linestyle="--", zorder=2,
            label=f"y = {slope:+.4g}·x + {intercept:+.4g}")

    if show_r:
        ax.text(0.04, 0.96, f"r = {r_obs:+.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, family="monospace",
                bbox=dict(facecolor="white", edgecolor="#CCCCCC",
                          boxstyle="round,pad=0.3", linewidth=0.6))

    ax.set_xlabel(f"Segment (1 → {n}, early → late)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, n + 1, max(1, n // 10)))
    ax.legend(loc="lower right", fontsize=7)

    y_min, y_max = float(vals.min()), float(vals.max())
    pad = (y_max - y_min) * 0.10 or abs(y_min) * 0.10 or 0.001
    ax.set_ylim(y_min - pad, y_max + pad)


def add_zero_line(ax) -> None:
    """Thin horizontal reference at y = 0."""
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.8, alpha=0.6)


def segment_color(seg_idx: int, n_segments: int):
    """Return a coolwarm color for segment seg_idx ∈ [0, n_segments-1]."""
    cmap = plt.get_cmap(SEGMENT_GRADIENT_CMAP)
    return cmap(seg_idx / max(1, n_segments - 1))


def roi_color(roi_key: str) -> str:
    return ROI_COLORS[roi_key]


def roi_display_name(roi_key: str) -> str:
    return load_config()["rois"][roi_key]["display_name"]


# ---------------------------------------------------------------------------
# Save.
# ---------------------------------------------------------------------------
def save_svg(fig: Figure, name: str) -> Path:
    """Write fig to plots/<name>.svg (no .svg suffix needed in name)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path
