"""Figure 5 - Narrative buildup along the brain transformation and across LLM layers.

Two panels (the former Figs 5 and 6, published as one figure):

  +-------------------------------------------+
  | A  language ERG curve                     |
  |    + per-context peak-lag markers 1..20   |
  |    + inset: peak-normalised rho(lag) per  |
  |      context window (bottom-right)        |
  +-------------------------------------------+
  | B  peak-lag vs context length, per layer  |
  +-------------------------------------------+

Panel A's markers, panel B and the inset all read from the same by-layer sweep
at PANEL_A_MARKER_LAYER, so their peak lags agree by construction.

Loads:
  results/erg/lang_full.npz                    (panel A curve)
  results/rsa_peak_by_layer/lang_<model>.npz   (panel A markers, panel B, inset)

CLI: --model {llama3,mistral7b}
Output: plots/fig5.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from mind_in_context.lib import style

# See fig4.py: authored width drives the type scale, so every figure reads at the
# same size once placed at the manuscript's column width.
FIG_W_IN = 12.0
K = FIG_W_IN / style.REFERENCE_FIGURE_WIDTH_IN
from mind_in_context.lib.io import load_result, make_argparser

ZOOM_LANG = (250, 400)            # lag window of the embedded context panel
INSET_MARKER_STRIDE = 3           # keep every Nth peak marker so labels stay legible
INSET_FS = 0.85                   # font scale shared by BOTH panel-A insets
CMAP = "jet"
# jet ends in a dark, near-black red that reads like the cool end. Stop the
# ramp short so the longest context lands on a vivid red instead.
CMAP_HI = 0.88
FULL_COLOR = "#D00000"            # full-context curve/marker (off the scale)
MAX_CONTEXT_FOR_PANEL_B = 90      # longest finite context drawn in panel B
MARKED_CONTEXTS_PANEL_A = 20      # only the dense 1..20 set gets triangles
PANEL_A_MARKER_LAYER = 25         # LLM layer whose per-context peak lags mark panel A
PANEL_A_ZOOM_MS = (280, 370)      # lag window shown in the main panel A axes
# Panel B is cluttered with all 26 layers; show a representative subset only.
SELECTED_LAYERS_PANEL_B = (5, 10, 15, 20, 25, 30)


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    args = parser.parse_args()

    erg = load_result("erg/lang_full")
    layers_d = load_result(f"rsa_peak_by_layer/lang_{args.model}")

    # Every part of this figure reads from the by-layer sweep, so panel A's
    # markers, panel B and the inset all share one source. (The separate
    # rsa_by_context file is NOT used: it is a different layer and a different
    # lag smoothing, which made its inset peaks disagree with panel A's.)
    li = layers_d["layers"].tolist().index(PANEL_A_MARKER_LAYER)
    ctx_d = _context_curves(layers_d, li, PANEL_A_MARKER_LAYER)

    # fig_layout() calls apply_rcparams() itself, so the size overrides have to
    # come AFTER it or they are reset before anything is drawn.
    fig, axes = style.fig_layout(rows=2, cols=1, panel_size=(FIG_W_IN, 5.2))
    _F = style.figure_fonts_for_width(FIG_W_IN)
    mpl.rcParams.update(_F)
    for _ax in axes:                      # ticks already exist: set them directly
        _ax.tick_params(axis="both", labelsize=_F["xtick.labelsize"])

    _draw_panel_a(axes[0], erg, layers_d["context_words"], layers_d["is_full"],
                  layers_d["peak_lags_ms"][li], PANEL_A_MARKER_LAYER)
    _draw_panel_b(axes[1], layers_d)

    # set_xlabel/set_ylabel keep the size the axes was created with (unlike
    # set_title, which re-reads rcParams), so resize the labels explicitly.
    for _ax in axes:
        _ax.xaxis.label.set_fontsize(_F["axes.labelsize"])
        _ax.yaxis.label.set_fontsize(_F["axes.labelsize"])

    # Former Fig 5, embedded as an inset in panel A's empty lower-right corner
    # (the ERG curve rises left-to-right, so that quadrant is free).
    # y0 has to leave room below for the inset's own tick labels and x-label,
    # otherwise they drop past panel A's frame onto panel A's tick labels.
    # Nudged right and down from [0.720, 0.140, 0.262, 0.455]: at delta = 100 ms
    # the context markers bunch at the top right and their fanned labels (17, 20)
    # reach far enough left that the inset's opaque background clipped them.
    ax_inset = axes[0].inset_axes([0.758, 0.118, 0.232, 0.430])
    _draw_context_panel(ax_inset, ctx_d, "Language area", ZOOM_LANG,
                        layer=PANEL_A_MARKER_LAYER, fs=INSET_FS,
                        marker_stride=INSET_MARKER_STRIDE)

    out = style.save_svg(fig, "fig5")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
def _draw_panel_a(ax, erg, ctx_words, is_full, peak_lags,
                  marker_layer: int) -> None:
    """ERG curve with numbered context markers placed ON the curve at their
    peak lags, with thin leader lines to labels alternating above/below
    Peak lags are
    taken from the by-layer sweep at `marker_layer`."""
    lags = erg["eval_lags_ms"]
    rho = erg["rho"][0]                              # (1, n_lags) -> (n_lags,)
    color = style.roi_color("lang")

    ax.plot(lags, rho, color=color, linewidth=2.0, label="Language ERG")
    style.add_word_onset(ax)

    keep = (~is_full) & (ctx_words <= MARKED_CONTEXTS_PANEL_A)
    sel_ctx = ctx_words[keep]
    sel_peaks = peak_lags[keep]
    order = np.argsort(sel_ctx)
    sel_ctx = sel_ctx[order]
    sel_peaks = sel_peaks[order]

    # Interpolate curve y at each peak lag - dots sit exactly on the line.
    curve_y = np.interp(sel_peaks.astype(float), lags.astype(float), rho)

    cmap = plt.get_cmap("turbo")
    n = max(1, len(sel_ctx))

    # Many contexts peak at nearly the same lag (e.g. contexts 5-20 within a
    # ~6 ms span), so annotating every one overplots. Group markers by lag:
    # draw ONE big dot per lag and show at most two context numbers per point
    # (the smallest and largest context landing there), one above / one below.
    from collections import OrderedDict
    groups: OrderedDict[int, list] = OrderedDict()
    for i, (w, lag, y_val) in enumerate(zip(sel_ctx.tolist(),
                                            sel_peaks.tolist(),
                                            curve_y.tolist())):
        groups.setdefault(int(round(lag)), []).append((int(w), i, y_val))

    # Per-side memory used to fan out labels whose peaks nearly coincide.
    _MIN_GAP_MS = 1.5     # only labels ~1 ms apart on a side actually collide
    _prev_lag = {True: None, False: None}
    _prev_dx = {True: 0.0, False: 0.0}

    for gi, (lag, members) in enumerate(sorted(groups.items())):
        ws = sorted(m[0] for m in members)
        idx_mean = float(np.mean([m[1] for m in members]))
        y_val = float(np.mean([m[2] for m in members]))
        c = cmap(idx_mean / max(1, n - 1))
        ax.scatter(lag, y_val, color=c, s=140, zorder=15,
                   edgecolors="black", linewidths=0.8)
        labels = [ws[0]] if len(ws) == 1 else [ws[0], ws[-1]]
        for j, w in enumerate(labels):
            # Consecutive markers alternate above / below the curve: their
            # peak lags are only a few ms apart, so a single row collides.
            above = (gi % 2 == 0) if len(labels) == 1 else (j == 0)
            # Still-crowded neighbours on the same side get pushed sideways,
            # which turns their leader into a diagonal (as in the inset).
            if _prev_lag[above] is not None and (lag - _prev_lag[above]) < _MIN_GAP_MS:
                dx = _prev_dx[above] + 18.0
            else:
                dx = 0.0
            _prev_lag[above], _prev_dx[above] = lag, dx
            ax.annotate(
                str(w),
                xy=(lag, y_val),
                xytext=(dx, 17 if above else -17),
                textcoords="offset points",
                fontsize=13, color=_darken(c), fontweight="bold", ha="center",
                va="bottom" if above else "top",
                arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.6),
                zorder=16,
            )

    # Full-context peak (all words) - the endpoint of the progression, marked
    # distinctly with a black star.
    if np.any(is_full):
        full_peak = float(np.asarray(peak_lags)[is_full][0])
        full_y = float(np.interp(full_peak, lags.astype(float), rho))
        ax.scatter(full_peak, full_y, marker="*", s=320, color="#111111",
                   edgecolors="white", linewidths=0.7, zorder=17)
        ax.annotate("full", xy=(full_peak, full_y), xytext=(0, 18),
                    textcoords="offset points", fontsize=14, color="#111111",
                    fontweight="bold", ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#111111",
                                    lw=0.8, alpha=0.6), zorder=18)

    _add_erg_overview_inset(ax, lags, rho, color, PANEL_A_ZOOM_MS)

    xlo, xhi = PANEL_A_ZOOM_MS
    # Curve is monotonically increasing here, so it touches the bottom-left
    # and top-right corners - pad both axes so it visibly enters the frame.
    x_pad = (xhi - xlo) * 0.03
    x_left = xlo - x_pad
    ax.set_xlim(x_left, xhi)
    # ERG is sampled every 25 ms, so [xlo, xhi] may contain no sample exactly at
    # the edges; interpolate the curve at the visible edges so the y-range
    # covers the whole drawn curve (otherwise the low-lag end is clipped).
    in_view = (lags >= x_left) & (lags <= xhi)
    edge_vals = [float(np.interp(x_left, lags.astype(float), rho)),
                 float(np.interp(float(xhi), lags.astype(float), rho))]
    rho_view = np.concatenate([rho[in_view], edge_vals])
    y_min, y_max = float(np.nanmin(rho_view)), float(np.nanmax(rho_view))
    pad = (y_max - y_min) * 0.12 or 0.001
    ax.set_ylim(y_min - pad * 0.7, y_max + pad)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel("ρ_ERG  (full podcast)")
    ax.set_title("Peak lag vs context length on ERG")
    style.panel_label(ax, "A")


# ---------------------------------------------------------------------------
def _add_erg_overview_inset(ax, lags, rho, color, zoom_ms) -> None:
    """Small inset showing the FULL ERG curve, with the main panel's zoom
    window shaded and flagged by two downward arrow-heads at its edges - so a
    reader sees which slice of the whole ERG the main axes magnify."""
    zoom_lo, zoom_hi = zoom_ms
    # Kept small and high: its x tick labels otherwise run into the context
    # markers that climb the ERG curve beneath it.
    axin = ax.inset_axes([0.045, 0.575, 0.285, 0.325])
    axin.plot(lags, rho, color=color, linewidth=1.1, zorder=2)
    axin.axvline(0, color="0.55", linestyle="--", linewidth=0.6, zorder=1)
    axin.axvspan(zoom_lo, zoom_hi, color=color, alpha=0.18, linewidth=0, zorder=1)

    ylo, yhi = float(np.nanmin(rho)), float(np.nanmax(rho))
    span = (yhi - ylo) or 0.001
    axin.set_xlim(float(lags.min()), float(lags.max()))
    axin.set_ylim(ylo - span * 0.15, yhi + span * 0.45)

    # Two arrow-heads pointing down onto the zoom-window edges.
    y_tail = yhi + span * 0.40
    for xb in (zoom_lo, zoom_hi):
        y_head = float(np.interp(float(xb), lags.astype(float), rho))
        axin.annotate("", xy=(xb, y_head), xytext=(xb, y_tail),
                      arrowprops=dict(arrowstyle="-|>", color="black",
                                      lw=1.0, shrinkA=0, shrinkB=0), zorder=5)

    axin.set_xticks([0, 500, 1000])
    axin.tick_params(labelsize=13 * INSET_FS, length=3, pad=2)
    axin.set_yticks([])
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    # Same scale as the context inset's title so the two read as a matched pair.
    axin.set_title("ERG- full range", fontsize=16 * INSET_FS, pad=3)


# ---------------------------------------------------------------------------
def _draw_panel_b(ax, d) -> None:
    layers = d["layers"]
    ctx_words = d["context_words"]
    is_full = d["is_full"]
    peak_lags = d["peak_lags_ms"]                   # (n_layers, n_ctx)

    # Drop 'full' from this panel; cap at MAX_CONTEXT_FOR_PANEL_B.
    keep = (~is_full) & (ctx_words <= MAX_CONTEXT_FOR_PANEL_B) & (ctx_words >= 2)
    x = ctx_words[keep]
    order = np.argsort(x)
    x = x[order]

    cmap = plt.get_cmap("coolwarm")
    layers_list = [int(v) for v in layers.tolist()]
    lmin, lmax = min(layers_list), max(layers_list)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", ">", "<"]

    # Only a representative subset of layers, coloured by their position in the
    # full layer range so the gradient stays meaningful.
    selected = [(i, L) for i, L in enumerate(layers_list)
                if L in SELECTED_LAYERS_PANEL_B]
    for j, (i, layer) in enumerate(selected):
        c = cmap((layer - lmin) / max(1, lmax - lmin))
        y = peak_lags[i][keep][order]
        # Layer 25 is the one panel A's markers and inset come from, so it is
        # drawn markedly heavier than the rest of the sweep.
        is_ref = layer == PANEL_A_MARKER_LAYER
        ax.plot(x, y, color=c, marker=markers[j % len(markers)],
                linewidth=6.5 if is_ref else 2.6,
                markersize=9.0 if is_ref else 6.5,
                markeredgecolor="black", markeredgewidth=0.4,
                label=f"Layer {layer}", zorder=4 if is_ref else 3)
        # Full-context peak lag for this layer: faint horizontal line, same hue.
        if np.any(is_full):
            full_peak = float(peak_lags[i][is_full][0])
            ax.axhline(full_peak, color=c, alpha=0.5, linestyle="--",
                       linewidth=1.4, zorder=1)

    ax.set_xlabel("Context length (words)")
    ax.set_ylabel("Peak lag (ms)")
    ax.set_title("Peak lag vs context length across LLM layers")
    ax.legend(loc="lower right", ncol=2, fontsize=11.5, framealpha=0.95,
              handlelength=1.4, columnspacing=1.0, labelspacing=0.3,
              borderpad=0.35)
    style.panel_label(ax, "B")


def _darken(rgba, factor: float = 0.62):
    """Darker shade of a marker colour, for its text label.

    The mid-gradient hues (yellow/lime) are nearly unreadable on white at
    full brightness; darkening only the digits keeps the colour coding while
    giving the text enough contrast.
    """
    r, g, b = rgba[:3]
    return (r * factor, g * factor, b * factor, 1.0)


def _context_curves(layers_d, layer_index: int, layer: int) -> dict:
    """Per-context curves at one layer, shaped like the rsa_by_context result.

    `rsa_peak_by_layer` stores the smoothed curves for every layer, so the
    inset can be drawn at the same layer (and the same lag smoothing) as
    panel A's markers without loading a second, inconsistent file.
    """
    smoothed = np.asarray(layers_d["curves_smoothed"][layer_index])
    return {"eval_lags_ms": layers_d["eval_lags_ms"],
            "curves_peak_norm": smoothed / np.nanmax(smoothed, axis=1,
                                                     keepdims=True),
            "context_words": layers_d["context_words"],
            "is_full": layers_d["is_full"],
            "layer": layer}


def _cmap_obj():
    """`jet` truncated at CMAP_HI so the warm end stays a bright red."""
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.get_cmap(CMAP)
    return LinearSegmentedColormap.from_list(
        "jet_hi", base(np.linspace(0.0, CMAP_HI, 256)))

def _color_for(ctx_words: np.ndarray, is_full: np.ndarray):
    """Map context size to color on a LOG scale so each window gets a distinct
    hue even when small contexts dominate the list (ctx=1 -> deep blue, ctx=10
    -> green, ctx=50 -> red).
    """
    finite = ctx_words[~is_full]
    norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))
    cmap_obj = _cmap_obj()
    colors = []
    for w, full in zip(ctx_words, is_full):
        if full:
            # Full context is the LONGEST window: give it the colormap's hot
            # end (a near-black red reads like the short/cool end instead).
            colors.append(cmap_obj(1.0))
        else:
            colors.append(cmap_obj(norm(float(w))))
    return colors, norm, cmap_obj

def _add_colorbar(fig, ax, ctx_words, is_full, label="Context length (words)",
                  fs: float = 1.0):
    from matplotlib.ticker import FixedLocator, NullFormatter
    finite = ctx_words[~is_full]
    vmin = float(finite.min())
    vmax = float(finite.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=_cmap_obj(), norm=norm)
    if fs < 1.0:
        # Inset use: `ax=` would shrink the (already small) axes and drop the
        # bar on top of the curves, so give the bar its own slot beside them.
        # Label the extremes only - a full axis label sits too far out.
        cax = ax.inset_axes([1.015, 0.0, 0.045, 1.0])
        cbar = fig.colorbar(sm, cax=cax)
        from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
        cbar.ax.yaxis.set_minor_locator(NullLocator())     # kill 3x10^0 etc.
        cbar.ax.yaxis.set_minor_formatter(NullFormatter())
        cbar.ax.yaxis.set_major_locator(FixedLocator([vmin, vmax]))
        cbar.ax.yaxis.set_major_formatter(
            FixedFormatter([f"{int(vmin)}", f"{int(vmax)}"]))
        cbar.ax.tick_params(labelsize=12 * fs, length=2, pad=1.5)
        cbar.outline.set_linewidth(0.6)
        return cbar

    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label(label, labelpad=4, fontsize=17 * fs)
    cbar.ax.tick_params(labelsize=12 * fs)
    cbar.outline.set_linewidth(0.6)
    # Minor tick at every integer from vmin to vmax - log-spaced positions
    # like 1, 2, 3, ..., 50 (not just the contexts we sampled).
    int_ticks = list(range(int(np.ceil(vmin)), int(np.floor(vmax)) + 1))
    cbar.ax.yaxis.set_minor_locator(FixedLocator(int_ticks))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())   # tick marks only
    cbar.ax.tick_params(which="minor", length=3, width=0.5, color="black",
                        labelleft=False, labelright=False)
    return cbar

def _draw_context_panel(ax, d, roi_name: str, xlim: tuple[int, int], *,
                      layer: int, letter: str | None = None,
                      exclude_contexts: tuple[int, ...] = (),
                      marker_stride: int = 1,
                      fs: float = 1.0) -> None:
    """Peak-normalised ρ(lag) per context window, zoomed to `xlim`.

    Drawn as an inset inside panel A. `fs` scales every font and marker size
    down for that smaller use.
    """
    lags = d["eval_lags_ms"]
    curves = d["curves_peak_norm"]
    ctx_words = d["context_words"]
    is_full = d["is_full"]

    # The full-context curve sits far outside the 2-20 word colour scale, so it
    # is drawn separately as a dotted strong-red line rather than being given a
    # colour that would be read against the colourbar.
    full_curve = curves[is_full][0] if np.any(is_full) else None
    keep_finite = ~is_full
    ctx_words = ctx_words[keep_finite]
    is_full = is_full[keep_finite]
    curves = curves[keep_finite]

    # Optional per-ROI filter: drop named context windows from the plot only
    # (the data on disk is unchanged).
    if exclude_contexts:
        keep = np.array([
            (w not in exclude_contexts) or full
            for w, full in zip(ctx_words.tolist(), is_full.tolist())
        ])
        ctx_words = ctx_words[keep]
        is_full = is_full[keep]
        curves = curves[keep]

    colors, _, _ = _color_for(ctx_words, is_full)
    mask = (lags >= xlim[0]) & (lags <= xlim[1])
    lags_z = lags[mask]
    for ci in range(curves.shape[0]):
        c = curves[ci]
        if not np.any(np.isfinite(c)):
            continue
        lw = 1.6 if is_full[ci] else 0.9
        ax.plot(lags_z, c[mask], color=colors[ci], linewidth=lw, alpha=0.9)

    # Full context: dotted strong red with large, clearly separated dots.
    if full_curve is not None:
        ax.plot(lags_z, full_curve[mask], color=FULL_COLOR, linewidth=2.6,
                linestyle=(0, (1, 1.6)), dash_capstyle="round", zorder=4,
                label="full context")

    # Triangle markers a touch above the curves. Stack when peaks coincide;
    # skip curves whose in-zoom argmax falls on the mask boundary (those
    # actually peak OUTSIDE the zoom - drawing a triangle at the edge is
    # misleading).
    from collections import defaultdict
    groups: dict[float, list[int]] = defaultdict(list)
    n_in_mask = int(mask.sum())
    for ci in range(curves.shape[0]):
        c = curves[ci]
        if not np.any(np.isfinite(c[mask])):
            continue
        peak_idx = int(np.nanargmax(c[mask]))
        # Edge of the displayed zoom = the smoothed curve still climbing or
        # still falling at the boundary; the real peak is elsewhere.
        if peak_idx == 0 or peak_idx == n_in_mask - 1:
            continue
        peak_lag = float(lags_z[peak_idx])
        groups[peak_lag].append(ci)

    # Triangles sit on one line, thinned: markers closer than min_dx to one
    # already drawn are dropped, so none is plotted on top of another.
    triangle_y = 1.009
    label_y = 1.036                           # lag numbers sit on one line
    min_dx = (xlim[1] - xlim[0]) * 0.012
    drawn: list[tuple[float, object, bool]] = []   # (lag, colour, is_full)
    for peak_lag in sorted(groups):
        if any(abs(peak_lag - x) < min_dx for x, _, _ in drawn):
            continue
        ci = groups[peak_lag][0]              # one marker per retained lag
        drawn.append((peak_lag, colors[ci], False))

    # Thin further by index when the retained peaks are still too dense for
    # their lag labels to be legible (keep every `marker_stride`-th).
    if marker_stride > 1:
        drawn = drawn[::marker_stride]
        # The largest context is the endpoint of the progression, so keep it
        # even if the stride skipped it - swap it in for the last one kept.
        ci_max = int(np.argmax(ctx_words))
        lag_max = next((lag for lag, cis in groups.items() if ci_max in cis), None)
        if lag_max is not None and not any(l == lag_max for l, _, _ in drawn):
            if drawn:
                drawn[-1] = (lag_max, colors[ci_max], False)
            else:
                drawn.append((lag_max, colors[ci_max], False))

    # Full-context peak gets its own marker on the same row.
    if full_curve is not None:
        fc = full_curve[mask]
        fpk = int(np.nanargmax(fc))
        if 0 < fpk < int(mask.sum()) - 1:
            drawn.append((float(lags_z[fpk]), FULL_COLOR, True))
    drawn.sort(key=lambda t: t[0])

    for peak_lag, col, _ in drawn:
        ax.scatter(peak_lag, triangle_y, marker="v", color=col, s=210 * fs**2,
                   edgecolor="black", linewidth=0.5, zorder=5, clip_on=False)

    # Lag values on one line above the row, fanned out and joined to their
    # markers by thin black leaders.
    n_lab = len(drawn)
    if n_lab:
        # The full label is pinned above its own marker (short leader), so the
        # remaining labels are fanned across the space to its LEFT only -
        # otherwise the last fanned label collides with it.
        span = (xlim[1] - xlim[0]) * 0.53
        full_lag = next((lag for lag, _, f in drawn if f), None)
        right_edge = (full_lag - (xlim[1] - xlim[0]) * 0.11
                      if full_lag is not None else xlim[1])
        n_fan = sum(1 for _, _, f in drawn if not f)
        fan = np.linspace(right_edge - span, right_edge, max(n_fan, 1))
        label_x, k = [], 0
        for _, _, is_f in drawn:
            if is_f:
                label_x.append(np.nan)          # replaced by the marker's lag
            else:
                label_x.append(fan[k]); k += 1
        for (peak_lag, col, is_full_marker), lx in zip(drawn, label_x):
            # The full-context label sits straight above its own marker, so
            # its leader stays short instead of reaching across the fan.
            # The full-context number is raised so that "(full)", set flush
            # beneath it, still clears the marker instead of sitting on it.
            ly = label_y + (0.016 if is_full_marker else 0.0)
            if is_full_marker:
                lx = peak_lag
            ax.annotate(f"{int(round(peak_lag))}",
                        xy=(peak_lag, triangle_y + 0.005),
                        xytext=(lx, ly),
                        ha="center", va="bottom", fontsize=11 * fs, color="black",
                        annotation_clip=False, zorder=6,
                        arrowprops=dict(arrowstyle="-", color="black",
                                        linewidth=0.6,
                                        shrinkA=1.0, shrinkB=1.0))
            if is_full_marker:
                ax.annotate("(full)", xy=(lx, ly), xycoords="data",
                            xytext=(0, -1.0), textcoords="offset points",
                            ha="center", va="top", fontsize=11 * fs,
                            color="black", zorder=6, annotation_clip=False)

    ax.set_xlim(*xlim)
    # Extend the y-axis a bit so stacked triangles fit between the curves
    # and the title; the curves themselves still sit in [0.80, 1.0].
    ax.set_ylim(0.80, 1.062)
    ax.set_xlabel("Lag from word onset (ms)", fontsize=17 * fs)
    ax.set_ylabel("Spearman corr (norm)", fontsize=17 * fs,
                  labelpad=2.0 if fs < 1.0 else None)
    # No title when this panel is embedded as an inset. It is centred at x=0.42,
    # so it hangs well past the inset's left edge and lands on panel A's own
    # context-marker labels (17-20 sit exactly there once the ERG curve is
    # computed at delta = 100 ms). The figure caption already says what the
    # inset shows. Standalone use keeps the title.
    if fs >= 1.0:
        ax.set_title("Temporal RSA Lag vs LLM context", fontsize=16 * fs,
                     loc="center", x=0.42, pad=22)
    if letter is not None:
        style.panel_label(ax, letter)
    ax.tick_params(axis="both", labelsize=13 * fs)
    _add_colorbar(ax.figure, ax, ctx_words, is_full, fs=fs)

if __name__ == "__main__":
    main()
