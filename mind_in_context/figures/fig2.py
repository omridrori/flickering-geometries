"""Figure 2 - Segment-resolved view of the ultra-fast transformation (language).

Layout - two framed super-panels on top (A = ERG left, B = PSTH right), the
per-segment matrices framed as C beneath. Within each block the mean ± SEM
panel sits above the per-segment curves:

  +===== A : ERG ==========+  +===== B : PSTH =========+
  | mean ±SEM  | avg.      |  | mean ±SEM  | amp.      |
  |            | scatter   |  |            | scatter   |
  | curves     | slope     |  | curves     | slope     |
  | (5 seg)    | scatter   |  | (5 seg)    | scatter   |
  +========================+  +========================+
  +===== C ================================================+
  | matrix1 | matrix2 | matrix3 | matrix4 | matrix5        |
  +========================================================+

The curves use 5 eqword segments; the scatters use 20 eqword
segments (more points for the trend test). Both come from segment_erg_psth
(baseline ON). The scatter r-boxes report the Pearson r with its p-value
(or 'n.s.'). PSTH is min-max normalised (curves over the 5, scatter over the 20).

Loads:
  results/segment_erg_psth/lang_5seg.npz    (ERG/PSTH curves, 5 eqword segments)
  results/segment_erg_psth/lang_20seg.npz   (ERG/PSTH scatter, 20 eqword segments)
  results/rdm_similarity_2d/lang.npz        (Δ-residual matrices, 5 segments)

Reproduce the inputs with:
  python -m mind_in_context.analyses.segment_erg_psth --roi lang --n-segments 5
  python -m mind_in_context.analyses.segment_erg_psth --roi lang --n-segments 20

Output: plots/fig2.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pearsonr

from mind_in_context.lib import style
from mind_in_context.lib.io import load_config, load_result, make_argparser
from mind_in_context.lib.stats import normalize_to_0_100

# Fonts are enlarged for this dense, wide multi-panel figure.
# savefig.bbox="tight" (project default) crops back to the outermost artists -
# here the super-panel frames - so the white border has to come from
# savefig.pad_inches, not from the layout `rect`.
_FONTS = {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
          "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
          "savefig.pad_inches": 0.32}


def main() -> None:
    parser = make_argparser(__doc__)
    parser.add_argument("--delta", type=int, default=None,
                        help="ERG gap Δ in ms. Defaults to the published "
                             "analysis; a non-default value loads the "
                             "'_d<delta>' results and writes fig2_d<delta>.svg.")
    args = parser.parse_args()

    # The published Δ lives with the analysis that produced the results, so
    # that figure and results agree on which file is the default one.
    from mind_in_context.analyses.segment_erg_psth import DELTA_MS
    delta_ms = DELTA_MS if args.delta is None else int(args.delta)
    tag = "" if delta_ms == DELTA_MS else f"_d{delta_ms}"

    curves = load_result(f"segment_erg_psth/lang_5seg{tag}")   # 5 eqword segments
    scatter = load_result(f"segment_erg_psth/lang_20seg{tag}")  # 20 eqword segments
    rdm2d = load_result("rdm_similarity_2d/lang")          # 5-segment matrices

    erg_d = {"eval_lags_ms": curves["eval_lags_ms"], "rho": curves["erg"]}
    # PSTH curves normalised across the 5 segments together (global 0-100).
    psth_norm, _, _ = normalize_to_0_100(curves["psth"])
    psth_d = {"eval_lags_ms": curves["eval_lags_ms"], "psth": psth_norm}
    # Scatter (20 points). PSTH amplitude normalised 0-100 over the 20 segments.
    psth_mean_norm, _, _ = normalize_to_0_100(scatter["psth_mean"])
    bars = {"erg_mean": scatter["erg_mean"],
            "erg_slope_per_ms": scatter["erg_slope_per_ms"],
            "psth_mean": psth_mean_norm,
            "psth_slope_per_ms": scatter["psth_slope_per_ms"]}

    style.apply_rcparams()
    mpl.rcParams.update(_FONTS)
    fig = plt.figure(figsize=(17.5, 10.6), constrained_layout=True)
    # `rect` reserves a margin all round (and a little extra on top) so every
    # super-panel frame is inset from the figure edge and its letter has room.
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.10, hspace=0.10,
                                wspace=0.06,
                                # leaves room for frame pad + outer margin
                                rect=(0.071, 0.071, 0.858, 0.836))
    # Row 1 is an empty spacer that separates the top blocks from the matrices,
    # leaving room for super-panel C's letter.
    gs_outer = GridSpec(3, 1, figure=fig, height_ratios=[2.0, 0.081, 2.0])
    # Column 2 is an empty spacer that separates super-panel A from B.
    gs_top = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs_outer[0],
                                     # curves get the wider column in both
                                     # blocks: A = [curves, scatter], spacer,
                                     # B = [curves, scatter]
                                     width_ratios=[1.4, 1.0, 0.68, 1.4, 1.0])
    gs_bot = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_outer[2])

    # ----- Super-panel A : ERG, on the LEFT (curves + scatter) --------------
    # The mean +/- SEM panel sits on TOP with the per-segment curves beneath it.
    ax_B = fig.add_subplot(gs_top[0, 0])          # mean, top row
    ax_A = fig.add_subplot(gs_top[1, 0])          # segments, bottom row
    _draw_curves(ax_A, ax_B, erg_d, ylabel="ERG (corr)")
    ax_C = fig.add_subplot(gs_top[0, 1])
    ax_D = fig.add_subplot(gs_top[1, 1])
    _draw_scatter(ax_C, ax_D, bars, kind="erg")

    # ----- Super-panel B : PSTH, on the RIGHT (curves + scatter) ------------
    # Same internal order as A: curves on the block's left, scatters on its right.
    ax_H = fig.add_subplot(gs_top[0, 3])          # mean, top row
    ax_G = fig.add_subplot(gs_top[1, 3])          # segments, bottom row
    _draw_curves(ax_G, ax_H, psth_d, ylabel="PSTH (norm.)",
                 curves_key="psth", norm_mean_separately=True)
    ax_E = fig.add_subplot(gs_top[0, 4])
    ax_F = fig.add_subplot(gs_top[1, 4])
    _draw_scatter(ax_E, ax_F, bars, kind="psth")

    # Titles live in the caption (Nature style); axis names are concise.
    for ax in (ax_A, ax_B, ax_C, ax_D, ax_E, ax_F, ax_G, ax_H):
        _clear_title(ax)
    # Tick range follows the actual segment count - a fixed 0..20 leaves half
    # the panel empty whenever the scatter is run with fewer segments.
    n_seg = int(np.asarray(bars["erg_mean"]).size)
    tick_step = 5 if n_seg > 12 else 2
    for ax in (ax_C, ax_D, ax_E, ax_F):
        ax.set_xlabel("Segment (early → late)")
        # draw_segment_scatter ticks every 2 segments; too many for these
        # narrow panels, so label more sparsely instead.
        ax.set_xticks(np.arange(0, n_seg + 1, tick_step))
    ax_C.set_ylabel("ERG avg. (corr)")
    ax_D.set_ylabel("ERG Slope (Δcorr)")
    ax_E.set_ylabel("PSTH Amp. (a.u.)")
    ax_F.set_ylabel("PSTH Slope (a.u.)")
    _fade_regression(ax_C, ax_D, ax_E, ax_F)

    # ----- Super-panel C : 5 Δ-residual matrices ----------------------------
    matrix_axes = _draw_matrices(fig, gs_bot, rdm2d)

    # Freeze the constrained layout so the r-box measurements below (which
    # depend on rendered text extents) stay valid through the final render.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    _annotate_rp(ax_C, bars["erg_mean"])
    _annotate_rp(ax_D, bars["erg_slope_per_ms"])
    _annotate_rp(ax_E, bars["psth_mean"])
    _annotate_rp(ax_F, bars["psth_slope_per_ms"])

    _draw_super_panels(fig, left_axes=(ax_A, ax_B, ax_C, ax_D),
                       right_axes=(ax_E, ax_F, ax_G, ax_H),
                       matrix_axes=matrix_axes)

    out = style.save_svg(fig, f"fig2{tag}")
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
def _draw_curves(ax_seg, ax_mean, d, *, ylabel: str, curves_key: str = "rho",
                 norm_mean_separately: bool = False) -> None:
    """Per-segment curves (colour = early->late) and their mean ± 1 SEM.

    The two axes are passed explicitly because the mean is drawn in the upper
    row of the figure and the segments beneath it.
    """
    lags = d["eval_lags_ms"]
    seg_curves = d[curves_key]                    # (n_segments, n_lags)
    n_seg = seg_curves.shape[0]

    for k in range(n_seg):
        ax_seg.plot(lags, seg_curves[k], color=style.segment_color(k, n_seg),
                    linewidth=1.8, label=f"seg {k+1}")
    _weak_onset(ax_seg)
    ax_seg.set_xlabel("Lag from word onset (ms)")
    ax_seg.set_ylabel(ylabel)
    ax_seg.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
                  ncol=min(n_seg, 5), fontsize=9, frameon=False,
                  columnspacing=1.1, handlelength=1.4, handletextpad=0.4)

    mean = seg_curves.mean(axis=0)
    sem = seg_curves.std(axis=0, ddof=1) / np.sqrt(n_seg)
    if norm_mean_separately:                      # H spans its own 0-100
        lo, hi = mean.min(), mean.max()
        scale = 100.0 / (hi - lo)
        mean = (mean - lo) * scale
        sem = sem * scale
    ax_mean.fill_between(lags, mean - sem, mean + sem,
                        color="black", alpha=0.18, linewidth=0)
    ax_mean.plot(lags, mean, color="black", linewidth=2.0)
    _weak_onset(ax_mean)
    ax_mean.set_xlabel("Lag (ms)")
    ax_mean.set_ylabel(ylabel)


def _draw_scatter(ax_top, ax_bot, bars, *, kind: str) -> None:
    """Per-segment scatter + faint regression for mean (top) and slope (bottom)."""
    if kind == "erg":
        mean, slope = bars["erg_mean"], bars["erg_slope_per_ms"] * 1000.0
    else:
        mean, slope = bars["psth_mean"], bars["psth_slope_per_ms"] * 1000.0
    # y-labels/titles are set by the caller; pass through segment_color scatter.
    style.draw_segment_scatter(ax_top, mean, ylabel="", title="")
    style.draw_segment_scatter(ax_bot, slope, ylabel="", title="")
    # Headroom so the r/p box sits in empty space instead of over the points.
    for ax in (ax_top, ax_bot):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.28)


def _draw_matrices(fig, gs_bot, rdm2d) -> list:
    """Bottom row: one Δ-residual RDM-similarity matrix per segment.

    Returns every axes it created (matrices + colourbar) so the super-panel
    frame can be drawn around the whole block.
    """
    created = []
    lags = rdm2d["eval_lags_ms"]
    R = rdm2d["R"]                                # (n_seg, n_lags, n_lags)
    n_seg = R.shape[0]
    # origin="lower" so lag t1 increases upward and the diagonal runs
    # bottom-left -> top-right, matching the lag×lag panels of fig3.
    extent = (lags[0], lags[-1], lags[0], lags[-1])
    r_lim = float(np.percentile(np.abs(R), 95))

    for k in range(n_seg):
        ax = fig.add_subplot(gs_bot[0, k])
        created.append(ax)
        im = ax.imshow(R[k], extent=extent, cmap="RdBu_r", origin="lower",
                       vmin=-r_lim, vmax=r_lim, aspect="equal")
        ax.set_xlabel("lag t2 (ms)")
        ax.set_title(f"seg {k+1}", loc="center", fontsize=14,
                     fontweight="bold", color="black", pad=6)
        if k == 0:
            ax.set_ylabel("lag t1 (ms)")
        for spine in ax.spines.values():          # thick segment-gradient frame
            spine.set_visible(True)
            spine.set_color(style.segment_color(k, n_seg))
            spine.set_linewidth(4.5)
        if k == n_seg - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                ticks=[-r_lim, 0, r_lim], format="%.2f")
            created.append(cbar.ax)
    return created


# ---------------------------------------------------------------------------
# Small annotation helpers (figure-specific)
# ---------------------------------------------------------------------------
def _weak_onset(ax) -> None:
    """Faint dashed word-onset line (lighter than style.add_word_onset)."""
    ax.axvline(0, color=style.WORD_ONSET_COLOR, linestyle="--",
               linewidth=0.8, alpha=0.45)


def _clear_title(ax) -> None:
    """Remove a panel's title (rcParams put it in the 'left' slot)."""
    for loc in ("left", "center", "right"):
        ax.set_title("", loc=loc)


def _fade_regression(*axes) -> None:
    """Fade each scatter's regression line to barely visible; drop its legend."""
    for ax in axes:
        for ln in ax.get_lines():                 # the only line is the fit
            ln.set_alpha(0.18)
            ln.set_linewidth(0.8)
        legend = ax.get_legend()                  # the equation legend
        if legend is not None:
            legend.remove()


def _annotate_rp(ax, values) -> None:
    """Set the scatter r-box: 'r = .. (p = ..)' with a smaller p inside the box
    when significant, else 'n.s.'. Requires a frozen layout (see main)."""
    values = np.asarray(values, dtype=float)
    r, p = pearsonr(np.arange(1, values.size + 1), values)
    if not ax.texts:
        return
    r_txt = ax.texts[0]
    # Small enough to fit these narrow panels on one line; the extra headroom
    # added in _draw_scatter keeps it clear of the points. draw_segment_scatter
    # sets it in monospace, which pads out the spaces around "=".
    r_txt.set_fontsize(11)
    r_txt.set_family("sans-serif")
    r_txt.set_position((0.03, 0.98))
    r_txt.set_ha("left")
    if p >= 0.05:
        r_txt.set_text("n.s.")
        return

    r_txt.set_text(f"r={r:+.3f}")
    r_txt.set_bbox(None)                           # own box is drawn around r+p
    r_txt.set_zorder(6)
    p_txt = "p<0.001" if p < 1e-3 else f"p={p:.3f}"
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    bb_r = r_txt.get_window_extent(renderer)
    x_after = inv.transform((bb_r.x1, bb_r.y0))[0]
    y_center = inv.transform(((bb_r.x0 + bb_r.x1) / 2,
                              (bb_r.y0 + bb_r.y1) / 2))[1]
    p_artist = ax.text(x_after + 0.008, y_center, f"({p_txt})",
                       transform=ax.transAxes, fontsize=8.5, va="center",
                       ha="left", color="black", zorder=6)
    fig.canvas.draw()
    bb_r = r_txt.get_window_extent(renderer)
    bb_p = p_artist.get_window_extent(renderer)
    xs = [bb_r.x0, bb_r.x1, bb_p.x0, bb_p.x1]
    ys = [bb_r.y0, bb_r.y1, bb_p.y0, bb_p.y1]
    x0, y0 = inv.transform((min(xs), min(ys)))
    x1, y1 = inv.transform((max(xs), max(ys)))
    # Right-aligned r would push the p-value off the panel: slide the pair back
    # in and re-measure before the box is drawn around them.
    overflow = x1 - 0.965
    if overflow > 0:
        for artist in (r_txt, p_artist):
            ax_x, ax_y = artist.get_position()
            artist.set_position((ax_x - overflow, ax_y))
        fig.canvas.draw()
        bb_r = r_txt.get_window_extent(renderer)
        bb_p = p_artist.get_window_extent(renderer)
        xs = [bb_r.x0, bb_r.x1, bb_p.x0, bb_p.x1]
        ys = [bb_r.y0, bb_r.y1, bb_p.y0, bb_p.y1]
        x0, y0 = inv.transform((min(xs), min(ys)))
        x1, y1 = inv.transform((max(xs), max(ys)))
    pad_x, pad_y = 0.018, 0.020
    ax.add_patch(FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y), (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y,
        transform=ax.transAxes, boxstyle="round,pad=0,rounding_size=0.025",
        linewidth=0.6, edgecolor="#CCCCCC", facecolor="white", zorder=5))


def _draw_super_panels(fig, *, left_axes, right_axes, matrix_axes) -> None:
    """Frame each super-panel block (A = left, B = right, C = matrices) and put
    its bold letter inside the frame's top margin."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    # `pad_top` leaves room for the block letter, which sits inside the frame.
    pad, pad_top, margin = 0.026, 0.090, 0.045

    def bounds(axes, top):
        bbs = [ax.get_tightbbox(renderer).transformed(inv) for ax in axes]
        return (min(b.x0 for b in bbs) - pad, min(b.y0 for b in bbs) - pad,
                max(b.x1 for b in bbs) + pad, max(b.y1 for b in bbs) + top)

    # A and B need the taller top margin because their letters sit above a
    # full-height y-label; C's letter has nothing to clear, so it uses less.
    blocks = {"A": bounds(left_axes, pad_top),
              "B": bounds(right_axes, pad_top)}
    if matrix_axes:
        blocks["C"] = bounds(matrix_axes, 0.030)

    # Share one left edge (A, C) and one right edge (B, C), both held off the
    # figure boundary, so no frame touches an edge and they line up.
    left = max(min(b[0] for b in blocks.values()), margin)
    right = min(max(b[2] for b in blocks.values()), 1.0 - margin)
    for key, (x0, y0, x1, y1) in blocks.items():
        blocks[key] = (left if key in ("A", "C") else x0, y0,
                       right if key in ("B", "C") else x1, y1)

    for text, (x0, y0, x1, y1) in blocks.items():
        # Keep every side off the figure boundary so no frame line is clipped.
        x0, x1 = max(x0, margin), min(x1, 1.0 - margin)
        y0, y1 = max(y0, margin), min(y1, 1.0 - margin)
        fig.add_artist(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.008",
            facecolor="none", edgecolor="#555555", linewidth=2.2, zorder=0.5))
        # Hard against the frame's left inset: any further right and the letter
        # sits directly over the first panel's (rotated, full-height) y-label.
        fig.text(x0 + 0.003, y1 - 0.004, text, fontsize=24, fontweight="bold",
                 color="black", ha="left", va="top", zorder=2.0)


if __name__ == "__main__":
    main()
