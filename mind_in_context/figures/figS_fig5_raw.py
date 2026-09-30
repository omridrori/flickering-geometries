"""Extended Data Fig. 6 - Fig 5 family of context-length RSA curves shown
*non-normalized* (raw Spearman ρ) over the full evaluated lag range.

The main figure (Fig 5) zooms into each ROI's peak window and peak-
normalizes every curve so context-driven peak shifts are easy to see.
This supplement shows the raw smoothed ρ at every context length, full
lag range, so the underlying magnitudes (and their decay with shrinking
context) are visible without normalization.

Loads:
  results/rsa_by_context/lang_<model>.npz
  results/rsa_by_context/aud_<model>.npz

Output: plots/figS_fig5_raw.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullFormatter

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

CMAP = "jet"


def _color_for(ctx_words: np.ndarray, is_full: np.ndarray):
    finite = ctx_words[~is_full]
    norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))
    cmap_obj = plt.get_cmap(CMAP)
    colors = []
    for w, full in zip(ctx_words, is_full):
        if full:
            colors.append("#3A0000")
        else:
            colors.append(cmap_obj(norm(float(w))))
    return colors


def _add_colorbar(fig, ax, ctx_words, is_full,
                  label="Context window (words)"):
    finite = ctx_words[~is_full]
    norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(CMAP), norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label(label, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    int_ticks = list(range(int(np.ceil(norm.vmin)),
                           int(np.floor(norm.vmax)) + 1))
    cbar.ax.yaxis.set_minor_locator(FixedLocator(int_ticks))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar.ax.tick_params(which="minor", length=3, width=0.5, color="black")


def _draw_raw(ax, d, roi_name: str, *, letter: str, layer: int) -> None:
    lags = d["eval_lags_ms"]
    curves = d["curves_raw"]          # no lag-axis smoothing, no peak-norm
    ctx_words = d["context_words"]
    is_full = d["is_full"]

    colors = _color_for(ctx_words, is_full)
    for ci in range(curves.shape[0]):
        c = curves[ci]
        if not np.any(np.isfinite(c)):
            continue
        lw = 1.6 if is_full[ci] else 0.9
        alpha = 0.95 if is_full[ci] else 0.85
        ax.plot(lags, c, color=colors[ci], linewidth=lw, alpha=alpha)

    style.add_word_onset(ax)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"{roi_name} | layer {layer} | raw ρ(lag) per context  "
                 f"(no lag smoothing)")
    style.panel_label(ax, letter)
    _add_colorbar(ax.figure, ax, ctx_words, is_full)


def main() -> None:
    parser = make_argparser(__doc__, with_model=True)
    args = parser.parse_args()

    lang = load_result(f"rsa_by_context/lang_{args.model}")
    aud  = load_result(f"rsa_by_context/aud_{args.model}")
    layer = int(lang["layer"])

    style.apply_rcparams()
    fig, axes = style.fig_layout(rows=1, cols=2, panel_size=(7.0, 5.0))
    _draw_raw(axes[0], lang, "Language area", letter="A", layer=layer)
    _draw_raw(axes[1], aud,  "Auditory area", letter="B", layer=layer)

    out = style.save_svg(fig, "figS_fig5_raw")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
