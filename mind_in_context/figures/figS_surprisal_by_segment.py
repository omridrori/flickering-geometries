"""Mean surprisal per narrative segment - Supplementary Fig. 2.

Loads results/surprisal_by_segment/seg<N>.npz and draws one panel: segment
mean ± s.e.m. against segment position, with the trend statistic in the title.

Usage
-----
    python figS_surprisal_by_segment.py [--n-segments 20]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

PANEL_IN = (6.0, 3.4)
C_ALL_WORDS = "#7A7A7A"   # the "all words" grey of Fig. 3


def _draw(ax, seg_mean, seg_sem, r, p, n_perm) -> None:
    pos = np.arange(1, seg_mean.size + 1)
    color = C_ALL_WORDS
    ax.errorbar(pos, seg_mean, yerr=seg_sem, fmt="o-", color=color,
                ecolor=style.ZERO_LINE_COLOR, elinewidth=1.0, capsize=2.5,
                ms=4.5, lw=1.4)
    ax.set_xlabel("Narrative segment (early → late)")
    ax.set_ylabel("Mean surprisal (bits)")
    ax.set_xticks([1, 5, 10, 15, 20] if seg_mean.size == 20 else pos)
    lo = float((seg_mean - seg_sem).min()); hi = float((seg_mean + seg_sem).max())
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"r = {r:+.2f}, p = {p:.2f} ({n_perm:,} permutations)")


def main() -> None:
    parser = make_argparser(__doc__)
    parser.add_argument("--n-segments", type=int, default=20)
    args = parser.parse_args()
    d = load_result(f"surprisal_by_segment/seg{args.n_segments}")
    fonts = style.figure_fonts_for_width(PANEL_IN[0])
    fig, axes = style.fig_layout(1, 1, panel_size=PANEL_IN, fonts=fonts)
    _draw(axes[0], d["seg_mean"], d["seg_sem"], float(d["r"]),
          float(d["p_perm_two_sided"]), int(d["n_perm"]))
    out = style.save_svg(fig, "figS_surprisal_by_segment")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
