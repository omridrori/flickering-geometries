"""Extended Data Fig. 4 - surprisal selectivity survives removing repeated words.

The narrative repeats itself, and the predictable words are largely
high-frequency function words that recur throughout it, so the split in
Figure 3 could in principle be a repetition effect rather than a surprisal
effect. Keeping only the first occurrence of each word form removes that
possibility, and the separation between the two curves is still there.

Loads:  results/surprisal_split_dedup/<roi>.npz
Output: plots/figS_surprisal_dedup.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

ZOOM = (-500, 1000)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    args = parser.parse_args()

    d = load_result(f"surprisal_split_dedup/{args.roi}")
    lags = np.asarray(d["eval_lags_ms"])
    keep = (lags >= ZOOM[0]) & (lags <= ZOOM[1])
    x = lags[keep]

    fig, axes = style.fig_layout(1, 1, panel_size=(6.0, 4.2))
    ax = axes[0]
    style.add_word_onset(ax)

    ax.plot(x, np.asarray(d["rho_all"])[keep], color="#7A7A7A", linewidth=2.2,
            alpha=0.85, label=f"All forms (n = {int(d['n_all'])})")
    ax.plot(x, np.asarray(d["rho_low"])[keep],
            color=style.SURPRISAL_COLORS["Q1"], linewidth=2.4,
            label=f"Low surprisal (n = {int(d['n_low'])}, "
                  f"{float(d['mean_low']):.2f} bits)")
    ax.plot(x, np.asarray(d["rho_high"])[keep],
            color=style.SURPRISAL_COLORS["Q5"], linewidth=2.4,
            label=f"High surprisal (n = {int(d['n_high'])}, "
                  f"{float(d['mean_high']):.2f} bits)")

    ax.set_xlim(*ZOOM)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel("ERG stability  ρ")
    ax.set_title(f"{style.roi_display_name(args.roi)} — one occurrence of each "
                 f"word form", fontsize=9)
    legend = ax.legend(loc="lower right", framealpha=0.95)
    for line in legend.get_lines():
        line.set_linewidth(4.0)

    fig.suptitle(
        f"{int(d['n_all'])} distinct forms out of {int(d['n_words'])} tokens; "
        f"Δ = {int(d['delta_ms'])} ms", fontsize=10)

    out = style.save_svg(fig, "figS_surprisal_dedup")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
