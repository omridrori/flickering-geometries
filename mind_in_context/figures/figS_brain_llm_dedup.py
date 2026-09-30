"""Extended Data Fig. 5 - the brain-model match is not produced by repeated words.

The narrative reuses its vocabulary, and a repeated word gets nearly the same
model representation each time, so part of the correspondence in Figure 4A could
come from repetition alone. Keeping one randomly chosen occurrence of each word
form removes that possibility; repeating the draw twenty times shows the answer
does not depend on which occurrence was picked.

One panel, the full lag range. The zoom around word onset was dropped: it shows
the same two curves over a sub-range of the same axis and adds nothing the full
range does not already carry.

Colours are set here rather than taken from lib.style.roi_color on purpose: the
published caption names the curves "blue" and "orange", so changing them would
silently falsify the caption. Everything else - fonts, layout, panel letters,
output path - goes through lib.style as usual.

Loads:  results/brain_llm_rsa_dedup/<roi>_<model>.npz
Output: plots/figS_brain_llm_dedup.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

C_FULL, C_DEDUP = "#0072B2", "#D55E00"      # blue, orange - see the docstring


def _draw(ax, lags, full, curves, xlim, title, *, legend=False):
    keep = (lags >= xlim[0]) & (lags <= xlim[1])
    x = lags[keep]
    mu = np.nanmean(curves, axis=0)[keep]
    sd = np.nanstd(curves, axis=0, ddof=1)[keep]

    style.add_word_onset(ax)
    ax.fill_between(x, mu - sd, mu + sd, color=C_DEDUP, alpha=0.25, linewidth=0)
    ax.plot(x, np.asarray(full)[keep], color=C_FULL, linewidth=2.0,
            label="All words")
    ax.plot(x, mu, color=C_DEDUP, linewidth=2.0,
            label="One instance per form")

    lo = min(np.nanmin(mu - sd), np.nanmin(np.asarray(full)[keep]))
    hi = max(np.nanmax(mu + sd), np.nanmax(np.asarray(full)[keep]))
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel(r"Brain$\times$LLM  $\rho$")
    ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(loc="upper left", framealpha=0.94, fontsize=8)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True, with_model=True)
    args = parser.parse_args()

    d = load_result(f"brain_llm_rsa_dedup/{args.roi}_{args.model}")
    lags = np.asarray(d["eval_lags_ms"])
    full = np.asarray(d["rsa_full"])
    curves = np.asarray(d["curves"])

    fig, axes = style.fig_layout(1, 1, panel_size=(5.4, 3.6))
    # A single panel carries no A/B letters - there is nothing to distinguish.
    _draw(np.atleast_1d(axes).ravel()[0], lags, full, curves,
          (lags[0], lags[-1]), "", legend=True)

    fig.suptitle(
        f"{style.roi_display_name(args.roi)} — {int(d['n_types'])} distinct "
        f"forms out of {int(d['n_words'])} tokens; mean of "
        f"{int(d['n_seeds'])} random draws ± 1 s.d.", fontsize=10)

    out = style.save_svg(fig, "figS_brain_llm_dedup")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
