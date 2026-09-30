"""Extended Data Fig. 3 - the ERG dip survives noise-ceiling correction.

rho_ERG is bounded above by how reliably each mega-RDM is estimated, so the
word-triggered dip has to be checked against a ceiling measured on the same
footing. Every term below compares two disjoint halves of the ROI contacts,
so none of them gets the shared-electrode boost that inflates the ordinary
ERG, and the ratio is interpretable:

  A  rel(t)    = rho( RDM_A(t), RDM_B(t)   )   the ceiling at lag t
  B  erg(t)    = rho( RDM_A(t), RDM_B(t+D) )   ERG across disjoint halves
  C  disatt(t) = erg(t) / sqrt( rel(t) * rel(t+D) )

Panel C is the result: the post-onset dip is still there once measurement
reliability has been divided out.

Curves are the mean over the analysis's random half-splits, with a +/-1 SEM
band across those splits.

Loads:  results/cross_half_erg/<roi>.npz
Output: plots/figS_cross_half_erg.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

BASELINE_WINDOW = (-500, -100)
POST_WINDOW = (0, 500)


def _draw(ax, lags, cube, color, ylabel, title, mark_min=False) -> None:
    """Mean across splits with a ±1 SEM band."""
    cube = np.atleast_2d(np.asarray(cube))
    mu = np.nanmean(cube, axis=0)
    n = cube.shape[0]
    sem = (np.nanstd(cube, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else None

    style.add_word_onset(ax)
    if sem is not None:
        ax.fill_between(lags, mu - sem, mu + sem, color=color, alpha=0.22,
                        linewidth=0)
    ax.plot(lags, mu, color=color, linewidth=2.0)

    if mark_min:
        post = (lags >= POST_WINDOW[0]) & (lags <= POST_WINDOW[1])
        k = int(np.nanargmin(mu[post]))
        ax.plot(lags[post][k], mu[post][k], "o", color=color, markersize=5,
                markerfacecolor="white", markeredgewidth=1.6, zorder=5)
        ax.annotate(f"{int(lags[post][k])} ms",
                    xy=(lags[post][k], mu[post][k]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=color)

    lo = np.nanmin(mu - (sem if sem is not None else 0.0))
    hi = np.nanmax(mu + (sem if sem is not None else 0.0))
    pad = 0.06 * (hi - lo) if hi > lo else 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(lags[0], lags[-1])
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    parser.add_argument("--all-panels", action="store_true",
                        help="Also draw the ceiling and the raw cross-half "
                             "ERG. The published figure is the disattenuated "
                             "curve alone, which is the default.")
    args = parser.parse_args()

    d = load_result(f"cross_half_erg/{args.roi}")
    lags = np.asarray(d["eval_lags_ms"])
    delta = int(d["delta_ms"])
    n_splits = int(d["n_splits"])
    color = style.roi_color(args.roi)

    if not args.all_panels:
        fig, axes = style.fig_layout(1, 1, panel_size=(5.2, 3.6))
        _draw(axes[0], lags, d["erg_disattenuated"], color,
              "ERG / √(rel·rel)",
              "Disattenuated ERG — the dip survives", mark_min=True)
    else:
        fig, axes = style.fig_layout(1, 3, panel_size=(4.2, 3.3))
        _draw(axes[0], lags, d["reliability"], color,
              f"ρ(RDM_A(t), RDM_B(t))", "Cross-half noise ceiling")
        _draw(axes[1], lags, d["erg_cross"], color,
              f"ρ(RDM_A(t), RDM_B(t+{delta}))", "Cross-half ERG")
        _draw(axes[2], lags, d["erg_disattenuated"], color,
              "ERG / √(rel·rel)",
              "Disattenuated ERG — the dip survives", mark_min=True)
        for ax, letter in zip(axes, "ABC"):
            style.panel_label(ax, letter)

    fig.suptitle(
        f"{style.roi_display_name(args.roi)} — {int(d['n_electrodes'])} "
        f"contacts, mean of {n_splits} random half-splits ± 1 SEM",
        fontsize=10)

    out = style.save_svg(fig, "figS_cross_half_erg")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
