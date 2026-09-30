"""Extended Data Fig. 1 - no single patient drives the reported latencies.

Coverage is very uneven, so the whole pipeline was rerun once per patient with
that patient's contacts withdrawn. Each panel shows the mean over those folds
with a +/-1 SEM band across them; individual folds are not drawn, since which
patient a given curve belongs to belongs in the text rather than in the figure.

  A  ERG curve                the dip does not move
  B  brain x LLM RSA curve    the peak does not move
  C  peak lag vs context      the increase holds in every fold

The full-ROI fold is stored in the same file but is deliberately not plotted:
every fold is a subset of it, so it sits systematically outside the band and
reads as a discrepancy when it is only a difference in electrode count.

Loads:  results/leave_one_patient_out/<roi>.npz
Output: plots/figS_lopo.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

ERG_XLIM = (-100, 600)
RSA_XLIM = (-750, 1500)


def _band(ax, x, cube, color):
    """Mean across folds with a ±1 SEM band. Returns (mean, sem)."""
    cube = np.atleast_2d(np.asarray(cube, dtype=float))
    mu = np.nanmean(cube, axis=0)
    n = cube.shape[0]
    sem = (np.nanstd(cube, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else None
    if sem is not None:
        ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=0.22,
                        linewidth=0)
    ax.plot(x, mu, color=color, linewidth=2.0)
    return mu, sem


def _lag_panel(ax, lags, cube, full, xlim, color, ylabel, title, mode):
    """The guide line and its label are read off the fold mean, never passed in.

    They used to be hard-coded (175 ms, 350 ms), which was correct only for the
    ERG gap in use when the figure was first drawn; changing cfg.shared.
    erg_delta_ms moved the dip to 150 ms and left the annotation contradicting
    the marker beside it. Deriving both from the data keeps them in step.
    """
    keep = (lags >= xlim[0]) & (lags <= xlim[1])
    style.add_word_onset(ax)
    mu, sem = _band(ax, lags[keep], cube[:, keep], color)
    ax.plot(lags[keep], np.asarray(full, dtype=float)[keep], color="black",
            linewidth=1.4, zorder=4)

    k = int(np.nanargmin(mu) if mode == "min" else np.nanargmax(mu))
    ref_lag = int(lags[keep][k])

    ax.axvline(ref_lag, color=style.ZERO_LINE_COLOR, linewidth=0.9,
               linestyle=(0, (3, 3)), zorder=0)
    lo = min(np.nanmin(mu - (sem if sem is not None else 0.0)),
             np.nanmin(np.asarray(full, dtype=float)[keep]))
    hi = max(np.nanmax(mu + (sem if sem is not None else 0.0)),
             np.nanmax(np.asarray(full, dtype=float)[keep]))
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.annotate(f"{ref_lag} ms", xy=(ref_lag, hi + pad), xytext=(5, -3),
                textcoords="offset points", ha="left", va="top", fontsize=8,
                color="#333333")

    ax.plot(ref_lag, mu[k], "o", color=color, markersize=5,
            markerfacecolor="white", markeredgewidth=1.6, zorder=5)

    ax.set_xlim(xlim)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    args = parser.parse_args()

    d = load_result(f"leave_one_patient_out/{args.roi}")
    names = [str(x) for x in d["fold_names"]]
    loo = np.array([i for i, n in enumerate(names) if n != "all"])
    full = names.index("all")
    n_fold = len(loo)
    color = style.roi_color(args.roi)

    fig, axes = style.fig_layout(1, 3, panel_size=(4.2, 3.3))

    _lag_panel(axes[0], np.asarray(d["erg_lags_ms"]),
               np.asarray(d["erg_curves"])[loo],
               np.asarray(d["erg_curves"])[full], ERG_XLIM, color,
               r"$\rho_{\rm ERG}$", "Event-related geometry", "min")
    # Short labels: the analysis is already named in the suptitle, and a wide
    # legend box has nowhere to sit in this panel without covering a curve.
    # Bottom-right is the only region both curves stay clear of, and the
    # legend is lifted above the word-onset rule so nothing is drawn over it.
    leg = axes[0].legend(handles=[
        Line2D([], [], color="black", lw=1.4, label="Full ROI"),
        Line2D([], [], color=color, lw=2.0, label=f"Fold mean (n = {n_fold})"),
        Patch(facecolor=color, edgecolor="none", alpha=0.22, label="± 1 SEM"),
    ], loc="lower right", fontsize=7.5, framealpha=1.0, borderaxespad=0.4,
        handlelength=1.5, labelspacing=0.35)
    leg.set_zorder(20)
    leg.get_frame().set_edgecolor("#CCCCCC")

    _lag_panel(axes[1], np.asarray(d["rsa_lags_ms"]),
               np.asarray(d["rsa_curves"])[loo],
               np.asarray(d["rsa_curves"])[full], RSA_XLIM, color,
               r"Brain$\times$LLM  $\rho$", "Brain–model correspondence",
               "max")

    ax = axes[2]
    ctx = np.asarray(d["context_words"], dtype=float)
    peaks = np.asarray(d["ctx_peaks"], dtype=float)
    mu, sem = _band(ax, ctx, peaks[loo], color)
    ax.plot(ctx, peaks[full], color="black", linewidth=1.4, zorder=4)
    lo = min(np.nanmin(mu - (sem if sem is not None else 0.0)),
             float(peaks[full].min()))
    hi = max(np.nanmax(mu + (sem if sem is not None else 0.0)),
             float(peaks[full].max()))
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([2, 5, 10, 15, 20])
    ax.set_xlabel("Context length (words)")
    ax.set_ylabel("Peak lag (ms)")
    ax.set_title("Peak lag against context length", fontsize=9)

    for ax, letter in zip(axes, "ABC"):
        style.panel_label(ax, letter)

    fig.suptitle(
        f"Leave-one-patient-out analysis — "
        f"{style.roi_display_name(args.roi)}, "
        f"{int(d['n_electrodes'][full])} contacts, {n_fold} folds",
        fontsize=10)

    out = style.save_svg(fig, "figS_lopo")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
