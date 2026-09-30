"""Extended Data Fig. 2 - the brain-model match across three model architectures.

Llama-3 and Mistral are transformers; RWKV is a recurrent network. All three
read the same narrative and are compared to the cortex the same way, so the
figure separates what depends on the text a model read from what depends on how
the model is built.

The published two-model version annotated each peak on the curve with a leader
line. That does not survive a third curve: RWKV peaks at rho = 0.088 and
Llama-3 at 0.086, so the two labels would sit on top of each other. Peak lag and
height moved into the legend, and each peak is marked with a dot instead.

Colours are inherited from the two-model figure so the pair reads as one family:
Llama-3 #e63946, Mistral #0072b2. RWKV takes Wong's bluish green, which stays
distinct from both in greyscale and under red-green colour blindness. They are
set here rather than through lib.style.roi_color because these identify MODELS,
not ROIs.

Loads:
  results/brain_llm_rsa/<roi>_llama3.npz
  results/brain_llm_rsa/<roi>_mistral7b.npz
  results/brain_llm_rsa_rwkv/<roi>.npz

Reproduce the inputs with:
  python -m mind_in_context.analyses.brain_llm_rsa --roi lang --model llama3
  python -m mind_in_context.analyses.brain_llm_rsa --roi lang --model mistral7b
  python -m mind_in_context.analyses.brain_llm_rsa_rwkv --roi lang

CLI:  --roi {lang,aud}
Output: plots/figS_brain_llm_rsa_models.svg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mind_in_context.lib import style
from mind_in_context.lib.io import load_result, make_argparser

# Matches the two-model figure it replaces, so the page layout does not move.
FIG_W_IN, FIG_H_IN = 8.96, 4.76

C_LLAMA, C_MISTRAL, C_RWKV = "#e63946", "#0072b2", "#009E73"


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    args = parser.parse_args()

    series = [
        ("Llama-3", load_result(f"brain_llm_rsa/{args.roi}_llama3"), C_LLAMA),
        ("Mistral-7B", load_result(f"brain_llm_rsa/{args.roi}_mistral7b"), C_MISTRAL),
        ("RWKV", load_result(f"brain_llm_rsa_rwkv/{args.roi}"), C_RWKV),
    ]

    fonts = style.figure_fonts_for_width(FIG_W_IN)
    fig, axes = style.fig_layout(1, 1, panel_size=(FIG_W_IN, FIG_H_IN),
                                 fonts=fonts)
    ax = axes[0]
    style.add_word_onset(ax)
    style.add_zero_line(ax)

    rows = []
    for name, d, colour in series:
        lags, y = np.asarray(d["eval_lags_ms"]), np.asarray(d["rsa_all"])
        k = int(np.nanargmax(y))
        pre = (lags >= -1000) & (lags <= -200)
        rows.append((name, lags[k], y[k], float(np.nanmean(y[pre]))))
        ax.plot(lags, y, color=colour, linewidth=2.0, solid_capstyle="round",
                zorder=3,
                label=f"{name} — {lags[k]} ms, ρ = {y[k]:.3f}")
        ax.scatter([lags[k]], [y[k]], s=34, color=colour, edgecolor="white",
                   linewidth=0.9, zorder=4)

    ax.set_xlim(-4400, 4400)
    ax.set_xlabel("Lag from word onset (ms)")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.legend(loc="upper left", frameon=False,
              fontsize=fonts["legend.fontsize"] * 0.9, handlelength=1.6)

    print(f"  {'model':12}{'peak lag':>10}{'peak rho':>10}{'pre-onset':>11}")
    for name, lag, rho, base in rows:
        print(f"  {name:12}{lag:>7d} ms{rho:>10.4f}{base:>11.4f}")

    # Curve-to-curve agreement, quoted in the caption.
    print("\n  pairwise correlation across the whole lag range:")
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            a = np.asarray(series[i][1]["rsa_all"])
            b = np.asarray(series[j][1]["rsa_all"])
            m = ~(np.isnan(a) | np.isnan(b))
            print(f"    {series[i][0]:12} vs {series[j][0]:12} "
                  f"r = {np.corrcoef(a[m], b[m])[0, 1]:.4f}")

    out = style.save_svg(fig, "figS_brain_llm_rsa_models")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
