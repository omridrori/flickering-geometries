"""Brain x RWKV RSA over the full lag range - the recurrent model for Extended Data Fig. 2.

Extended Data Fig. 2 compares the brain-model match across model architectures. Llama-3
and Mistral are both transformers; RWKV is a recurrent network, so it tests
whether the word-locked peak depends on the architecture that produced the
embeddings or only on the text the model read.

Every parameter is taken from analyses/brain_llm_rsa.py, which produced the
Llama-3 and Mistral curves in that figure. If any of these drift the third
curve stops being comparable to the other two:

    lags            -4000 .. +4000 ms in 25 ms steps
    brain window    +-100 ms half-window, 25 ms cache
    winsorize       cfg.shared.winsorize_sd
    baseline        none (wide-range RSA)
    LLM RDM         llm.cosine_rdm - column-centre, row L2-normalise, cosine
    lag smoothing   +-50 ms moving average along the lag axis

Only the all-electrode curve is computed. The low-audio and circular-shift
controls in brain_llm_rsa.py belong to Figure 4A, not to this figure.

RWKV is not in cfg["models"]: its embeddings come from their own script,
preprocessing/generate_rwkv_embeddings.py, and only layer 23 is used. The path
is therefore given here explicitly instead of going through
llm.embeddings_path().

CLI:  --roi {lang,aud}   (default lang)
Output: results/brain_llm_rsa_rwkv/<roi>.npz
Used by: figures/figS_brain_llm_rsa_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm import tqdm

from mind_in_context.lib import llm
from mind_in_context.lib.data import BaselineExtractor
from mind_in_context.lib.io import (EMBEDDINGS_DIR, load_config, make_argparser,
                                    save_result)
from mind_in_context.lib.rdm import lag_window_average, spearman

LAG_START_MS = -4000
LAG_END_MS = 4000
LAG_STEP_MS = 25
HALF_WINDOW_MS = 100
SMOOTH_HALF_WINDOW_MS = 50
RWKV_LAYER = 23

EMB_PATH = EMBEDDINGS_DIR / "rwkv" / f"rwkv_L{RWKV_LAYER}_ctx_full_emb.npy"


def main() -> None:
    parser = make_argparser(__doc__, with_roi=True)
    args = parser.parse_args()

    s = load_config()["shared"]
    eval_lags = np.arange(LAG_START_MS, LAG_END_MS + 1, LAG_STEP_MS, dtype=int)
    print(f"  ROI {args.roi}: lags [{LAG_START_MS}, {LAG_END_MS}] ms, "
          f"half_window={HALF_WINDOW_MS} ms")

    ext = BaselineExtractor(
        roi_key=args.roi,
        eval_lags_ms=eval_lags,
        half_window_ms=HALF_WINDOW_MS,
        winsorize_sd=s["winsorize_sd"],
        baseline_window_ms=None,
    )
    print(f"  electrodes: {ext.n_electrodes}   words: {ext.n_words}")

    if not EMB_PATH.exists():
        raise SystemExit(f"  RWKV embeddings not found: {EMB_PATH}")
    emb = np.load(EMB_PATH).astype(np.float64)
    n_words = min(emb.shape[0], ext.n_words)
    emb = emb[:n_words]
    print(f"  RWKV layer {RWKV_LAYER} full-context embeddings: {emb.shape}")

    print("  building RWKV RDM (centre, L2-normalise, cosine) ...")
    llm_rdm = llm.cosine_rdm(emb)

    raw = np.full(len(eval_lags), np.nan)
    for t, lag in enumerate(tqdm(eval_lags, desc="    RSA")):
        raw[t] = spearman(ext.rdm(int(lag)), llm_rdm)

    rsa = lag_window_average(raw, max(1, SMOOTH_HALF_WINDOW_MS // LAG_STEP_MS))

    pk = int(np.nanargmax(rsa))
    pre = (eval_lags >= -1000) & (eval_lags <= -200)
    print(f"\n  peak rho = {rsa[pk]:.4f} at {eval_lags[pk]} ms")
    print(f"  pre-onset baseline mean rho = {np.nanmean(rsa[pre]):.4f}")

    out = save_result(
        f"brain_llm_rsa_rwkv/{args.roi}",
        eval_lags_ms=eval_lags,
        rsa_all=rsa,
        rsa_all_raw=raw,
        peak_lag_ms=np.int64(eval_lags[pk]),
        peak_rho=np.float64(rsa[pk]),
        n_words=np.int64(n_words),
        n_electrodes=np.int64(ext.n_electrodes),
        layer=np.int64(RWKV_LAYER),
        half_window_ms=np.int64(HALF_WINDOW_MS),
        smooth_half_window_ms=np.int64(SMOOTH_HALF_WINDOW_MS),
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
