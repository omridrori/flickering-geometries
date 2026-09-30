"""Per-word GPT-2-XL surprisal from the transcript shipped with the dataset.

Single loader shared by every analysis that ranks or bins words by surprisal
(analyses/surprisal_split.py, analyses/surprisal_by_segment.py), so the
definition - sum over a word's sub-tokens of −log2 p(token | preceding
context), from the 'true_prob' column of stimuli/gpt2-xl/transcript.tsv -
lives in exactly one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import GPT2XL_TRANSCRIPT_TSV


def word_surprisal(n_words: int | None = None) -> np.ndarray:
    """Surprisal in bits, indexed by word_idx (length n_words, or all words)."""
    df = pd.read_csv(GPT2XL_TRANSCRIPT_TSV, sep="\t")
    df["surprisal"] = -np.log2(df["true_prob"].astype(float).clip(lower=1e-12))
    df_w = df.groupby("word_idx")["surprisal"].sum().sort_index()
    if n_words is not None:
        df_w = df_w[df_w.index < n_words]
    out = np.full(int(df_w.index.max()) + 1, np.nan)
    out[df_w.index.to_numpy(dtype=np.int64)] = df_w.to_numpy(dtype=float)
    return out
