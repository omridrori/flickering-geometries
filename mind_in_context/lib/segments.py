"""Word-segment helpers - equal word count vs equal time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import TRANSCRIPT_TSV


def _load_word_starts(n_words: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (word_idx_sorted_by_start, start_seconds) for words < n_words."""
    df = pd.read_csv(TRANSCRIPT_TSV, sep="\t")
    df["word_idx"] = df["word_idx"].astype(int)
    df_w = df.groupby("word_idx").agg(start=("start", "min")).reset_index()
    df_w = df_w[df_w["word_idx"] < n_words].sort_values("start").reset_index(drop=True)
    return df_w["word_idx"].to_numpy(dtype=np.int64), df_w["start"].to_numpy()


def eqword_segments(n_words: int, n_segments: int
                    ) -> tuple[list[np.ndarray], np.ndarray]:
    """Split the podcast into n_segments chronological slices of equal word count.

    Returns:
        groups: list[np.ndarray]   each = word_idx values in that segment
        time_edges: (n_segments, 2) array of [start_sec, end_sec] per segment
    """
    word_idx_sorted, start_sorted = _load_word_starts(n_words)
    n_total = len(word_idx_sorted)
    edges = np.linspace(0, n_total, n_segments + 1).astype(int)
    groups = [word_idx_sorted[edges[k]:edges[k + 1]] for k in range(n_segments)]
    time_edges = np.array([
        (float(start_sorted[edges[k]]), float(start_sorted[edges[k + 1] - 1]))
        for k in range(n_segments)
    ])
    return groups, time_edges


def eqtime_segments(n_words: int, n_segments: int
                    ) -> tuple[list[np.ndarray], np.ndarray]:
    """Split the podcast into n_segments equal-duration time slices.

    Returns:
        groups: list[np.ndarray]   each = word_idx values whose start lies in
                that time slice
        time_edges: (n_segments, 2) array of [start_sec, end_sec] per segment
    """
    word_idx_sorted, start_sorted = _load_word_starts(n_words)
    edges = np.linspace(start_sorted.min(), start_sorted.max(), n_segments + 1)
    seg_id = np.digitize(start_sorted, edges[1:-1], right=False)
    # word_idx_sorted is sorted by start, so seg_id values are monotone
    groups = [word_idx_sorted[seg_id == k] for k in range(n_segments)]
    time_edges = np.array([(float(edges[k]), float(edges[k + 1]))
                           for k in range(n_segments)])
    return groups, time_edges


def full_time_span(n_words: int) -> tuple[float, float]:
    """First / last word-start (seconds) for the first n_words words."""
    _, start_sorted = _load_word_starts(n_words)
    return (float(start_sorted.min()), float(start_sorted.max()))
