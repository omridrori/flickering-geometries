"""Config loading, path resolution, results save/load, standard argparse."""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths.
#
# Code, config, results and plots live inside the repository. Everything large
# (the dataset, the word-locked brain caches, the model embeddings) lives in a
# data directory that is NOT tracked:
#
#   <data dir>/                     default: <repo>/data     (env FG_DATA_DIR)
#     ds005574/                     the OpenNeuro dataset    (env BIDS_ROOT)
#     transcript.tsv                preprocessing/make_transcript.py
#     audio_xcorr.json              preprocessing/generate_audio_xcorr.py
#     cache/                        preprocessing/generate_brain_cache.py
#     embeddings/                   preprocessing/generate_*_embeddings*.py
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2]                                 # repo root
PKG_ROOT = _HERE.parents[1]                             # mind_in_context/

CONFIG_PATH = PKG_ROOT / "config.yaml"
CACHE_DIR = PKG_ROOT / "cache"
RESULTS_DIR = PKG_ROOT / "results"
PLOTS_DIR = PKG_ROOT / "plots"

DATA_DIR = Path(os.environ.get("FG_DATA_DIR", ROOT / "data"))
BIDS_ROOT = Path(os.environ.get("BIDS_ROOT", DATA_DIR / "ds005574"))

# Dataset files, used as distributed.
STIMULI_DIR = BIDS_ROOT / "stimuli"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "ecogprep"
PODCAST_WAV = STIMULI_DIR / "podcast.wav"
PODCAST_TRANSCRIPT_CSV = STIMULI_DIR / "podcast_transcript.csv"
# Token-level GPT-2-XL transcript with per-token surprisal (true_prob).
GPT2XL_TRANSCRIPT_TSV = STIMULI_DIR / "gpt2-xl" / "transcript.tsv"

# Word-locked high-gamma caches, shape (n_lags, n_words, n_electrodes).
# The 25 ms cache covers -5000..5000 ms; the 1 ms cache covers -1000..999 ms.
BRAIN_CACHE_DIR = DATA_DIR / "cache"
BRAIN_CACHE_NPY_25MS = BRAIN_CACHE_DIR / "brain_cache_25ms.npy"
BRAIN_CACHE_NPY_1MS = BRAIN_CACHE_DIR / "brain_cache_1ms.npy"
BRAIN_CACHE_JSON_25MS = BRAIN_CACHE_DIR / "brain_cache_25ms_params.json"
BRAIN_CACHE_JSON_1MS = BRAIN_CACHE_DIR / "brain_cache_1ms_params.json"
# Default = 25 ms (covers the standard analysis range). Its params file also
# records the electrode order of the cache's last axis.
BRAIN_CACHE_NPY = BRAIN_CACHE_NPY_25MS
BRAIN_CACHE_JSON = BRAIN_CACHE_JSON_25MS

# One row per word: word_idx, word, start, end.
TRANSCRIPT_TSV = DATA_DIR / "transcript.tsv"
# Per-electrode peak cross-correlation with the audio envelope, used to select
# the least audio-responsive contacts for the audio control.
AUDIO_XCORR_JSON = DATA_DIR / "audio_xcorr.json"
# Per-word model embeddings.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Atlas lookup (electrode -> Destrieux label), computed once and kept in the
# repository so that every run uses the same contact set.
ROI_INDICES_CACHE_JSON = PKG_ROOT / "cache" / "roi_electrode_indices.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load mind_in_context/config.yaml (cached)."""
    import yaml
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def primary_model() -> str:
    cfg = load_config()
    for key, spec in cfg["models"].items():
        if spec.get("primary"):
            return key
    raise ValueError("No primary model defined in config.yaml")


# ---------------------------------------------------------------------------
# Results / cache I/O
# ---------------------------------------------------------------------------
def save_result(name: str, **arrays: np.ndarray) -> Path:
    """Save arrays to results/<name>.npz, creating parent dirs."""
    path = RESULTS_DIR / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_result(name: str) -> np.lib.npyio.NpzFile:
    """Load results/<name>.npz."""
    return np.load(RESULTS_DIR / f"{name}.npz", allow_pickle=False)


def cache_path(kind: str, key: str) -> Path:
    """Return cache/<kind>/<key>.npz (creates parent on demand at write time)."""
    return CACHE_DIR / kind / f"{key}.npz"


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------
def make_argparser(description: str, *, with_roi: bool = False,
                   with_model: bool = False, with_plot_only: bool = False
                   ) -> argparse.ArgumentParser:
    """Standard argparse with common flags."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=description)
    if with_roi:
        parser.add_argument("--roi", choices=list(cfg["rois"].keys()),
                            required=True, help="ROI key from config.yaml")
    if with_model:
        parser.add_argument("--model", choices=list(cfg["models"].keys()),
                            default=primary_model(),
                            help="LLM key from config.yaml (default: primary)")
    if with_plot_only:
        parser.add_argument("--plot-only", action="store_true",
                            help="Skip computation; load .npz and replot only.")
    return parser


def read_brain_cache_params() -> dict[str, Any]:
    """Read the brain-cache metadata (lags, shape, subjects, electrode order)."""
    with open(BRAIN_CACHE_JSON) as fh:
        return json.load(fh)
