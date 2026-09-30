"""Build the word-locked high-gamma caches from the Podcast dataset (ds005574).

Every analysis reads brain activity from one of two caches, each an array of
shape (n_lags, n_words, n_electrodes), float32:

  25 ms cache   lags -5000 .. +5000 ms in 25 ms steps   (401 lags, ~10 GB)
                used by the ERG, the lag x lag matrices and the wide-range
                brain-LLM comparison
  1 ms cache    lags -1000 .. +999 ms in 1 ms steps     (2000 lags, ~52 GB)
                used where the position of a peak is estimated (context
                peak-lag analyses)

For each word, a window around its onset is cut from the dataset's high-gamma
derivative (70-200 Hz envelope) of every electrode, z-scored with that
electrode's mean and standard deviation over the whole recording, and resampled
to the cache's lag grid. Windows that run past the ends of the recording are
zero-padded. No further filtering is applied.

The last axis holds all electrodes of all subjects: subjects in sorted order
and, within a subject, the channel order of its high-gamma file. That order is
written to the companion params file and is what the ROI selection relies on.

Input (see README for the download):
  <BIDS_ROOT>/stimuli/en_core_web_lg/transcript.tsv     word onsets
  <BIDS_ROOT>/derivatives/ecogprep/sub-XX/ieeg/         high-gamma .fif files

Output:
  <data dir>/cache/brain_cache_<resolution>.npy
  <data dir>/cache/brain_cache_<resolution>_params.json

Usage:
  python mind_in_context/preprocessing/generate_brain_cache.py --resolution 25ms
  python mind_in_context/preprocessing/generate_brain_cache.py --resolution 1ms
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from numpy.lib.format import open_memmap
from scipy.signal import resample
from tqdm import tqdm

from mind_in_context.lib.io import (BIDS_ROOT, BRAIN_CACHE_DIR, DERIVATIVES_DIR,
                                    STIMULI_DIR)

LAG_GRIDS = {
    "25ms": np.arange(-5000, 5000 + 25, 25),
    "1ms": np.arange(-1000, 1000, 1),
}
NORMALIZATION = "global_zscore"   # z-score with whole-recording statistics
FILTER = "none"                   # the high-gamma derivative is used as is
TRANSCRIPT_SUBDIR = "en_core_web_lg"


def load_transcript() -> pd.DataFrame:
    """One row per word: word_idx, word, start (onset in seconds)."""
    path = STIMULI_DIR / TRANSCRIPT_SUBDIR / "transcript.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Transcript not found at {path}.\n"
            f"Download ds005574 and point BIDS_ROOT at it (currently {BIDS_ROOT}).")
    df = pd.read_csv(path, sep="\t")
    words = (df.groupby("word_idx")
               .agg(word=("word", "first"), start=("start", "first"))
               .reset_index())
    print(f"   > transcript: {len(words)} words")
    return words


def load_all_subjects() -> tuple[dict, list]:
    """Load the high-gamma recording of every subject.

    Returns {subject id -> mne.io.Raw} and a list of per-subject dicts
    (id, n_elec, ch_names, sfreq), in sorted subject order.
    """
    subject_dirs = sorted(d for d in DERIVATIVES_DIR.iterdir()
                          if d.is_dir() and d.name.startswith("sub-"))
    if not subject_dirs:
        raise RuntimeError(f"No sub-XX folders under {DERIVATIVES_DIR}.")

    brain_data: dict = {}
    subject_info: list = []
    for sub_dir in tqdm(subject_dirs, desc="Loading subjects"):
        subject_id = sub_dir.name.replace("sub-", "")
        bids_path = BIDSPath(root=DERIVATIVES_DIR, subject=subject_id,
                             task="podcast", description="highgamma",
                             suffix="ieeg", extension=".fif", datatype="ieeg",
                             check=False)
        matches = bids_path.match()
        if not matches:
            print(f"   ! no high-gamma file for sub-{subject_id}, skipping")
            continue
        raw = mne.io.read_raw_fif(matches[0], preload=True, verbose="ERROR")
        brain_data[subject_id] = raw
        subject_info.append({"id": subject_id, "n_elec": len(raw.ch_names),
                             "ch_names": list(raw.ch_names),
                             "sfreq": raw.info["sfreq"]})
    print(f"   > loaded {len(brain_data)} subjects")
    return brain_data, subject_info


def compute_global_stats(brain_data: dict) -> dict:
    """Per-electrode mean and std over the full recording, per subject."""
    stats = {}
    for sub_id, raw in tqdm(brain_data.items(), desc="Global statistics"):
        data = raw.get_data()                       # (n_elec, n_times)
        stats[sub_id] = {"mean": np.mean(data, axis=1, keepdims=True),
                         "std": np.std(data, axis=1, keepdims=True)}
    return stats


def build_cache(words: pd.DataFrame, brain_data: dict, subject_info: list,
                global_stats: dict, lags_ms: np.ndarray, out: np.ndarray) -> None:
    """Fill `out` (n_lags, n_words, n_electrodes) one word at a time."""
    n_lags = len(lags_ms)
    for word_idx, row in tqdm(words.iterrows(), total=len(words), desc="Words"):
        onset_s = row["start"]
        windows = []
        for sub in subject_info:
            raw = brain_data[sub["id"]]
            sfreq = sub["sfreq"]

            # Window boundaries in samples of this subject's recording.
            win_start = int(round((onset_s + lags_ms[0] / 1000.0) * sfreq))
            win_end = int(round((onset_s + lags_ms[-1] / 1000.0) * sfreq)) + 1
            window = np.zeros((sub["n_elec"], win_end - win_start), dtype=np.float64)

            # Copy the part that lies inside the recording; the rest stays zero.
            valid_start = max(0, win_start)
            valid_end = min(raw.n_times, win_end)
            if valid_start < valid_end:
                segment = raw.get_data(start=valid_start, stop=valid_end)
                at = valid_start - win_start
                window[:, at:at + segment.shape[1]] = segment

            stats = global_stats[sub["id"]]
            window = (window - stats["mean"]) / (stats["std"] + 1e-8)
            windows.append(resample(window, num=n_lags, axis=-1))

        full = np.concatenate(windows, axis=0)      # (n_electrodes, n_lags)
        out[:, word_idx, :] = full.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", choices=list(LAG_GRIDS), default="25ms")
    args = parser.parse_args()

    lags_ms = LAG_GRIDS[args.resolution]
    npy_path = BRAIN_CACHE_DIR / f"brain_cache_{args.resolution}.npy"
    json_path = BRAIN_CACHE_DIR / f"brain_cache_{args.resolution}_params.json"
    BRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"BIDS root : {BIDS_ROOT}")
    print(f"Output    : {npy_path}")
    print(f"Lags      : {lags_ms[0]} .. {lags_ms[-1]} ms ({len(lags_ms)} points)")

    words = load_transcript()
    brain_data, subject_info = load_all_subjects()
    if not brain_data:
        raise RuntimeError("No brain data was loaded.")
    global_stats = compute_global_stats(brain_data)

    n_lags, n_words = len(lags_ms), len(words)
    n_elec = sum(s["n_elec"] for s in subject_info)
    print(f"Shape     : ({n_lags}, {n_words}, {n_elec}), "
          f"{n_lags * n_words * n_elec * 4 / 1e9:.1f} GB")

    # Written straight to disk, one word at a time; never held in RAM.
    out = open_memmap(npy_path, mode="w+", dtype=np.float32,
                      shape=(n_lags, n_words, n_elec))
    build_cache(words, brain_data, subject_info, global_stats, lags_ms, out)
    out.flush()
    del out

    params = {
        "lags_ms": lags_ms.tolist(),
        "normalization": NORMALIZATION,
        "filter": FILTER,
        "shape": [n_lags, n_words, n_elec],
        "shape_description": ["n_lags", "n_words", "n_electrodes"],
        "subjects": [s["id"] for s in subject_info],
        "n_electrodes_per_subject": {s["id"]: s["n_elec"] for s in subject_info},
        "electrodes": [{"subject_id": s["id"], "electrode_name": ch}
                       for s in subject_info for ch in s["ch_names"]],
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=1)
    print(f"Saved {npy_path.name} and {json_path.name}")


if __name__ == "__main__":
    main()
