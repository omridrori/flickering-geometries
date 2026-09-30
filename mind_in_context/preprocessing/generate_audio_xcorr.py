"""Cross-correlate every electrode with the speech envelope.

For each electrode, the continuous high-gamma time series is cross-correlated
with the envelope of the podcast audio over lags of -1500 .. +1500 ms, and the
peak correlation and its lag are recorded. The brain-LLM audio control uses the
peak correlation to pick the least audio-responsive contacts of a region.

Audio envelope: band-pass 200-5000 Hz (Butterworth, order 5), resampled to the
recording's sampling rate, then the magnitude of the Hilbert transform.

Input (see README for the download):
  <BIDS_ROOT>/stimuli/podcast.wav
  <BIDS_ROOT>/derivatives/ecogprep/sub-XX/ieeg/   high-gamma .fif files

Output:
  <data dir>/audio_xcorr.json    one record per electrode, sorted by peak
                                 correlation: subject_id, electrode_name,
                                 peak_correlation, peak_lag_ms,
                                 peak_lag_samples, sfreq_hz

Usage:
  python mind_in_context/preprocessing/generate_audio_xcorr.py
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
from scipy import signal
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm

from mind_in_context.lib.io import AUDIO_XCORR_JSON, DERIVATIVES_DIR, PODCAST_WAV

AUDIO_LOWCUT_HZ = 200
AUDIO_HIGHCUT_HZ = 5000
BUTTER_ORDER = 5
MAX_LAG_MS = 1500
NORMALIZE_CORR = True     # scale to [-1, 1], like a Pearson correlation
PEAK_MODE = "max"         # "max": largest correlation; "abs": largest magnitude


def discover_subjects() -> list[str]:
    return [d.name.replace("sub-", "") for d in sorted(DERIVATIVES_DIR.iterdir())
            if d.is_dir() and d.name.startswith("sub-")]


def audio_envelope(wave: np.ndarray, fs: int, to_fs: float) -> np.ndarray:
    """Band-pass -> resample to the recording's rate -> Hilbert envelope."""
    nyq = 0.5 * fs
    b, a = signal.butter(BUTTER_ORDER, [AUDIO_LOWCUT_HZ / nyq, AUDIO_HIGHCUT_HZ / nyq],
                         btype="band")
    y = signal.lfilter(b, a, wave)
    y = signal.resample(y, num=int(round(wave.size / fs * to_fs)))
    return np.abs(signal.hilbert(y - y.mean())).astype(np.float32, copy=False)


def xcorr_1d(x: np.ndarray, y: np.ndarray, maxlags: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation of two centred signals, restricted to +-maxlags samples."""
    x0 = x - x.mean()
    y0 = y - y.mean()
    corr = correlate(x0, y0, mode="full", method="fft")
    lags = correlation_lags(x0.size, y0.size, mode="full")
    if NORMALIZE_CORR:
        corr = corr / (np.sqrt(np.dot(x0, x0) * np.dot(y0, y0)) + 1e-12)
    mid = int(np.where(lags == 0)[0].item())
    sl = slice(mid - maxlags, mid + maxlags + 1)
    return corr[sl].astype(np.float32, copy=False), lags[sl].astype(np.int32, copy=False)


def pick_peak(corr: np.ndarray, lags: np.ndarray) -> tuple[float, int]:
    score = corr if PEAK_MODE == "max" else np.abs(corr)
    idx = int(np.argmax(score))
    return float(corr[idx]), int(lags[idx])


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    if not PODCAST_WAV.exists():
        raise FileNotFoundError(
            f"{PODCAST_WAV} not found. Download ds005574 first (see README).")
    subjects = discover_subjects()
    if not subjects:
        raise RuntimeError(f"No sub-XX folders under {DERIVATIVES_DIR}.")

    audio_fs, wave = wavfile.read(str(PODCAST_WAV))
    if wave.ndim > 1:
        wave = wave[:, 0]
    wave = np.asarray(wave, dtype=np.float32)

    records = []
    for sid in tqdm(subjects, desc="Subjects"):
        bids_path = BIDSPath(root=DERIVATIVES_DIR, subject=sid, task="podcast",
                             description="highgamma", suffix="ieeg",
                             extension=".fif", datatype="ieeg", check=False)
        matches = bids_path.match()
        if not matches:
            print(f"   ! no high-gamma file for sub-{sid}, skipping")
            continue
        raw = mne.io.read_raw_fif(matches[0], preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        maxlags = int(round(MAX_LAG_MS / 1000.0 * sfreq))

        env = audio_envelope(wave, int(audio_fs), sfreq)
        n_times = int(min(env.shape[0], raw.n_times))   # trim to the shorter
        env = env[:n_times]

        for ch_idx, ch_name in enumerate(tqdm(raw.ch_names, desc=f"sub-{sid}",
                                              leave=False)):
            x = (raw.get_data(picks=[ch_idx], start=0, stop=n_times)
                    .reshape(-1).astype(np.float32, copy=False))
            corr, lags = xcorr_1d(x, env, maxlags)
            peak_corr, peak_lag = pick_peak(corr, lags)
            records.append({
                "subject_id": str(sid),
                "electrode_name": str(ch_name),
                "peak_correlation": float(peak_corr),
                "peak_lag_ms": float(peak_lag / sfreq * 1000.0),
                "peak_lag_samples": int(peak_lag),
                "sfreq_hz": sfreq,
            })

    if not records:
        raise RuntimeError("No electrodes were processed.")
    df = (pd.DataFrame(records)
            .sort_values(by=["peak_correlation"], ascending=False)
            .reset_index(drop=True))
    AUDIO_XCORR_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIO_XCORR_JSON.write_text(json.dumps(df.to_dict(orient="records"), indent=1),
                                encoding="utf-8")
    print(f"{len(df)} electrodes -> {AUDIO_XCORR_JSON}")


if __name__ == "__main__":
    main()
