"""Brain cache loading, ROI selection, lag-window averaging, baseline correction.

Analyses use all contacts in each ROI (73 language / 135 auditory).

Most analyses should NOT call the low-level helpers below - use the high-level
`BaselineExtractor` class instead, which does load + ROI + baseline + slab
pre-load + windowed extraction + caching in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .constants import CACHE_ORIGIN_MS, CACHE_STEP_MS
from .io import (
    AUDIO_XCORR_JSON,
    BRAIN_CACHE_JSON,
    BRAIN_CACHE_JSON_1MS,
    BRAIN_CACHE_JSON_25MS,
    BRAIN_CACHE_NPY,
    BRAIN_CACHE_NPY_1MS,
    BRAIN_CACHE_NPY_25MS,
    DERIVATIVES_DIR,
    ROI_INDICES_CACHE_JSON,
    load_config,
)


# ---------------------------------------------------------------------------
# Brain cache loader.
# ---------------------------------------------------------------------------
def load_brain_cache(path: Path = BRAIN_CACHE_NPY,
                     json_path: Path | None = None) -> np.ndarray:
    """Load a brain cache as a read-only memory map.

    Accepts a standard .npy file, or a raw float32 memmap whose shape is given
    by a companion params JSON (`<name>_params.json` next to the file unless
    json_path is passed).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Brain cache not found: {path}\n"
            "Build it with preprocessing/generate_brain_cache.py (see README).")
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic == b"\x93NUMPY":
        return np.load(path, mmap_mode="r")
    if json_path is None:
        json_path = path.with_name(path.stem + "_params.json")
    with open(json_path) as fh:
        params = json.load(fh)
    return np.memmap(path, dtype=np.float32, mode="r", shape=tuple(params["shape"]))


def cache_meta(resolution: str) -> tuple[Path, int, int, Path]:
    """Return (npy_path, origin_ms, step_ms, params_json) for a cache resolution."""
    if resolution == "1ms":
        return (BRAIN_CACHE_NPY_1MS, -1000, 1, BRAIN_CACHE_JSON_1MS)
    if resolution == "25ms":
        return (BRAIN_CACHE_NPY_25MS, -5000, 25, BRAIN_CACHE_JSON_25MS)
    raise ValueError(f"Unknown cache resolution {resolution!r}")


def electrode_table() -> tuple[list[str], dict[str, list[str]]]:
    """Electrode order of the cache's last axis.

    Returns (subject order, {subject id -> channel names in cache order}), as
    recorded by preprocessing/generate_brain_cache.py: subjects in sorted
    order, and within each subject the channel order of its high-gamma file.
    """
    with open(BRAIN_CACHE_JSON) as fh:
        params = json.load(fh)
    subjects = [str(s) for s in params["subjects"]]
    per_subject: dict[str, list[str]] = {s: [] for s in subjects}
    for entry in params["electrodes"]:
        per_subject[str(entry["subject_id"])].append(str(entry["electrode_name"]))
    return subjects, per_subject


# ---------------------------------------------------------------------------
# ROI -> cache column indices (atlas-based).
# ---------------------------------------------------------------------------
def _normalize_roi_label(label: str) -> str:
    parts = str(label).split()
    if parts and parts[0] in {"L", "R"}:
        return " ".join(parts[1:])
    return str(label)


def _build_atlas_indices(atlas_label: str) -> np.ndarray:
    """Map every cache column to a Destrieux atlas label, return matches.

    Heavy nilearn/MNE query - runs once, then cached in
    cache/roi_electrode_indices.json keyed by atlas_label.
    """
    if ROI_INDICES_CACHE_JSON.exists():
        with open(ROI_INDICES_CACHE_JSON) as fh:
            cached = json.load(fh)
        if atlas_label in cached:
            return np.array(cached[atlas_label], dtype=np.int64)

    import mne
    from mne_bids import BIDSPath
    from nilearn import datasets, image
    from scipy.spatial import KDTree
    from tqdm import tqdm

    atlas = datasets.fetch_atlas_destrieux_2009()
    atlas_img = image.load_img(atlas["maps"])
    atlas_data = atlas_img.get_fdata().astype(int)
    atlas_label_map = {
        int(idx): str(name)
        for idx, name in zip(atlas["labels"].index, atlas["labels"].name)
    }
    nonzero = np.nonzero(atlas_data)
    atlas_coords_mm = np.vstack(image.coord_transform(*nonzero, atlas_img.affine)).T
    tree = KDTree(atlas_coords_mm)

    subject_order, elec_per_subject = electrode_table()

    matched: list[int] = []
    global_idx = 0
    for sid in tqdm(subject_order, desc=f"  ROI '{atlas_label}'"):
        bids_path = BIDSPath(
            root=DERIVATIVES_DIR, subject=sid, task="podcast",
            description="highgamma", suffix="ieeg", extension=".fif",
            datatype="ieeg", check=False,
        )
        matches = bids_path.match()
        if not matches:
            global_idx += len(elec_per_subject[sid])
            continue
        raw = mne.io.read_raw_fif(matches[0], preload=False, verbose="ERROR")
        montage = raw.get_montage()
        ch_pos_m = montage.get_positions()["ch_pos"] if montage else {}
        for ch_name in elec_per_subject[sid]:
            label_here = "Unknown"
            if ch_name in ch_pos_m:
                pos_mm = np.asarray(ch_pos_m[ch_name]) * 1000.0
                _, nn_idx = tree.query(pos_mm.reshape(1, 3))
                xi = nonzero[0][nn_idx[0]]
                yi = nonzero[1][nn_idx[0]]
                zi = nonzero[2][nn_idx[0]]
                label_here = _normalize_roi_label(
                    atlas_label_map.get(int(atlas_data[xi, yi, zi]), "Unknown")
                )
            if label_here == atlas_label:
                matched.append(global_idx)
            global_idx += 1

    indices = np.array(matched, dtype=np.int64)
    existing: dict = {}
    if ROI_INDICES_CACHE_JSON.exists():
        with open(ROI_INDICES_CACHE_JSON) as fh:
            existing = json.load(fh)
    existing[atlas_label] = indices.tolist()
    ROI_INDICES_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ROI_INDICES_CACHE_JSON, "w") as fh:
        json.dump(existing, fh, indent=2)
    return indices


def roi_indices(roi_key: str) -> np.ndarray:
    """Cache column indices for all electrodes in the named ROI ('lang' / 'aud')."""
    cfg = load_config()
    if roi_key not in cfg["rois"]:
        raise ValueError(f"Unknown ROI key '{roi_key}'. "
                         f"Known: {list(cfg['rois'].keys())}")
    return _build_atlas_indices(cfg["rois"][roi_key]["atlas_label"])


def _global_electrode_keys() -> list[tuple[str, str]]:
    """(subject_id, electrode_name) for every cache column, in cache order."""
    subjects, elec_per_subject = electrode_table()
    keys: list[tuple[str, str]] = []
    for sid in subjects:
        for ch in elec_per_subject[sid]:
            keys.append((sid, ch))
    return keys


def low_audio_electrode_indices(roi_key: str, fraction: float = 1 / 3
                                ) -> np.ndarray:
    """Bottom `fraction` of ROI electrodes by peak audio cross-correlation.

    Returns the cache column indices that audio-tracking is least likely to
    explain - used as a control in brain–LLM RSA. The values come from
    preprocessing/generate_audio_xcorr.py.

    ROI membership comes from the atlas via roi_indices(); the JSON is consulted
    only for the value. It also carries a roi_name of its own, which disagrees
    with the atlas on at least one contact (sub-05 GA19: Opercular there,
    Triangular in the atlas), so filtering on it silently shrank the ROI.

    Ranking is by the peak correlation itself, not its magnitude: the source
    values are a maximum over lags (PEAK_MODE "max"), so they are already
    signed the one way. For the language ROI only 3 of 73 are negative, none
    below -0.006, and ranking by |value| selects exactly the same contacts.
    """
    roi = sorted(roi_indices(roi_key).tolist())
    key_to_global = {k: i for i, k in enumerate(_global_electrode_keys())}
    global_to_key = {i: k for k, i in key_to_global.items()}

    with open(AUDIO_XCORR_JSON) as fh:
        xcorr = {(str(e["subject_id"]).zfill(2), str(e["electrode_name"])):
                 float(e["peak_correlation"]) for e in json.load(fh)}

    missing = [global_to_key[g] for g in roi if global_to_key[g] not in xcorr]
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(roi)} {roi_key!r} contacts have no audio "
            f"cross-correlation in {AUDIO_XCORR_JSON.name}: {missing[:5]}"
        )
    values = np.asarray([xcorr[global_to_key[g]] for g in roi])
    n_keep = max(1, int(len(roi) * fraction))
    keep = np.argsort(values)[:n_keep]
    return np.asarray(sorted(roi[i] for i in keep), dtype=np.int64)


# ---------------------------------------------------------------------------
# Lag indexing & window extraction.
# ---------------------------------------------------------------------------
def lag_to_index(lag_ms: int, origin_ms: int = CACHE_ORIGIN_MS,
                 step_ms: int = CACHE_STEP_MS) -> int:
    """Convert a lag in ms to a cache row index."""
    return int((lag_ms - origin_ms) // step_ms)


def lag_window_bounds(lag_ms: int, half_window_ms: int, n_lags: int,
                      origin_ms: int = CACHE_ORIGIN_MS,
                      step_ms: int = CACHE_STEP_MS) -> tuple[int, int]:
    """Clamped (lo, hi) inclusive cache indices for a half-window centred at lag_ms."""
    centre = lag_to_index(lag_ms, origin_ms, step_ms)
    half_pts = max(1, half_window_ms // step_ms)
    lo = max(0, centre - half_pts)
    hi = min(n_lags - 1, centre + half_pts)
    return lo, hi


def winsorize(activity: np.ndarray, n_sd: float = 3.0) -> np.ndarray:
    """Clip to ±n_sd standard deviations per electrode (across words)."""
    mean = activity.mean(axis=0, keepdims=True)
    std = np.maximum(activity.std(axis=0, keepdims=True), 1e-8)
    return np.clip(activity, mean - n_sd * std, mean + n_sd * std)


def extract_window(brain: np.ndarray, lag_ms: int, half_window_ms: int,
                   roi_idx: np.ndarray, *, do_winsorize: bool = True,
                   winsorize_sd: float = 3.0) -> np.ndarray:
    """Average a half-window centred at lag_ms across the ROI.

    Returns (n_words, n_electrodes) activity matrix.
    """
    n_lags = brain.shape[0]
    lo, hi = lag_window_bounds(lag_ms, half_window_ms, n_lags)
    activity = brain[lo:hi + 1][:, :, roi_idx].mean(axis=0)
    if do_winsorize:
        activity = winsorize(activity, winsorize_sd)
    return activity


# ---------------------------------------------------------------------------
# Baseline correction.
# ---------------------------------------------------------------------------
def baseline_mean(brain: np.ndarray, roi_idx: np.ndarray,
                  baseline_window_ms: tuple[int, int],
                  origin_ms: int = CACHE_ORIGIN_MS,
                  step_ms: int = CACHE_STEP_MS) -> np.ndarray:
    """Per-word, per-electrode mean over the baseline window. (n_words, n_elec)."""
    n_lags = brain.shape[0]
    lo, hi = lag_window_bounds(
        (baseline_window_ms[0] + baseline_window_ms[1]) // 2,
        (baseline_window_ms[1] - baseline_window_ms[0]) // 2,
        n_lags, origin_ms, step_ms,
    )
    accum = None
    for i in range(lo, hi + 1):
        slab = brain[i, :, :][:, roi_idx]
        accum = slab.astype(np.float32) if accum is None else accum + slab
    return accum / (hi - lo + 1)


# ---------------------------------------------------------------------------
# High-level extractor - encapsulates the full pipeline that every brain-side
# analysis needs. Most scripts should use this and never touch the helpers above.
# ---------------------------------------------------------------------------
class BaselineExtractor:
    """Pre-loads cache rows and serves baseline-corrected activity / RDMs.

    Construction loads only the slab of cache rows needed to evaluate the
    requested set of `eval_lags_ms` (and `+ delta_ms`, if non-zero) under the
    given `half_window_ms`. Baseline is always taken from the 25 ms cache so
    the long pre-onset window (e.g. -5000 .. -1000 ms) is always available,
    even when the 1 ms cache is used for fine-resolution evaluation.

    Methods
    -------
    activity(lag_ms) -> ndarray
        (n_words, n_electrodes) baseline-corrected, winsorised activity.
        Cached.
    rdm(lag_ms, word_subset=None) -> ndarray
        Condensed RDM over the requested word subset (or all words). Cached
        per (lag_ms, id(word_subset)).

    Properties
    ----------
    n_words : int
    n_electrodes : int
    """

    def __init__(self, *, roi_key: str | None = None,
                 electrode_indices: np.ndarray | None = None,
                 eval_lags_ms,
                 half_window_ms: int, winsorize_sd: float,
                 baseline_window_ms: tuple[int, int] | None,
                 delta_ms: int = 0,
                 cache: str = "25ms"):
        """Pass either roi_key (full ROI) or electrode_indices (custom subset).

        If baseline_window_ms is None, no baseline subtraction is applied;
        activity() returns winsorised raw activity instead.
        """
        # avoid circular import at top of file
        from .rdm import compute_rdm_condensed

        eval_path, eval_origin, eval_step, eval_json = cache_meta(cache)
        base_path, base_origin, base_step, base_json = cache_meta("25ms")

        eval_brain = load_brain_cache(eval_path, eval_json)
        if cache == "25ms":
            base_brain = eval_brain
        else:
            base_brain = load_brain_cache(base_path, base_json)

        self.n_words = eval_brain.shape[1]
        if (roi_key is None) == (electrode_indices is None):
            raise ValueError("Pass exactly one of roi_key / electrode_indices.")
        self.roi_idx = (np.asarray(electrode_indices, dtype=np.int64)
                        if electrode_indices is not None
                        else roi_indices(roi_key))
        self.n_electrodes = len(self.roi_idx)

        self.origin_ms = eval_origin
        self.step_ms = eval_step
        self.half_window_ms = half_window_ms
        self.winsorize_sd = winsorize_sd
        self._compute_rdm = compute_rdm_condensed

        # Baseline (n_words, n_electrodes) - always from 25 ms cache. None
        # means "no baseline correction".
        self._baseline = (
            None if baseline_window_ms is None
            else baseline_mean(base_brain, self.roi_idx, baseline_window_ms,
                               origin_ms=base_origin, step_ms=base_step)
        )

        # Determine the slab of eval cache rows we'll need.
        n_lags_cache = eval_brain.shape[0]
        needed: set[int] = set()
        for lag in eval_lags_ms:
            for ll in (lag, lag + delta_ms) if delta_ms else (lag,):
                lo, hi = lag_window_bounds(int(ll), half_window_ms, n_lags_cache,
                                           origin_ms=eval_origin,
                                           step_ms=eval_step)
                needed.update(range(lo, hi + 1))
        self._idx_min = min(needed)
        self._idx_max = max(needed)
        n_load = self._idx_max - self._idx_min + 1

        self._brain_roi = np.empty(
            (n_load, self.n_words, self.n_electrodes), dtype=np.float32,
        )
        for i in tqdm(range(n_load), desc="    pre-load", leave=False):
            self._brain_roi[i] = eval_brain[self._idx_min + i, :, :][:, self.roi_idx]

        self._half_pts = max(1, half_window_ms // eval_step)
        self._activity_cache: dict[int, np.ndarray] = {}
        self._rdm_cache: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------ activity
    def activity(self, lag_ms: int) -> np.ndarray:
        """Lag-windowed activity at lag_ms.

        - Optionally winsorised (skipped when winsorize_sd is None or <= 0).
        - Optionally baseline-corrected (skipped when baseline_window_ms was
          None at construction).
        """
        if lag_ms in self._activity_cache:
            return self._activity_cache[lag_ms]
        c = (lag_ms - self.origin_ms) // self.step_ms - self._idx_min
        lo = max(0, c - self._half_pts)
        hi = min(self._brain_roi.shape[0] - 1, c + self._half_pts)
        act = self._brain_roi[lo:hi + 1].mean(axis=0)
        if self.winsorize_sd is not None and self.winsorize_sd > 0:
            act = winsorize(act, self.winsorize_sd)
        if self._baseline is not None:
            act = act - self._baseline
        self._activity_cache[lag_ms] = act
        return act

    # --------------------------------------------------------------------- rdm
    def rdm(self, lag_ms: int, word_subset: np.ndarray | None = None) -> np.ndarray:
        """Condensed RDM at lag_ms for the given word subset (or all words)."""
        # id(word_subset) is stable for the lifetime of the array - keep the
        # array alive for the duration of analysis, do not mutate.
        key = (lag_ms, id(word_subset) if word_subset is not None else 0)
        if key in self._rdm_cache:
            return self._rdm_cache[key]
        act = self.activity(lag_ms)
        if word_subset is not None:
            act = act[word_subset]
        v = self._compute_rdm(act)
        self._rdm_cache[key] = v
        return v
