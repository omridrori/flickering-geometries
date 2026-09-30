"""RDM construction and Spearman correlations between condensed RDMs.

GPU-accelerated when CUDA is available; falls back to scipy on CPU.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# GPU detection.
# ---------------------------------------------------------------------------
try:
    import torch
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_GPU else "cpu")
except Exception:
    torch = None
    USE_GPU = False
    DEVICE = None


def _torch_pdist_correlation(tensor_2d) -> "torch.Tensor":
    """Pairwise correlation distance (1 − Pearson r), upper-triangle, on GPU."""
    n_samples, n_features = tensor_2d.shape
    centered = tensor_2d - tensor_2d.mean(dim=1, keepdim=True)
    cov = (centered @ centered.T) / max(1, n_features - 1)
    stds = tensor_2d.std(dim=1)
    corr = cov / ((stds[:, None] @ stds[None, :]) + 1e-8)
    mask = torch.ones_like(corr, dtype=torch.bool).triu(diagonal=1)
    return (1.0 - corr)[mask]


def compute_rdm_condensed(activations: np.ndarray) -> np.ndarray:
    """Condensed correlation-distance RDM from (n_words, n_features) activations.

    Returns numpy ndarray on CPU; internally uses GPU when available.
    NaN/Inf values are zeroed before computation.
    """
    clean = np.where(np.isfinite(activations), activations, 0.0)
    if USE_GPU:
        result = _torch_pdist_correlation(
            torch.from_numpy(clean).float().to(DEVICE)
        )
        return result.cpu().numpy()
    return pdist(clean, metric="correlation")


def spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman rank correlation between two condensed RDMs."""
    if not USE_GPU or torch is None:
        rho, _ = spearmanr(np.asarray(rdm_a), np.asarray(rdm_b))
        return float(rho)
    a = torch.from_numpy(np.asarray(rdm_a)).float().to(DEVICE)
    b = torch.from_numpy(np.asarray(rdm_b)).float().to(DEVICE)
    a_r = a.argsort().argsort().float(); a_r -= a_r.mean()
    b_r = b.argsort().argsort().float(); b_r -= b_r.mean()
    num = (a_r * b_r).sum()
    den = torch.sqrt((a_r ** 2).sum()) * torch.sqrt((b_r ** 2).sum())
    return float((num / (den + 1e-8)).item())


def compute_rdm_condensed_gpu(activations: np.ndarray):
    """Same as compute_rdm_condensed but stays on GPU when available.

    Returns a GPU tensor on CUDA, or a numpy ndarray on CPU. Use inside hot
    loops that immediately feed the RDM into another GPU op.
    """
    clean = np.where(np.isfinite(activations), activations, 0.0)
    if USE_GPU and torch is not None:
        return _torch_pdist_correlation(
            torch.from_numpy(clean).float().to(DEVICE)
        )
    return pdist(clean, metric="correlation")


def rank_centered_gpu(vec):
    """Rank-transform `vec` and centre by mean - output stays on GPU.

    Use this to pre-rank a fixed RDM (e.g. an LLM RDM) once outside an inner
    loop, then call `spearman_from_ranked(centered_ranks, fresh_vec)` against
    many other vectors without re-ranking the fixed one.

    Falls back to numpy on CPU.
    """
    if not USE_GPU or torch is None:
        a = np.asarray(vec, dtype=np.float64)
        r = a.argsort().argsort().astype(np.float64)
        r -= r.mean()
        return r
    if isinstance(vec, np.ndarray):
        # `from_numpy` shares memory and warns on read-only memmaps; copy
        # to defuse the warning and ensure a writable buffer on GPU upload.
        t = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32)).to(DEVICE)
    else:
        t = vec.float()
    r = t.argsort().argsort().float()
    r -= r.mean()
    return r


def spearman_two_ranked(rx, ry) -> float:
    """Spearman ρ when BOTH inputs are already rank-centred. Pure dot-product."""
    if not USE_GPU or torch is None:
        num = float((rx * ry).sum())
        den = float(np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum()) + 1e-8)
        return num / den
    num = (rx * ry).sum()
    den = torch.sqrt((rx ** 2).sum()) * torch.sqrt((ry ** 2).sum())
    return float((num / (den + 1e-8)).item())


def spearman_from_ranked(centered_ranks_y, x_unranked) -> float:
    """Spearman ρ when y is already rank-centred (output of rank_centered_gpu).

    `x_unranked` may be a numpy array or a GPU tensor (e.g. the output of
    compute_rdm_condensed_gpu when USE_GPU is True). No host<->device transfer
    happens here on the GPU path - both inputs must already be on the same
    device.
    """
    if not USE_GPU or torch is None:
        x = np.asarray(x_unranked, dtype=np.float64)
        x_r = x.argsort().argsort().astype(np.float64)
        x_r -= x_r.mean()
        y_r = centered_ranks_y
        num = float((x_r * y_r).sum())
        den = float(np.sqrt((x_r ** 2).sum()) * np.sqrt((y_r ** 2).sum()) + 1e-8)
        return num / den
    if isinstance(x_unranked, np.ndarray):
        x = torch.from_numpy(x_unranked).float().to(DEVICE)
    else:
        x = x_unranked.float()
    x_r = x.argsort().argsort().float()
    x_r -= x_r.mean()
    num = (x_r * centered_ranks_y).sum()
    den = torch.sqrt((x_r ** 2).sum()) * torch.sqrt((centered_ranks_y ** 2).sum())
    return float((num / (den + 1e-8)).item())


def lag_window_average(curve: np.ndarray, half_window_pts: int) -> np.ndarray:
    """Centered moving average along the lag axis, shrinking at edges.

    Used to smooth dynamic-RSA curves over neighbouring lag indices. NaNs are
    treated as missing - if a window is all-NaN the result is NaN.
    """
    x = np.asarray(curve, dtype=np.float64)
    n = int(x.shape[0])
    h = int(max(0, half_window_pts))
    if h == 0 or n == 0:
        return x.copy()
    finite = np.isfinite(x)
    vals = np.where(finite, x, 0.0)
    csum = np.cumsum(vals)
    ccount = np.cumsum(finite.astype(np.int64))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n - 1, i + h)
        sum_i = csum[hi] - (csum[lo - 1] if lo > 0 else 0.0)
        cnt_i = int(ccount[hi] - (ccount[lo - 1] if lo > 0 else 0))
        out[i] = (sum_i / cnt_i) if cnt_i > 0 else np.nan
    return out


def compute_grouped_erg(extractor, eval_lags_ms, groups, delta_ms: int):
    """ρ_ERG(t) = Spearman(RDM(t), RDM(t+delta_ms)) for each (group, lag).

    Parameters
    ----------
    extractor : BaselineExtractor
        Pre-built with eval_lags_ms (and the same delta_ms) so all RDMs hit cache.
    eval_lags_ms : iterable of int
    groups : list[np.ndarray]
        Each entry is an array of word indices defining the group.
    delta_ms : int
        Lag offset for the t+delta partner.

    Returns
    -------
    rho : ndarray of shape (len(groups), len(eval_lags_ms))
    """
    from tqdm import tqdm
    rho = np.full((len(groups), len(list(eval_lags_ms))), np.nan)
    eval_lags = list(eval_lags_ms)
    for t, lag in enumerate(tqdm(eval_lags, desc="    rho_ERG")):
        for k, g in enumerate(groups):
            a = extractor.rdm(int(lag), g)
            b = extractor.rdm(int(lag) + delta_ms, g)
            rho[k, t] = spearman(a, b)
    return rho
