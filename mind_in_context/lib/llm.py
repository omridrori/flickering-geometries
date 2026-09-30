"""LLM embedding loading and RDM construction.

Primary model: Llama-3-8B. Mistral-7B is supported as a robustness check via
the --model flag. Embeddings are pre-computed by the scripts in
preprocessing/ and stored under <data dir>/embeddings/:

  <model>/context_windows/layer_<L>/embeddings/<model>_layer_<L>_window_<K>_word_embeddings.npy
      each word embedded with attention restricted to its K preceding words
  <model>/full_context/embeddings/<model>_layer_<L>_full_context_word_embeddings.npy
      each word embedded with the whole preceding narrative available

where <model> is "llama3" or "mistral".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize

from .io import EMBEDDINGS_DIR, load_config
from .rdm import compute_rdm_condensed

# Folder / file prefix used for each config model key.
_MODEL_SHORT = {"llama3": "llama3", "mistral7b": "mistral"}


def _short(model_key: str) -> str:
    cfg = load_config()
    if model_key not in cfg["models"]:
        raise ValueError(f"Unknown model '{model_key}'")
    return _MODEL_SHORT[model_key]


def full_context_embeddings_path(model_key: str, layer: int | None = None) -> Path:
    """Path of the full-narrative embedding file for (model, layer)."""
    short = _short(model_key)
    if layer is None:
        layer = load_config()["models"][model_key]["layer"]
    return (EMBEDDINGS_DIR / short / "full_context" / "embeddings"
            / f"{short}_layer_{layer}_full_context_word_embeddings.npy")


def embeddings_path(model_key: str, layer: int, context_length) -> Path:
    """Path of the embedding file for (model, layer, context_length).

    context_length is a number of preceding words K, or "full" for the whole
    preceding narrative.
    """
    if context_length == "full":
        return full_context_embeddings_path(model_key, layer)
    short = _short(model_key)
    return (EMBEDDINGS_DIR / short / "context_windows" / f"layer_{layer}" / "embeddings"
            / f"{short}_layer_{layer}_window_{context_length}_word_embeddings.npy")


def load_embeddings(model_key: str, layer: int | None = None,
                    context_length="full") -> np.ndarray:
    """Load per-word embeddings -> (n_words, embedding_dim)."""
    cfg = load_config()
    if layer is None:
        layer = cfg["models"][model_key]["layer"]
    path = embeddings_path(model_key, layer, context_length)
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {path}\n"
            "Generate them with the scripts in preprocessing/ (see README).")
    return np.load(path)


def llm_rdm(model_key: str, layer: int | None = None,
            context_length="full") -> np.ndarray:
    """Condensed correlation-distance RDM for (model, layer, context)."""
    return compute_rdm_condensed(load_embeddings(model_key, layer, context_length))


def cosine_rdm(embeddings: np.ndarray) -> np.ndarray:
    """Model RDM used for the brain–LLM comparison.

    Column-center + row L2-normalize + cosine pdist. After centering and
    normalising, cosine(x, y) ≈ 1 + Pearson r between the centered vectors,
    so this is *not* identical to scipy's pdist(correlation) - and the
    difference matters at the few-thousandths level we're measuring.
    """
    centered = embeddings - embeddings.mean(axis=0)
    normalized = normalize(centered, axis=1, norm="l2")
    return pdist(normalized, metric="cosine")


def cosine_rdm_cached(model_key: str, layer: int, context_length,
                      use_pca: bool = False) -> np.ndarray:
    """cosine_rdm with an on-disk cache, to avoid recomputing 13.2-million-pair
    RDMs on every run.

    The cache lives next to the embeddings, in
    `<embeddings folder>/rdm_cache/model_rdm_<hash>.npy`, keyed by
        sha256(f"stem={stem};n_words={N};use_pca={0|1}")[:16]
    """
    emb_path = embeddings_path(model_key, layer, context_length)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}\n"
            "Generate them with the scripts in preprocessing/ (see README).")

    emb = np.load(emb_path)
    n_words = int(emb.shape[0])
    pca_flag = 1 if use_pca else 0
    desc = f"stem={emb_path.stem};n_words={n_words};use_pca={pca_flag}"
    key = hashlib.sha256(desc.encode("utf-8")).hexdigest()[:16]

    cache_dir = emb_path.parent / "rdm_cache"
    npy_path = cache_dir / f"model_rdm_{key}.npy"
    meta_path = cache_dir / f"model_rdm_{key}.json"

    if npy_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            if (str(meta.get("embeddings_stem", "")) == emb_path.stem
                    and int(meta.get("n_words", -1)) == n_words
                    and int(meta.get("use_pca", -1)) == pca_flag):
                return np.load(npy_path, mmap_mode="r")
        except Exception:
            pass

    rdm = cosine_rdm(emb.astype(np.float64))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, rdm)
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "cache_key": key,
            "embeddings_stem": emb_path.stem,
            "n_words": n_words,
            "use_pca": pca_flag,
            "desc": desc,
        }, f, indent=2)
    tmp.replace(meta_path)
    return rdm
