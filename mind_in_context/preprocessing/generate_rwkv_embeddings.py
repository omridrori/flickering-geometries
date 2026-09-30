"""Full-context word embeddings from RWKV-6 "Finch" 7B (recurrent model).

RWKV is an attention-free recurrent network, used to check that the brain-LLM
match does not depend on the transformer architecture. The whole transcript is
run through the model in a single forward pass, and the hidden state of the
requested layer is mean-pooled over the sub-tokens of each word.

Model: RWKV/v6-Finch-7B-HF at a pinned revision, loaded in 8-bit.

Input:
  <BIDS_ROOT>/stimuli/podcast_transcript.csv

Output:
  <data dir>/embeddings/rwkv/rwkv_L<layer>_ctx_full_emb.npy   (n_words, hidden), float16

Usage:
  python mind_in_context/preprocessing/generate_rwkv_embeddings.py
  python mind_in_context/preprocessing/generate_rwkv_embeddings.py --layer 23
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from mind_in_context.lib.io import EMBEDDINGS_DIR, PODCAST_TRANSCRIPT_CSV

MODEL_REPO = "RWKV/v6-Finch-7B-HF"
MODEL_REVISION = "02946e470978a56fc96bc8c47cb942c50bc9c71a"
DEFAULT_LAYER = 23


def emb_path(layer: int) -> Path:
    return EMBEDDINGS_DIR / "rwkv" / f"rwkv_L{layer}_ctx_full_emb.npy"


def generate(layer: int) -> None:
    import torch
    from numpy.lib.format import open_memmap
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    words = pd.read_csv(PODCAST_TRANSCRIPT_CSV)["word"].astype(str).tolist()
    n_words = len(words)

    print(f"  loading {MODEL_REPO}@{MODEL_REVISION[:8]} in 8-bit ...")
    tok = AutoTokenizer.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        "modeling_rwkv6.Rwkv6ForCausalLM", MODEL_REPO, revision=MODEL_REVISION)
    # The model class keeps some modules in fp32, which is incompatible with
    # 8-bit loading; clear that list on the base class.
    for base in cls.__mro__:
        if base.__name__ == "Rwkv6PreTrainedModel":
            base._keep_in_fp32_modules = []
    model = cls.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0},
    )
    model.eval()
    n_hidden = model.config.hidden_size

    print("  tokenizing words ...")
    word_ids = [tok(" " + w, return_tensors=None)["input_ids"] for w in words]
    word_ids = [wi if len(wi) > 0 else [0] for wi in word_ids]

    # One token sequence for the whole transcript, with each word's token span.
    full_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for wi in word_ids:
        start = len(full_ids)
        full_ids.extend(wi)
        spans.append((start, len(full_ids)))
    print(f"  {len(full_ids)} tokens for {n_words} words")

    input_t = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to("cuda:0")
    print("  running a single forward pass over the transcript ...")
    with torch.inference_mode():
        out = model(input_t, output_hidden_states=True)

    # Keep only the requested layer; move it to the CPU to free GPU memory.
    hs = out.hidden_states[layer][0].float().cpu()   # (n_tokens, hidden)
    del out
    torch.cuda.empty_cache()

    path = emb_path(layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = open_memmap(path, mode="w+", dtype=np.float16, shape=(n_words, n_hidden))
    for i, (s, e) in enumerate(tqdm(spans, desc="  pooling words")):
        emb[i] = hs[s:e].mean(dim=0).numpy().astype(np.float16)
    emb.flush()
    print(f"  -> {path}  shape={emb.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = parser.parse_args()
    generate(args.layer)


if __name__ == "__main__":
    main()
