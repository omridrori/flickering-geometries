"""Full-context word embeddings from Mistral-7B (robustness model).

The whole transcript is passed through the model in a single forward pass with
causal attention, so the representation of each word is conditioned on the
entire preceding narrative. The hidden state of the requested layer is
mean-pooled over the sub-tokens of each word.

The model is loaded with 4-bit NF4 quantisation (bitsandbytes), which fits the
full sequence on a 12 GB GPU.

Input:
  <BIDS_ROOT>/stimuli/podcast_transcript.csv

Output:
  <data dir>/embeddings/mistral/full_context/embeddings/
      mistral_layer_<L>_full_context_word_embeddings.npy     (n_words, 4096)

Usage:
  python mind_in_context/preprocessing/generate_mistral_embeddings_full_context.py
  python mind_in_context/preprocessing/generate_mistral_embeddings_full_context.py --layer 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mind_in_context.lib.io import EMBEDDINGS_DIR, PODCAST_TRANSCRIPT_CSV

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DEFAULT_LAYER = 15        # index into the Hugging Face hidden_states tuple
PREPEND_BOS = True

RESULTS_DIR = EMBEDDINGS_DIR / "mistral" / "full_context"
EMB_DIR = RESULTS_DIR / "embeddings"


def load_words() -> list[str]:
    return pd.read_csv(PODCAST_TRANSCRIPT_CSV)["word"].astype(str).tolist()


def tokenize_words(words: list[str], tokenizer) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Token ids of the whole transcript, and each word's token span [start, end)."""
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    raw_ids = enc["input_ids"]
    word_ids = np.array(enc.word_ids())

    bos_id = tokenizer.bos_token_id
    has_bos = bool(bos_id is not None and PREPEND_BOS)
    offset = 1 if has_bos else 0

    _, first_token = np.unique(word_ids, return_index=True)
    starts = first_token + offset
    ends = np.append(starts[1:], len(word_ids) + offset)

    ids = ([int(bos_id)] + raw_ids) if has_bos else raw_ids
    return torch.tensor(ids, dtype=torch.long), starts, ends


def full_context_word_embeddings(words: list[str], tokenizer, model,
                                 device: torch.device, layer: int) -> np.ndarray:
    """One forward pass over the transcript, mean-pooled per word -> (n_words, hidden)."""
    input_ids, starts, ends = tokenize_words(words, tokenizer)
    seq_len = int(input_ids.shape[0])
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and seq_len > int(max_pos):
        raise ValueError(f"Sequence too long for the model ({seq_len} > {max_pos}).")

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)
    if layer >= len(outputs.hidden_states):
        raise ValueError(f"Layer {layer} requested, the model returned "
                         f"{len(outputs.hidden_states)} hidden states.")

    h = outputs.hidden_states[layer][0]             # (seq_len, hidden)
    out = np.zeros((len(words), int(h.shape[-1])), dtype=np.float32)
    for w in tqdm(range(len(words)), desc="Pooling words"):
        out[w] = h[int(starts[w]):int(ends[w])].mean(dim=0).float().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = parser.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = EMB_DIR / f"mistral_layer_{args.layer}_full_context_word_embeddings.npy"
    print(f"Model : {MODEL_NAME}\nLayer : {args.layer}\nOutput: {emb_path}")

    words = load_words()
    print(f"Transcript: {len(words)} words")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb = full_context_word_embeddings(words, tokenizer, model, device, args.layer)
    np.save(emb_path, emb.astype(np.float32))
    with open(RESULTS_DIR / f"metadata_layer_{args.layer}.json", "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL_NAME, "layer": args.layer,
                   "n_words": int(emb.shape[0]), "hidden_size": int(emb.shape[1]),
                   "prepend_bos": PREPEND_BOS}, fh, indent=2)
    print(f"Saved {emb_path}")


if __name__ == "__main__":
    main()
