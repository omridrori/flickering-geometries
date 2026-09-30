"""Full-context word embeddings from Llama-3-8B.

Every word is embedded with the whole preceding narrative available. The
transcript is fed in chunks while the key-value cache is kept, so chunk N
attends to chunks 1..N-1: this is equivalent to a single pass over the full
sequence, with a lower peak memory.

The hidden state of each requested layer is mean-pooled over the sub-tokens of
each word. The model is loaded with 4-bit NF4 quantisation (bitsandbytes),
which fits the full sequence on a 12 GB GPU.

Llama-3 is a gated model: accept its licence on Hugging Face and set HF_TOKEN.

Input:
  <BIDS_ROOT>/stimuli/podcast_transcript.csv

Output:
  <data dir>/embeddings/llama3/full_context/embeddings/
      llama3_layer_<L>_full_context_word_embeddings.npy     (n_words, 4096)

Usage (layers as positional arguments; default 0, 5, 10, 15, 20, 25, 30, 31):
  python mind_in_context/preprocessing/generate_llama3_embeddings_full_context.py
  python mind_in_context/preprocessing/generate_llama3_embeddings_full_context.py 15 25
"""

from __future__ import annotations

import json
import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mind_in_context.lib.io import EMBEDDINGS_DIR as EMBEDDINGS_ROOT  # noqa: E402
from mind_in_context.lib.io import PODCAST_TRANSCRIPT_CSV  # noqa: E402

# --------------------------------------------------------------------------------------
# Paths & Configuration
# --------------------------------------------------------------------------------------

TRANSCRIPT_CSV: Path = PODCAST_TRANSCRIPT_CSV

RESULTS_DIR: Path = EMBEDDINGS_ROOT / "llama3" / "full_context"
EMBEDDINGS_DIR: Path = RESULTS_DIR / "embeddings"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# Chunk size for processing (tokens per forward pass)
# 512-1024 is usually a sweet spot for speed/memory balance with cache
CHUNK_SIZE = 512

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def ensure_directories() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def print_gpu_memory_usage(device: torch.device, label: str = "") -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available() and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory {label}: Alloc: {allocated:.2f} GB, Rsrv: {reserved:.2f} GB")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def process_single_layer(
    model,
    tokenizer,
    words: List[str],
    n_words: int,
    layer: int,
    hidden_size: int,
    device: torch.device
) -> None:
    """Process a single layer and save embeddings."""
    print(f"\n{'='*60}")
    print(f"Processing Layer {layer}")
    print(f"{'='*60}")

    # Tokenize EVERYTHING
    print("Tokenizing entire transcript...")
    tokenized_full = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    full_input_ids = tokenized_full["input_ids"][0]  # Flatten to 1D
    full_word_ids = tokenized_full.word_ids()
    total_tokens = len(full_input_ids)
    print(f"Total tokens: {total_tokens}")

    # Prepare Output File
    out_path = EMBEDDINGS_DIR / f"llama3_layer_{layer}_full_context_word_embeddings.npy"
    print(f"Preparing output file: {out_path}")
    memmap = open_memmap(out_path, dtype="float32", mode="w+", shape=(n_words, hidden_size))

    # Process in chunks with KV Cache
    print(f"Processing in chunks of {CHUNK_SIZE} tokens with KV cache...")

    past_key_values = None
    all_hidden_states = []

    # Create chunks
    num_chunks = (total_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE

    with torch.inference_mode():
        for i in tqdm(range(num_chunks), desc=f"Layer {layer} Forward Pass"):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, total_tokens)

            chunk_input_ids = full_input_ids[start_idx:end_idx].unsqueeze(0).to(device)

            # Forward pass with cache
            outputs = model(
                input_ids=chunk_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # Extract specific layer for THIS chunk
            chunk_states = outputs.hidden_states[layer].detach().cpu().squeeze(0)
            all_hidden_states.append(chunk_states)

            del outputs
            torch.cuda.empty_cache()

    print("Forward pass complete. Concatenating states...")
    full_hidden_states = torch.cat(all_hidden_states, dim=0)
    print(f"Full hidden states shape: {full_hidden_states.shape}")

    print("Aggregating tokens to words...")

    # Aggregate
    valid_indices = [i for i, w in enumerate(full_word_ids) if w is not None]
    current_word_idx = 0
    current_token_vectors = []

    for i in tqdm(valid_indices, desc=f"Layer {layer} Aggregating"):
        w_idx = full_word_ids[i]

        if w_idx != current_word_idx:
            if current_token_vectors:
                avg_vec = torch.stack(current_token_vectors).mean(dim=0)
                if avg_vec.dtype == torch.bfloat16:
                    avg_vec = avg_vec.float()
                if current_word_idx < n_words:
                    memmap[current_word_idx] = avg_vec.numpy()

            current_word_idx = w_idx
            current_token_vectors = []

        current_token_vectors.append(full_hidden_states[i])

    # Last word
    if current_token_vectors and current_word_idx < n_words:
        avg_vec = torch.stack(current_token_vectors).mean(dim=0)
        if avg_vec.dtype == torch.bfloat16:
            avg_vec = avg_vec.float()
        memmap[current_word_idx] = avg_vec.numpy()

    memmap.flush()
    print(f"Saved embeddings to {out_path}")

    # Clean up
    del full_hidden_states, all_hidden_states, past_key_values
    torch.cuda.empty_cache()
    gc.collect()

def main(layers: Optional[List[int]] = None) -> None:
    ensure_directories()

    print(f"--- Generating Llama-3 Embeddings (Full Context via KV Cache) ---")
    print(f"Transcript: {TRANSCRIPT_CSV}")
    print(f"Model: {MODEL_NAME}")

    if "HF_TOKEN" not in os.environ:
        print("\nWARNING: HF_TOKEN environment variable not found.")

    # 1. Load Data
    print("Loading transcript...")
    df = pd.read_csv(TRANSCRIPT_CSV)
    words = df["word"].astype(str).tolist()
    n_words = len(words)
    print(f"Total words: {n_words}")

    # 2. Load Model & Tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using compute dtype: {compute_dtype}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded. Hidden size: {hidden_size}, Total layers: {num_layers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_memory_usage(device, "After load")

    # 3. Determine layers to process
    if layers is None:
        # Default: 5, 10, 15, 20, 25, 30, plus layer 0 and last layer
        layers = [5, 10, 15, 20, 25, 30]
        if 0 not in layers:
            layers.insert(0, 0)
        last_layer = num_layers - 1
        if last_layer not in layers:
            layers.append(last_layer)

    print(f"\nProcessing layers: {layers}")
    print(f"Total layers to process: {len(layers)}")

    # 4. Process each layer
    for layer in tqdm(layers, desc="Processing all layers"):
        try:
            process_single_layer(
                model=model,
                tokenizer=tokenizer,
                words=words,
                n_words=n_words,
                layer=layer,
                hidden_size=hidden_size,
                device=device
            )
        except Exception as e:
            print(f"\nERROR processing layer {layer}: {e}")
            print("Continuing with next layer...")
            continue

    # 5. Save metadata
    metadata = {
        "model": MODEL_NAME,
        "layers": layers,
        "type": "full_context_kv_cache",
        "n_words": n_words,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }
    with open(RESULTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("--- Done processing all layers ---")
    print("="*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("layers", type=int, nargs="*",
                        help="hidden_states indices to extract "
                             "(default: 0, 5, 10, 15, 20, 25, 30 and the last layer)")
    args = parser.parse_args()
    main(layers=args.layers or None)
