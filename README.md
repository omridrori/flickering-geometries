# Flickering geometries

Code for the paper

> **Flickering geometries: ultra-fast and infra-slow context-related transformations of
> representational similarity structures during narrative comprehension in the human brain**
> Omri Drori, Doron Friedman, Yacov Hel-Or, Rafael Malach (under review).

The analyses use the public "Podcast" ECoG dataset (OpenNeuro
[ds005574](https://openneuro.org/datasets/ds005574)) and word embeddings from three language
models (Llama-3-8B, Mistral-7B, RWKV-6 7B). Every figure of the paper that is produced by code
can be regenerated from this repository.

---

## 1. Repository layout

```
mind_in_context/
  config.yaml        all shared parameters (ROIs, lags, window, Δ, colours)
  lib/               shared code: data loading, RDMs, statistics, plotting style
  preprocessing/     turn the raw dataset into the inputs the analyses read
  analyses/          one script per analysis; each writes results/<name>/<roi>....npz
  analyses/statistics/  the statistical tests reported in the text
  figures/           one script per figure; reads results/, writes plots/<fig>.svg
  results/           the numerical results behind every figure (small .npz files)
  plots/             the figures as produced by figures/
  cache/roi_electrode_indices.json   contacts of each region (atlas lookup, fixed)
```

Large files (the dataset, the word-locked brain caches, the embeddings) are **not** in the
repository. They go in a data directory, `data/` next to `mind_in_context/` by default:

```
data/
  ds005574/            the OpenNeuro dataset                 (step 3)
  transcript.tsv       preprocessing/make_transcript.py      (step 4)
  audio_xcorr.json     preprocessing/generate_audio_xcorr.py (step 4)
  cache/               preprocessing/generate_brain_cache.py (step 4)
  embeddings/          preprocessing/generate_*_embeddings*.py (step 5)
```

Two environment variables move it elsewhere:

| variable      | meaning                        | default               |
|---------------|--------------------------------|-----------------------|
| `FG_DATA_DIR` | the data directory             | `<repo>/data`         |
| `BIDS_ROOT`   | the dataset root               | `$FG_DATA_DIR/ds005574` |

---

## 2. Installation

### System requirements

- **Tested on:** Windows 11 Pro (build 26200), Python 3.11.10, with the package versions
  pinned in `requirements.txt` (PyTorch 2.5.1 with CUDA 12.1).
- **Redrawing the figures:** any standard computer; no special hardware.
- **Recomputing the analyses:** about 65 GB of free disk space for the brain caches. A CUDA GPU
  speeds up the RDM computations; without one they run on the CPU.
- **Generating the embeddings:** an NVIDIA GPU with 12 GB of memory (tested on an RTX 4000 Ada
  Laptop GPU, 12 GB).

### Install

```bash
conda create -n flickering python=3.11
conda activate flickering
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Typical install time: about 5 minutes (measured on a laptop, 3 of them for the PyTorch download).

All commands below are run from the repository root.

### Quick demo (no data needed)

```bash
python mind_in_context/figures/fig2.py
```

This redraws Figure 2 of the paper from the results included in the repository and writes it to
`mind_in_context/plots/fig2.svg`. Expected output: the same figure as in the paper.

---

## 3. Download the data

The dataset is OpenNeuro **ds005574** ("Podcast" ECoG, 9 patients, a 30-minute story):
<https://openneuro.org/datasets/ds005574>.

Download it with DataLad, the OpenNeuro CLI, or the "Download" button on the dataset page, into
`data/ds005574`. For example:

```bash
pip install openneuro-py
openneuro-py download --dataset ds005574 --target-dir data/ds005574
```

The analyses use these parts of it:

| path inside ds005574                              | used for                         |
|---------------------------------------------------|----------------------------------|
| `derivatives/ecogprep/sub-XX/ieeg/*highgamma*.fif`| high-gamma activity (70-200 Hz) and contact positions |
| `stimuli/podcast.wav`                             | audio control                    |
| `stimuli/podcast_transcript.csv`                  | words and onsets                 |
| `stimuli/en_core_web_lg/transcript.tsv`           | word onsets for the brain cache  |
| `stimuli/gpt2-xl/transcript.tsv`                  | GPT-2-XL surprisal of each word  |

---

## 4. Preprocessing: brain data

Run once, in this order.

```bash
# word table: word_idx, word, start, end
python mind_in_context/preprocessing/make_transcript.py

# word-locked high-gamma caches, shape (n_lags, n_words, n_electrodes)
python mind_in_context/preprocessing/generate_brain_cache.py --resolution 25ms   # ~10 GB
python mind_in_context/preprocessing/generate_brain_cache.py --resolution 1ms    # ~52 GB

# each electrode's peak cross-correlation with the speech envelope (audio control)
python mind_in_context/preprocessing/generate_audio_xcorr.py
```

The 25 ms cache (lags −5000..5000 ms) is used by most analyses. The 1 ms cache
(lags −1000..999 ms) is used where a peak latency is estimated (Figure 5,
Extended Data Figs. 1, 6, 8). The region contacts (73 language, 135 auditory) are read from
`mind_in_context/cache/roi_electrode_indices.json`.

---

## 5. Preprocessing: language-model embeddings

The models are downloaded from Hugging Face on first use:

| model | Hugging Face id | used in |
|-------|-----------------|---------|
| Llama-3-8B (primary) | `meta-llama/Meta-Llama-3-8B` | Figs. 4-6, ED 1, 2, 5, 6, 8, 10, Supp. 1 |
| Mistral-7B           | `mistralai/Mistral-7B-v0.1`  | ED 2 |
| RWKV-6 "Finch" 7B    | `RWKV/v6-Finch-7B-HF` (revision pinned in the script) | ED 2 |

**Llama-3 is a gated model.** Accept its licence on its Hugging Face page, then create an access
token and make it available:

```bash
export HF_TOKEN=hf_...        # Windows PowerShell:  $env:HF_TOKEN = "hf_..."
```

Then:

```bash
# Llama-3, full preceding narrative as context (layers 0,5,10,15,20,25,30,31)
python mind_in_context/preprocessing/generate_llama3_embeddings_full_context.py

# Llama-3, fixed context windows of 1..20 words, for the layers of Figure 5
python mind_in_context/preprocessing/generate_llama3_embeddings_context_windows.py --layers 5 10 15 20 25 30

# Mistral-7B, full context, layer 15
python mind_in_context/preprocessing/generate_mistral_embeddings_full_context.py

# RWKV-6, full context, layer 23
python mind_in_context/preprocessing/generate_rwkv_embeddings.py
```

Each script's `--help` lists its options and output paths.

---

## 6. Reproducing the figures

Every analysis writes its numbers to `mind_in_context/results/` and every figure script reads
only from there. The results of the paper are included, so **all figures except 4 and 6 can be
redrawn immediately, without the data**:

```bash
python mind_in_context/figures/fig2.py
python mind_in_context/figures/figS_lopo.py --roi lang
```

Figures are written to `mind_in_context/plots/`.

Figures 4 and 6 also need element-wise correspondence matrices, which are about 100 MB each and
therefore not included; the commands below generate them.

To recompute a figure from the data, run its analysis commands first, then the figure script.
Commands are written as `python -m mind_in_context.<module>`; default options reproduce the paper
unless shown. `A` = `mind_in_context.analyses`, `S` = `mind_in_context.analyses.statistics`.

| Figure | Analysis commands | Figure script |
|---|---|---|
| **Fig. 2** ERG builds up with context | `A.segment_erg_psth --roi lang --n-segments 5`<br>`A.segment_erg_psth --roi lang --n-segments 20`<br>`A.rdm_similarity_2d --roi lang`<br>test: `S.erg_psth_amplitude_coupling --roi lang` | `fig2.py` |
| **Fig. 3** high- vs low-surprisal words | `A.surprisal_split --roi lang`<br>`A.rdm_similarity_2d_surprisal --roi lang --baseline diag`<br>test: `S.surprisal_dip_permutation --roi lang` | `fig3.py` |
| **Fig. 4** brain-LLM match over time | `A.brain_llm_rsa --roi lang --model llama3`<br>`A.brain_llm_rsa_permutation --roi lang --model llama3`<br>`A.brain_llm_rsa_circshift --roi lang --model llama3`<br>`A.brain_llm_correspondence --roi lang --model llama3 --lag-ms 350 --smooth-half-window-ms 50`<br>`A.brain_llm_correspondence --roi lang --model llama3 --lag-ms 350 --smooth-half-window-ms 50 --random-shifts 10 --shift-range 600 2568 --shift-seed 0` | `fig4.py` |
| **Fig. 5** match delayed with longer context | `A.erg --roi lang --full`<br>`A.rsa_by_context --roi lang --model llama3`<br>`A.rsa_peak_by_layer --roi lang --model llama3`<br>tests: `S.peak_lag_monotonicity --model llama3`, `S.peak_lag_monotonicity --model llama3 --layer 25`, `S.layer_slope_correlation --model llama3` | `fig5.py` |
| **Fig. 6** match by word-pair distance | `A.brain_llm_correspondence --roi lang --model llama3 --lags-ms 100 400 --smooth-half-window-ms 50`<br>test: `A.brain_llm_far_gain_circshift --roi lang --model llama3` | `fig6.py` |
| **ED Fig. 1** leave one patient out | `A.leave_one_patient_out --roi lang` | `figS_lopo.py --roi lang` |
| **ED Fig. 2** three model architectures | `A.brain_llm_rsa --roi lang --model llama3`<br>`A.brain_llm_rsa --roi lang --model mistral7b`<br>`A.brain_llm_rsa_rwkv --roi lang` | `figS_brain_llm_rsa_models.py` |
| **ED Fig. 3** noise-ceiling corrected ERG | `A.cross_half_erg --roi lang` | `figS_cross_half_erg.py --roi lang` |
| **ED Fig. 4** surprisal, repeated words removed | `A.surprisal_split_dedup --roi lang` | `figS_surprisal_dedup.py --roi lang` |
| **ED Fig. 5** brain-LLM, repeated words removed | `A.brain_llm_rsa_dedup --roi lang --model llama3` | `figS_brain_llm_dedup.py --roi lang` |
| **ED Fig. 6** context curves, not rescaled | `A.rsa_by_context --roi lang --model llama3`<br>`A.rsa_by_context --roi aud --model llama3` | `figS_fig5_raw.py` |
| **ED Fig. 8** precision of each peak lag | `A.peak_lag_subsample_ci --roi lang --model llama3` | `figS_peak_lag_ci.py --roi lang` |
| **ED Fig. 10** distance, window by window | `A.brain_llm_distance_by_window --roi lang --model llama3` | `figS_distance_by_window.py` |
| **Supp. Fig. 1** audio control | `A.low_audio_random_subsets --roi lang --model llama3` | `figS_low_audio_subset.py --roi lang` |
| **Supp. Fig. 2** surprisal across the story | `A.surprisal_by_segment --n-segments 20` | `figS_surprisal_by_segment.py` |

Figure 1 and the Methods figures are schematics. Extended Data Figs. 7 and 9 are not yet
scripted in this repository.

Every script accepts `--help`.

---

## 7. Licence

MIT, see `LICENSE`. The dataset and the language models are distributed under their own terms.
