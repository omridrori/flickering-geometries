"""Write the word-level transcript table used by the analyses.

The dataset ships the narrative as stimuli/podcast_transcript.csv, one row per
word (word, start, end; times in seconds). This adds the running word index and
saves it as a tab-separated file:

  <data dir>/transcript.tsv      columns: word_idx, word, start, end

Usage:
  python mind_in_context/preprocessing/make_transcript.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from mind_in_context.lib.io import PODCAST_TRANSCRIPT_CSV, TRANSCRIPT_TSV


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    if not PODCAST_TRANSCRIPT_CSV.exists():
        raise FileNotFoundError(
            f"{PODCAST_TRANSCRIPT_CSV} not found. Download ds005574 first (see README).")
    df = pd.read_csv(PODCAST_TRANSCRIPT_CSV)
    df.insert(0, "word_idx", range(len(df)))
    TRANSCRIPT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df[["word_idx", "word", "start", "end"]].to_csv(TRANSCRIPT_TSV, sep="\t", index=False)
    print(f"{len(df)} words -> {TRANSCRIPT_TSV}")


if __name__ == "__main__":
    main()
