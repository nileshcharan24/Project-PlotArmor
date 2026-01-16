"""
Streaming pre-tokenizer for large text corpora.

Reads input text in chunks (lines-based) to avoid loading the entire file into RAM,
tokenizes with GPT2TokenizerFast, and writes token IDs directly to a binary file via
np.memmap. Shows real-time progress via tqdm. Designed to be safe on 16GB RAM laptops
and runnable on Kaggle as a preprocessing step (e.g., `!python tools/pre_tokenize.py`).
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import GPT2TokenizerFast
from tqdm import tqdm


def estimate_total_lines(input_path: str) -> Optional[int]:
    """
    Roughly estimate total lines for tqdm total. Returns None if cannot estimate.
    """
    try:
        with open(input_path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def pre_tokenize(
    input_path: str,
    output_path: str,
    chunk_lines: int = 10_000,
    dtype: np.dtype = np.uint16,
) -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare output paths
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: stream lines, tokenize, and write to memmap in chunks to avoid RAM blowup
    total_lines_est = estimate_total_lines(input_path)

    # We do a two-pass approach to know total token count without storing all tokens in RAM:
    # Pass 1: count tokens
    total_tokens = 0
    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_lines_est, desc="Counting tokens", unit="lines", disable=total_lines_est is None)
        while True:
            lines = []
            for _ in range(chunk_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            encodings = tokenizer(
                lines,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=1024,
            )
            total_tokens += sum(len(ids) for ids in encodings["input_ids"])
            pbar.update(len(lines))
        pbar.close()

    # Allocate memmap with known length
    tokens_memmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))

    # Pass 2: write tokens into memmap
    write_pos = 0
    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_lines_est, desc="Writing tokens", unit="lines", disable=total_lines_est is None)
        while True:
            lines = []
            for _ in range(chunk_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            encodings = tokenizer(
                lines,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=1024,
            )
            for ids in encodings["input_ids"]:
                end_pos = write_pos + len(ids)
                tokens_memmap[write_pos:end_pos] = np.asarray(ids, dtype=dtype)
                write_pos = end_pos
            pbar.update(len(lines))
        pbar.close()

    tokens_memmap.flush()
    print(f"Saved pretokenized tokens to {output_path} (total tokens: {total_tokens}, dtype: {dtype})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize large text file into memmap binary")
    parser.add_argument("--input", type=str, default="research/data/tinystories_train.txt", help="Input text file")
    parser.add_argument("--output", type=str, default="research/data/tinystories_train.bin", help="Output memmap/binary file")
    parser.add_argument("--chunk_lines", type=int, default=10_000, help="Number of lines per chunk")
    args = parser.parse_args()

    pre_tokenize(args.input, args.output, args.chunk_lines)
