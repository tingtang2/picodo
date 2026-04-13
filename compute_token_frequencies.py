"""Compute per-token frequency histogram from a pretokenized .bin dataset.

Usage:
    python compute_token_frequencies.py --ds_path ~/datasets/fineweb_gpt2.bin

Outputs a .npy file with shape [V] (V = vocab size, default 50257 for GPT-2)
containing the normalized frequency of each token in the training data.
Frequencies sum to 1.0.

The output path defaults to the same directory as the input, with the suffix
replaced by _freqs.npy: e.g. fineweb_gpt2.bin -> fineweb_gpt2_freqs.npy.
"""

import os
import fire
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


def compute_frequencies(
    ds_path: str,
    vocab_size: int = 50257,
    output_path: str = None,
):
    """Read a pretokenized .bin file and compute per-token frequencies.

    Args:
        ds_path: Path to the .bin file (same format as download_fineweb.py output).
        vocab_size: Vocabulary size (default 50257 for GPT-2).
        output_path: Where to save the .npy file. Defaults to <ds_path>_freqs.npy.
    """
    ds_path = os.path.expanduser(ds_path)
    if output_path is None:
        output_path = ds_path.replace(".bin", "_freqs.npy")

    # Read header
    header = np.fromfile(ds_path, dtype=np.int32, count=256)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    print(f"Dataset: {ds_path}")
    print(f"Tokens:  {num_tokens:_}")
    print(f"Vocab:   {vocab_size}")

    # Memory-map the token data (avoids loading the full file into RAM).
    tokens = np.memmap(ds_path, dtype=np.uint16, mode="r", offset=256 * 4)
    assert len(tokens) >= num_tokens

    # Compute histogram in chunks to keep memory bounded.
    chunk_size = 100_000_000  # 100M tokens per chunk
    counts = np.zeros(vocab_size, dtype=np.int64)
    n_chunks = (num_tokens + chunk_size - 1) // chunk_size
    for i in tqdm(range(n_chunks), desc="counting"):
        start = i * chunk_size
        end = min(start + chunk_size, num_tokens)
        chunk = np.asarray(tokens[start:end])  # force read from memmap
        # Clip any token IDs >= vocab_size (padding tokens from V-rounding).
        chunk = chunk[chunk < vocab_size]
        counts += np.bincount(chunk, minlength=vocab_size)

    # Normalize to frequencies (sum to 1).
    total = counts.sum()
    freqs = counts.astype(np.float32) / float(total)

    # Report statistics.
    nonzero = np.count_nonzero(counts)
    print(f"Nonzero tokens: {nonzero} / {vocab_size} ({100*nonzero/vocab_size:.1f}%)")
    print(f"Most common:  token {np.argmax(counts)} (freq {freqs[np.argmax(counts)]:.6f})")
    print(f"Least common (nonzero): freq {freqs[freqs > 0].min():.2e}")
    print(f"Median freq:  {np.median(freqs[freqs > 0]):.2e}")

    np.save(output_path, freqs)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    fire.Fire(compute_frequencies)
