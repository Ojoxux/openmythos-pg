"""Wikipedia-ja のロード・tokenize・memmap保存と、学習時のサンプリング。"""
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint32  # llm-jp vocab ~100k exceeds uint16


def tokenize_corpus(
    out_dir: Path | str,
    tokenizer_name: str = "llm-jp/llm-jp-3-1.8b",
    dataset_name: str = "wikipedia",
    dataset_config: str = "20231101.ja",
    val_ratio: float = 0.005,
    seed: int = 42,
) -> dict[str, int]:
    """Wikipedia-ja をtokenizeし、train.bin / val.bin に保存。

    Returns:
        dict with keys "train_tokens", "val_tokens", "vocab_size"
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = tokenizer.vocab_size
    print(f"[data] vocab_size = {vocab_size}")

    print(f"[data] loading dataset: {dataset_name}/{dataset_config}")
    ds = load_dataset(dataset_name, dataset_config, split="train")
    print(f"[data] {len(ds):,} articles")

    rng = np.random.default_rng(seed)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    train_count = 0
    val_count = 0
    eos_id = tokenizer.eos_token_id or 0

    with train_path.open("wb") as ftrain, val_path.open("wb") as fval:
        # batched tokenization for speed
        batch_size = 1000
        for batch_start in range(0, len(ds), batch_size):
            batch = ds[batch_start : batch_start + batch_size]
            encs = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
            for ids in encs:
                arr = np.array(ids + [eos_id], dtype=TOKEN_DTYPE)
                if rng.random() < val_ratio:
                    arr.tofile(fval)
                    val_count += len(arr)
                else:
                    arr.tofile(ftrain)
                    train_count += len(arr)
            if batch_start % 10000 == 0:
                print(
                    f"[data] {batch_start:,}/{len(ds):,} "
                    f"(train={train_count:,}, val={val_count:,})"
                )

    print(f"[data] DONE. train={train_count:,} val={val_count:,}")
    return {
        "train_tokens": train_count,
        "val_tokens": val_count,
        "vocab_size": vocab_size,
    }


def load_memmap(path: Path | str) -> np.memmap:
    """Load a tokens.bin as uint32 memmap. Length is inferred from file size."""
    path = Path(path)
    itemsize = np.dtype(TOKEN_DTYPE).itemsize
    size = path.stat().st_size // itemsize
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r", shape=(size,))


def sample_batch(
    mm: np.memmap,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """memmap からランダムに batch_size 個の (seq_len+1) チャンクを切り出し、(x, y) を返す。

    y[i, t] = x[i, t+1] (next-token prediction target).
    """
    n = len(mm)
    assert n > seq_len + 1, f"memmap too short: {n} <= seq_len+1 ({seq_len + 1})"
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    x = np.stack([np.array(mm[s : s + seq_len], dtype=np.int64) for s in starts])
    y = np.stack([np.array(mm[s + 1 : s + 1 + seq_len], dtype=np.int64) for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)
