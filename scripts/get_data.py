import datasets
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from pathlib import Path
from itertools import islice, chain
import argparse
from typing import Iterable, Dict, Any


def _expand_examples(example_iter, series_fields):
    """GluonTS-style generator from a stream of HF examples."""
    for ex in example_iter:
        start = np.datetime64(ex["timestamp"][0], "s")
        for field in series_fields:
            yield {"start": start, "target": np.asarray(ex[field])}


def _series_fields_from_first(example):
    # mimic the original "all sequence fields except timestamp"
    return [
        k
        for k, v in example.items()
        if k != "timestamp" and isinstance(v, (list, tuple, np.ndarray))
    ]


def _batched(iterable: Iterable[Dict[str, Any]], batch_size: int):
    """Yield lists of dicts of length batch_size (last one may be shorter)."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_sharded(
    ex_iter: Iterable[Dict[str, Any]],
    out_dir: Path,
    compression: str | None,
    shard_size: int,
):
    """
    Write an iterator of GluonTS-style dicts into multiple .arrow shards.
    Each shard is written independently so memory stays bounded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    for shard in _batched(ex_iter, shard_size):
        shard_path = out_dir / f"{shard_idx:05d}.arrow"
        ArrowWriter(compression=compression).write_to_file(
            iter(shard), path=str(shard_path)
        )
        shard_idx += 1
    return shard_idx


def write_one_subset(
    hf_name: str,
    prefix: str,
    outdir: Path,
    n_total: int,
    n_val: int,
    compression: str | None = None,
    shard_size: int = 100_000,  # tune this to your RAM; 50k–250k are common
):
    # --- streaming load: avoids downloading the whole split ---
    ds_stream = datasets.load_dataset(
        "autogluon/chronos_datasets", hf_name, split="train", streaming=True
    ).with_format("numpy")

    it = iter(ds_stream)
    first = next(it)  # peek to infer fields
    series_fields = _series_fields_from_first(first)

    n_train = n_total - n_val

    # reconstruct iterator including the peeked first example
    full_iter = chain([first], it)

    # Create train/val expanded generators
    train_iter = _expand_examples(islice(full_iter, n_train), series_fields)
    val_iter = _expand_examples(islice(full_iter, n_val), series_fields)

    # Write in shards so Arrow doesn't try to hold everything at once
    train_dir = outdir / "train" / f"{prefix}"
    val_dir = outdir / "val" / f"{prefix}"

    n_train_shards = _write_sharded(train_iter, train_dir, compression, shard_size)
    n_val_shards = _write_sharded(val_iter, val_dir, compression, shard_size)

    print(
        f"Wrote {n_train_shards} train shard(s) and {n_val_shards} val shard(s) "
        f"for {hf_name} (prefix='{prefix}')."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total", type=int, default=100_000)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--compression", type=str, default="lz4")
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--shard_size", type=int, default=100_000)
    args = parser.parse_args()

    compression = None if args.compression.lower() == "none" else args.compression

    n_val = args.n_val
    if args.n_val is None:
        n_val = int(round(args.n_total * args.val_fraction))
    print(f"Using n_val={n_val}")

    OUTDIR = Path(args.outdir)
    (OUTDIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "val").mkdir(parents=True, exist_ok=True)

    write_one_subset(
        "training_corpus_tsmixup_10m",
        "tsmixup",
        OUTDIR,
        args.n_total,
        n_val,
        compression,
        shard_size=args.shard_size,
    )
    write_one_subset(
        "training_corpus_kernel_synth_1m",
        "kernelsynth",
        OUTDIR,
        args.n_total,
        n_val,
        compression,
        shard_size=args.shard_size,
    )
