import datasets
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from pathlib import Path
from itertools import islice, chain
import argparse


def _expand_examples(example_iter, series_fields):
    """GluonTS-style generator from a stream of HF examples."""
    for ex in example_iter:
        start = np.datetime64(ex["timestamp"][0], "s")
        for field in series_fields:
            yield {"start": start, "target": np.asarray(ex[field])}


def _series_fields_from_first(example):
    # mimic the original "all sequence fields except timestamp"
    fields = []
    for k, v in example.items():
        if k == "timestamp":
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            fields.append(k)
    return fields


def write_one_subset(
    hf_name: str,
    prefix: str,
    outdir: Path,
    n_total: int,
    val_fraction: float,
    compression: str,
):
    # --- streaming load: avoids downloading the whole split ---
    ds_stream = datasets.load_dataset(
        "autogluon/chronos_datasets",
        hf_name,
        split="train",
        streaming=True,
    ).with_format("numpy")

    it = iter(ds_stream)
    first = next(it)  # peek to infer fields
    series_fields = _series_fields_from_first(first)

    n_val = int(round(n_total * val_fraction))
    n_train = n_total - n_val

    # reconstruct full iterator including the peeked first example
    full_iter = chain([first], it)

    # train = first n_train rows (no shuffle), val = next n_val rows
    train_iter = _expand_examples(islice(full_iter, n_train), series_fields)
    val_iter = _expand_examples(islice(full_iter, n_val), series_fields)

    ArrowWriter(compression=compression).write_to_file(
        train_iter, path=str(outdir / "train" / f"{prefix}.arrow")
    )
    ArrowWriter(compression=compression).write_to_file(
        val_iter, path=str(outdir / "val" / f"{prefix}.arrow")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_total", type=int, default=100_000)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--compression", type=str, default="lz4")
    parser.add_argument("--outdir", type=str, default="./data")
    args = parser.parse_args()

    if args.compression.lower() == "none":
        args.compression = None

    N_TOTAL = args.n_total
    VAL_FRACTION = args.val_fraction
    COMPRESSION = args.compression

    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.joinpath("train").mkdir(parents=True, exist_ok=True)
    OUTDIR.joinpath("val").mkdir(parents=True, exist_ok=True)

    write_one_subset(
        "training_corpus_tsmixup_10m",
        "tsmixup-data",
        OUTDIR,
        N_TOTAL,
        VAL_FRACTION,
        COMPRESSION,
    )
    write_one_subset(
        "training_corpus_kernel_synth_1m",
        "kernelsynth-data",
        OUTDIR,
        N_TOTAL,
        VAL_FRACTION,
        COMPRESSION,
    )
