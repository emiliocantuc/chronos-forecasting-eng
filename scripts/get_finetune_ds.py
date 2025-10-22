import datasets
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from pathlib import Path
from itertools import islice, chain

N_TOTAL = 100_000
VAL_FRACTION = 0.05
COMPRESSION = None  # "lz4"  # set to None for peak write speed
OUTDIR = Path("./data")
OUTDIR.joinpath("train").mkdir(parents=True, exist_ok=True)
OUTDIR.joinpath("val").mkdir(parents=True, exist_ok=True)


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


def write_one_subset(hf_name: str, prefix: str):
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

    n_val = int(round(N_TOTAL * VAL_FRACTION))
    n_train = N_TOTAL - n_val

    # reconstruct full iterator including the peeked first example
    full_iter = chain([first], it)

    # train = first n_train rows (no shuffle), val = next n_val rows
    train_iter = _expand_examples(islice(full_iter, n_train), series_fields)
    val_iter = _expand_examples(islice(full_iter, n_val), series_fields)

    ArrowWriter(compression=COMPRESSION).write_to_file(
        train_iter, path=str(OUTDIR / "train" / f"{prefix}.arrow")
    )
    ArrowWriter(compression=COMPRESSION).write_to_file(
        val_iter, path=str(OUTDIR / "val" / f"{prefix}.arrow")
    )


if __name__ == "__main__":
    # TSMixup
    write_one_subset("training_corpus_tsmixup_10m", "tsmixup-data")
    # KernelSynth
    write_one_subset("training_corpus_kernel_synth_1m", "kernelsynth-data")
