# https://github.com/amazon-science/chronos-forecasting/discussions/162

import datasets
import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from tqdm.auto import tqdm


def hf_to_gluonts_univariate(hf_dataset: datasets.Dataset, batch_size: int = 1000):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples

    pbar = tqdm(total=dataset_length)
    for batch in hf_dataset.iter(batch_size=batch_size):
        batch = [dict(zip(batch, t)) for t in zip(*batch.values())]
        for hf_entry in batch:
            for field in series_fields:
                yield {
                    "start": np.datetime64(hf_entry["timestamp"][0], "s"),
                    "target": np.array(hf_entry[field]),
                }
        pbar.update(batch_size)
    pbar.close()


if __name__ == "__main__":
    # Increase this until saturation
    batch_size = 50_000

    # Load TSMixup data and convert it into GluonTS arrow format
    ds = datasets.load_dataset(
        "autogluon/chronos_datasets",
        "training_corpus_tsmixup_10m",
        split="train",
    )
    ds.set_format("numpy")
    ArrowWriter(compression="lz4").write_to_file(
        hf_to_gluonts_univariate(ds, batch_size=batch_size),
        path="data/tsmixup-data.arrow",
    )

    # Load KernelSynth data and convert it into GluonTS arrow format
    ds = datasets.load_dataset(
        "autogluon/chronos_datasets",
        "training_corpus_kernel_synth_1m",
        split="train",
    )
    ds.set_format("numpy")
    ArrowWriter(compression="lz4").write_to_file(
        hf_to_gluonts_univariate(ds, batch_size=batch_size),
        path="data/kernelsynth-data.arrow",
    )
