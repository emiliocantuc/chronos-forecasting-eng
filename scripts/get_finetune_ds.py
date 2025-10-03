# Downloads the datasets we use for finetuning
# We use a fraction of the official datasets used for Chronos' *training*
# and hold out a "validation" set to eval our tuning.
# (not a true val set since the model saw it during training, we'll evaluate
# on the official test set -- see evaluate.py)

from datasets import load_dataset
from pathlib import Path
from gluonts.dataset.arrow import ArrowWriter
import json
import os
from typing import Dict, Iterable, List

DATASET = "autogluon/chronos_datasets"
NAMES = ["training_corpus_kernel_synth_1m", "training_corpus_tsmixup_10m"]

OUTDIR = "outdir_arrow"
N_PER_SUBSET = 10_000  # how many VALID series to take from each subset
VAL_FRACTION = 0.1
FREQ = "H"

MIN_TARGET_LEN = 64 + 64  # strictly greater than this (i.e., > 128)


def ensure_dirs(base: str):
    Path(base, "train").mkdir(parents=True, exist_ok=True)
    Path(base, "val").mkdir(parents=True, exist_ok=True)


def valid_example(ex) -> bool:
    tgt = ex.get("target")
    return isinstance(tgt, (list, tuple)) and len(tgt) > MIN_TARGET_LEN


def to_record(ex) -> Dict:
    # HF entry has 'timestamp' (list) and 'target' (list). No 'start'.
    return {
        "start": str(ex["timestamp"][0]).replace(" ", "T"),
        "target": [float(v) for v in ex["target"]],
        # Optionally keep the id:
        # "item_id": ex.get("id"),
    }


def take_valid(
    iterable: Iterable[Dict], n: int, counters: Dict[str, int]
) -> List[Dict]:
    """Consume the (streaming) iterable, returning up to n VALID records.
    Mutates counters['skipped'] and counters['consumed']."""
    out: List[Dict] = []
    if n <= 0:
        return out
    for ex in iterable:
        counters["consumed"] += 1
        if not valid_example(ex):
            counters["skipped"] += 1
            continue
        out.append(to_record(ex))
        if len(out) >= n:
            break
    return out


def main():
    ensure_dirs(OUTDIR)

    meta = {
        "freq": FREQ,
        "min_target_len": MIN_TARGET_LEN,
        "subsets": {},
    }

    for name in NAMES:
        print(f"Processing subset: {name}")
        ds_stream = load_dataset(DATASET, name=name, split="train", streaming=True)

        n_val = int(round(N_PER_SUBSET * VAL_FRACTION))
        n_train = max(N_PER_SUBSET - n_val, 0)

        train_path = os.path.join(OUTDIR, "train", f"{name}.arrow")
        val_path = os.path.join(OUTDIR, "val", f"{name}.arrow")

        counters = {"skipped": 0, "consumed": 0}

        # 1) Gather TRAIN valid examples
        train_records = take_valid(ds_stream, n_train, counters)

        # 2) Gather VAL from the remainder of the SAME stream
        val_records = take_valid(ds_stream, n_val, counters)

        # Write .arrow files
        ArrowWriter(compression="lz4").write_to_file(train_records, path=train_path)
        ArrowWriter(compression="lz4").write_to_file(val_records, path=val_path)

        meta["subsets"][name] = {
            "train_file": os.path.relpath(train_path, OUTDIR),
            "val_file": os.path.relpath(val_path, OUTDIR),
            "requested_train": n_train,
            "requested_val": n_val,
            "written_train": len(train_records),
            "written_val": len(val_records),
            "skipped_due_to_short_target": counters["skipped"],
            "consumed_total": counters["consumed"],
        }

        print(
            f"  wrote train: {len(train_records)}  val: {len(val_records)}  "
            f"skipped(short): {counters['skipped']}  consumed: {counters['consumed']}"
        )

    # Save overall metadata
    meta_path = os.path.join(OUTDIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Done. Arrow files under '{OUTDIR}/train' and '{OUTDIR}/val'. Meta: {meta_path}"
    )


if __name__ == "__main__":
    main()
