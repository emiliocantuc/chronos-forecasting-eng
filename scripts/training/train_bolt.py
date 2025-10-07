# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from copy import deepcopy
import logging
import itertools
from pathlib import Path
from functools import partial
import random
from typing import List, Iterator, Optional

import typer
from typer_config import use_yaml_config
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    T5Config,
    Trainer,
    TrainingArguments,
)
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
)

from chronos.chronos_bolt_eng import (
    ChronosBoltModelForForecasting,
    ChronosBoltWithEngressionModel,
)

from train import (
    ShuffleMixin,
    is_main_process,
    log_on_main,
    get_next_path,
    has_enough_observations,
    save_training_info,
)


app = typer.Typer(pretty_exceptions_enable=False)


class ChronosBoltDataset(IterableDataset, ShuffleMixin):
    """
    Yields fixed-length tensors for Chronos-Bolt:
      context:      (L_ctx,) float32 (left-padded with NaNs)
      mask:         (L_ctx,) float32 1.0 where observed, 0.0 else
      target:       (L_pred,) float32
      target_mask:  (L_pred,) float32 1.0 where observed, 0.0 else

    It uses the same GluonTS InstanceSplitter pipeline you had.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.0,  # optional synthetic drop on past_target
        min_past: Optional[int] = None,
        imputation_method: Optional[
            MissingValueImputation
        ] = None,  # unused (Bolt tolerates NaNs)
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()
        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")

        self.datasets = datasets
        self.probabilities = probabilities
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob
        self.min_past = min_past or prediction_length
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target
        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        return split_transform.apply(data, is_train=True)

    def create_test_data(self, data):
        return self._create_instance_splitter("test").apply(data, is_train=False)

    def create_validation_data(self, data):
        return self._create_instance_splitter("validation").apply(data, is_train=False)

    def to_hf_format(self, entry: dict) -> dict:
        # GluonTS fields from InstanceSplitter:
        # - past_target          (L_ctx,)
        # - past_is_pad          (L_ctx,) 1 if pad on the left, else 0
        # - future_target        (L_pred,)
        # They may contain NaNs for missing values.

        past = entry["past_target"].astype(self.np_dtype)
        fut = entry["future_target"].astype(self.np_dtype)

        # Observed masks: 1 where not NaN
        past_mask = (~np.isnan(past)).astype(self.np_dtype)
        fut_mask = (~np.isnan(fut)).astype(self.np_dtype)

        # Convert NaNs to zeros; Bolt will use masks to zero them before patching anyway.
        past = np.nan_to_num(past, nan=0.0)
        fut = np.nan_to_num(fut, nan=0.0)

        return {
            "context": torch.from_numpy(past.astype(np.float32)),
            "mask": torch.from_numpy(past_mask.astype(np.bool_)),
            "target": torch.from_numpy(fut.astype(np.float32)),
            "target_mask": torch.from_numpy(fut_mask.astype(np.bool_)),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(partial(self.preprocess_entry, mode=self.mode), ds)
            for ds in self.datasets
        ]

        if self.mode == "training":
            iterables = [self.create_training_data(ds) for ds in preprocessed_datasets]
        elif self.mode == "test":
            iterables = [self.create_test_data(ds) for ds in preprocessed_datasets]
        else:
            iterables = [
                self.create_validation_data(ds) for ds in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )
        probs = [p / sum(probs) for p in probs]

        iters = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iters)), p=probs)
                try:
                    yield self.to_hf_format(next(iters[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [p / sum(probs) for p in probs]
        else:
            for entry in itertools.chain(*iters):
                yield self.to_hf_format(entry)


def bolt_collate(batch):
    out = {k: torch.stack([ex[k] for ex in batch], dim=0) for k in batch[0].keys()}
    for k, v in out.items():
        if v.dtype.is_floating_point:  # leave bool masks alone
            out[k] = v.to(torch.float32)
    return out


# TODO make these args
def param_groups(model, lr_backbone, lr_head, wd=0.01):
    head, back = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n.startswith(("out_proj", "sample_head")):
            head.append(p)
            print(f"Head param: {n} {p.shape}")
        else:
            back.append(p)
    # assert len(head) > 0 and len(back) > 0
    return [
        {"params": back, "lr": lr_backbone, "weight_decay": wd},
        {"params": head, "lr": lr_head, "weight_decay": wd},
    ]


class BoltTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_m: int = None,
        eval_m: int = None,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-4,
        mc_dropout: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_m = train_m
        self.eval_m = eval_m
        self.mc_dropout = mc_dropout
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        assert train_m is None or isinstance(self.model, ChronosBoltWithEngressionModel)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                param_groups(
                    self.model,
                    lr_backbone=self.lr_backbone,
                    lr_head=self.lr_head,
                    wd=0.01,
                ),
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        forward_args = {
            "context": inputs["context"],
            "mask": inputs.get("mask", None),
            "target": inputs.get("target", None),
            "target_mask": inputs.get("target_mask", None),
        }
        if self.train_m is not None:
            forward_args["m"] = self.train_m

        model.train()
        out = model(**forward_args)
        loss = out.loss

        return (loss, out) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        prev_mode = model.training
        model.train(self.mc_dropout)

        forward_args = {
            "context": inputs["context"],
            "mask": inputs.get("mask", None),
            "target": inputs.get("target", None),
            "target_mask": inputs.get("target_mask", None),
        }
        if self.eval_m is not None:
            forward_args["m"] = self.eval_m

        out = model(**forward_args)

        model.train(prev_mode)

        loss = out.loss.detach().to("cpu") if out.loss is not None else None
        preds = out.quantile_preds.detach().to("cpu")

        labels = inputs.get("target")
        mask = inputs.get("target_mask")
        labels = labels.detach().to("cpu") if labels is not None else None
        mask = mask.detach().to("cpu") if mask is not None else None

        label_pack = {"labels": labels, "mask": mask} if labels is not None else None
        return loss, preds, label_pack

    def log(self, logs, *args, **kwargs) -> None:
        # Merge in the last loss terms, if present
        terms = getattr(self.model, "_last_terms", None)
        if terms:
            logs = dict(logs)  # copy to avoid mutating caller's dict
            logs.setdefault("loss_term1", float(terms["loss_term1"]))
            logs.setdefault("loss_term2", float(terms["loss_term2"]))
            logs.setdefault("loss_total", float(terms["loss_total"]))
            logs.setdefault("std_across_samples", float(terms["std_across_samples"]))
        super().log(logs, *args, **kwargs)


def load_random_bolt_model(
    base_t5_model_id: str = "google/t5-efficient-tiny",
    context_length: int = 512,
    prediction_length: int = 64,
    input_patch_size: int = 16,
    input_patch_stride: int = 16,
    quantiles: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    use_reg_token: bool = False,
    convert_to_eng: bool = True,
):
    # start from a small T5 config
    t5 = T5Config.from_pretrained(base_t5_model_id)
    # slightly smaller initializer is usually nicer for time series
    t5.initializer_factor = getattr(t5, "initializer_factor", 0.05)

    # inject the Chronos-Bolt runtime config
    t5.chronos_config = {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "input_patch_size": input_patch_size,
        "input_patch_stride": input_patch_stride,
        "quantiles": quantiles,
        "use_reg_token": use_reg_token,
    }

    model = ChronosBoltModelForForecasting(t5)
    if convert_to_eng:
        model = convert_bolt_to_engression(model, out_dir=None)

    # set decoder start token if not present (T5 uses pad_token_id by default)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    return model


def load_pretrained_bolt_model(
    base_model_id: str = "amazon/chronos-bolt-tiny",
    convert_to_eng: bool = True,
):
    model = ChronosBoltModelForForecasting.from_pretrained(base_model_id)
    if convert_to_eng:
        model = convert_bolt_to_engression(model, out_dir=None)
    return model


def convert_bolt_to_engression(
    base_model: ChronosBoltModelForForecasting, out_dir: str | None = None
):
    cfg = base_model.config

    # carry over chronos_config (context_length, prediction_length, patch sizes, etc.)
    assert hasattr(cfg, "chronos_config")

    cfg.architectures = ["ChronosBoltWithEngressionModel"]  # so HF knows which class
    cfg.chronos_pipeline_class = "ChronosBoltWithEngressionPipeline"
    eng = ChronosBoltWithEngressionModel(config=cfg)

    base_state = base_model.state_dict()
    del base_model  # free memory

    missing, unexpected = eng.load_state_dict(base_state, strict=False)
    # TODO
    # assert len(unexpected) == 0 and len(missing) == 1 and missing[0] == "o_proj"

    if out_dir is not None:
        eng.save_pretrained(out_dir)
    return eng


# Thanks GPT5
def make_mean_wql_compute_metrics(q_levels, metric_name: str = "mean_wql"):
    """
    Returns a HF compute_metrics that outputs a single scalar:
    the Mean Weighted Sum Quantile Loss (WQL), averaged over series & quantiles.

    Expects:
      eval_pred.predictions:
        - (N, Q, L) quantile forecasts with Q == len(q_levels), or
        - (N, M, L) samples (we take np.quantile over axis=1)
      eval_pred.label_ids: dict with
        - 'labels': (N, L)
        - 'mask'  : (N, L) optional {0,1}/bool
    """
    q_levels = np.asarray(q_levels, dtype=np.float64)

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        pack = eval_pred.label_ids

        y = pack["labels"].astype(np.float64)  # (N, L)
        m = pack.get("mask", None)
        if m is None:
            m = np.ones_like(y, dtype=bool)
        else:
            m = m.astype(bool)

        # Ensure (N, Q, L)
        if preds.ndim == 3 and preds.shape[1] == len(q_levels):
            preds_q = preds.astype(np.float64)
        elif preds.ndim == 3:  # samples (N, M, L)
            preds_q = np.quantile(preds, q=q_levels, axis=1)  # (Q, N, L)
            preds_q = np.transpose(preds_q, (1, 0, 2))  # (N, Q, L)
        else:
            raise ValueError(
                f"predictions shape {preds.shape} must be (N,Q,L) or (N,M,L)."
            )

        # Denominator per series: sum_t |y_t| over observed steps
        denom = (np.abs(y) * m).sum(axis=1) + 1e-8  # (N,)

        # Pinball loss per (series, quantile, time)
        u = y[:, None, :] - preds_q  # (N, Q, L)
        u = u * m[:, None, :]  # mask
        pinball = np.maximum(
            q_levels[None, :, None] * u, (q_levels[None, :, None] - 1.0) * u
        )  # (N, Q, L)

        # Weighted Sum Quantile Loss per (series, quantile)
        wsql_bq = pinball.sum(axis=2) / denom[:, None]  # (N, Q)

        # Mean across series & quantiles
        mean_wql = float(wsql_bq.mean())
        return {metric_name: mean_wql}

    return compute_metrics


@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    eval_data_paths: str,
    training_probability: Optional[str] = None,
    eval_probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 100,
    eval_steps: int = 100,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 1,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 1,  # TODO was 2
    # ---- Bolt bits ----
    model_id: str = "google/t5-efficient-tiny",
    input_patch_size: int = 16,
    input_patch_stride: int = 16,
    quantiles: str = "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]",
    use_reg_token: bool = False,
    random_init: bool = False,
    # Engression bits
    train_m: int = 2,
    eval_m: int = 64,
    engression: bool = True,
    lr_backbone: float = 1e-5,
    lr_head: float = 1e-4,
    # --------------------
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    seed: Optional[int] = None,
):
    if seed is None:
        seed = random.randint(0, 2**32)

    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    eval_data_paths = ast.literal_eval(eval_data_paths)
    assert isinstance(training_data_paths, list)
    assert isinstance(eval_data_paths, list)

    def _parse_probs(p, n):
        if p is None:
            return [1.0 / n] * n
        probs = ast.literal_eval(p)
        assert isinstance(probs, list) and len(probs) == n
        s = sum(probs)
        assert s > 0, "probabilities must sum to > 0"
        return [x / s for x in probs]

    training_probability = _parse_probs(training_probability, len(training_data_paths))
    eval_probability = _parse_probs(eval_probability, len(eval_data_paths))

    if isinstance(quantiles, str):
        quantiles = ast.literal_eval(quantiles)
    assert isinstance(quantiles, list) and len(quantiles) > 0

    if dataloader_num_workers > len(training_data_paths):
        log_on_main(
            f"Setting the number of data loader workers to {len(training_data_paths)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(training_data_paths)

    output_dir = get_next_path("run", base_dir=Path(output_dir), file_type="")

    log_on_main("Loading/Filtering datasets", logger)
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(p), freq="h"),
        )
        for p in training_data_paths
    ]

    val_dataset = ChronosBoltDataset(
        datasets=[
            Filter(
                partial(
                    has_enough_observations,
                    min_length=min_past + prediction_length,
                    max_missing_prop=max_missing_prop,
                ),
                FileDataset(path=Path(p), freq="h"),
            )
            for p in eval_data_paths
        ],
        probabilities=eval_probability,
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="validation",
    )

    # ---- Load Chronos-Bolt model ----
    log_on_main("Initializing Chronos-Bolt", logger)
    if "bolt" in model_id and not random_init:
        log_on_main(f"Loading pretrained Chronos-Bolt: {model_id}", logger)
        model = load_pretrained_bolt_model(
            base_model_id=model_id,
            convert_to_eng=engression,
        )

        log_on_main(
            f"Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
            logger,
        )
    else:
        log_on_main(f"Loading random-init Chronos-Bolt based on: {model_id}", logger)
        model = load_random_bolt_model(
            base_t5_model_id=model_id,
            context_length=context_length,
            prediction_length=prediction_length,
            input_patch_size=input_patch_size,
            input_patch_stride=input_patch_stride,
            quantiles=quantiles,
            use_reg_token=use_reg_token,
        )

    # Save runtime config into HF model.config so checkpoints are self-describing

    cfg = model.config.chronos_config  # already from the checkpoint
    cfg.update(
        {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "input_patch_size": input_patch_size,
            "input_patch_stride": input_patch_stride,
            "quantiles": quantiles,
            # Leave cfg["use_reg_token"] as-is!
        }
    )
    model.config.chronos_config = cfg

    # ---- Dataset ----
    shuffled_train_dataset = ChronosBoltDataset(
        datasets=train_datasets,
        probabilities=training_probability,
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        # learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,  # TODO ?
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,  # keep our (context,mask,target,...) dict intact
    )

    # ---- Trainer ----
    quantiles = model.config.chronos_config["quantiles"]  # e.g. [0.1,...,0.9]
    compute_metrics = make_mean_wql_compute_metrics(quantiles)

    trainer = BoltTrainer(
        model=model,
        train_m=train_m if engression else None,
        eval_m=eval_m if engression else None,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        mc_dropout=False,  # TODO
        args=training_args,
        train_dataset=shuffled_train_dataset,
        eval_dataset=val_dataset,
        data_collator=bolt_collate,
        compute_metrics=compute_metrics,
    )

    log_on_main("Training", logger)
    trainer.train()

    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final", training_config=raw_training_config
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
