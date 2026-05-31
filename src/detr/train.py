from __future__ import annotations

import copy
import datetime
import json
import math
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from tqdm import tqdm

from detr.evaluate import evaluate
from detr.logger import MetricLogger, SmoothedValue
from detr.model import Bundle
from detr.parameters import Augmentation, Train
from detr.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResize,
    RandomSelect,
    RandomSizeCrop,
)


def augmentation_transforms(params: Augmentation | None = None) -> list:
    """Return the geometric augmentation transforms parametrised by *params*.

    The returned list is meant to be appended to a base
    ``[ToTensor, NormalizeImage]`` pipeline.  Box format conversion is added
    automatically downstream by :func:`detr.aux.load_dataset`.
    """
    if params is None:
        params = Augmentation()
    return [
        RandomHorizontalFlip(p=params.hflip_prob),
        RandomSelect(
            RandomResize(params.scales, max_size=params.max_size),
            Compose(
                [
                    RandomResize(params.pre_crop_scales),
                    RandomSizeCrop(params.crop_min_size, params.crop_max_size),
                    RandomResize(params.scales, max_size=params.max_size),
                ]
            ),
            p=1.0 - params.crop_branch_prob,
        ),
    ]


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        loss_dict_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
        }
        loss_value = sum(loss_dict_scaled.values()).item()

        if not math.isfinite(loss_value):
            tqdm.write(f"Loss is {loss_value}, stopping training")
            tqdm.write(str(loss_dict))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(class_error=loss_dict["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    tqdm.write("Averaged stats: " + str(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def run(
    bundle: Bundle,
    train_loader: Iterable,
    val_loader: Iterable,
    params: Train | None = None,
    output_dir: Path | str | None = None,
    base_ds=None,
) -> Bundle:
    """Train *bundle* for the given number of epochs and return an updated bundle.

    The original bundle is not mutated — a shallow copy is returned with the
    newly trained weights and accumulated logs.

    Parameters
    ----------
    bundle:
        A :class:`~detr.model.Bundle` produced by ``Bundle.build`` or
        ``Bundle.load_from_file``.
    train_loader:
        DataLoader yielding ``(NestedTensor, targets)`` batches for training.
    val_loader:
        DataLoader yielding ``(NestedTensor, targets)`` batches for validation.
    params:
        Training hyperparameters.  Defaults to ``Train()`` (sensible
        defaults for fine-tuning).
    output_dir:
        Directory where checkpoints and logs are written.  Pass ``None`` to
        disable persistence.
    base_ds:
        COCO API object for the validation set, used by the evaluator.
        Pass ``None`` to skip COCO evaluation (loss stats are still logged).

    Returns
    -------
    Bundle
        A new ``Bundle`` with updated weights and ``logs`` appended.
    """
    if params is None:
        params = Train()

    output_dir = Path(output_dir) if output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if base_ds is None:
        dataset = getattr(val_loader, "dataset", None)
        coco_api = getattr(dataset, "coco_api", None)
        if callable(coco_api):
            base_ds = coco_api()

    device = torch.device(bundle.device)

    # Work on a copy so the caller's bundle is not mutated
    result = copy.copy(bundle)
    result.logs = list(bundle.logs)

    param_dicts = [
        {
            "params": [
                p
                for n, p in result.ai_model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in result.ai_model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": params.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=params.lr, weight_decay=params.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_drop)

    start_time = time.time()
    for epoch in range(params.epochs):
        train_stats = train_one_epoch(
            result.ai_model,
            result.criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            params.clip_max_norm,
        )
        lr_scheduler.step()

        test_stats = evaluate(
            result.ai_model,
            result.criterion,
            result.postprocessors,
            val_loader,
            base_ds,
            device,
        )

        log_entry = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in test_stats.items()},
        }
        result.logs.append(log_entry)

        if output_dir is not None:
            result.export(output_dir / "checkpoint")
            if (epoch + 1) % params.lr_drop == 0 or (epoch + 1) % 100 == 0:
                result.export(output_dir / f"checkpoint{epoch:04}")
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")

    return result
