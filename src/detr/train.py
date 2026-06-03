from __future__ import annotations

import copy
import datetime
import math
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from loguru import logger
from tqdm import tqdm

from detr import coco
from detr.model import Bundle
from detr.parameters import Augmentation, Train
from detr.transforms import RandomResize, RandomSizeCrop


def augmentation_transforms(params: Augmentation | None = None) -> list:
    """Return the geometric augmentation transforms parametrised by *params*.

    The returned list is meant to be appended to a base pipeline ending in
    ``Normalize``. :class:`~detr.transforms.FinalizeTargets` is added automatically
    downstream by :func:`detr.aux.load_dataset`.
    """
    if params is None:
        params = Augmentation()
    return [
        v2.RandomHorizontalFlip(p=params.hflip_prob),
        v2.RandomChoice(
            [
                RandomResize(params.scales, max_size=params.max_size),
                v2.Compose(
                    [
                        RandomResize(params.pre_crop_scales),
                        RandomSizeCrop(params.crop_min_size, params.crop_max_size),
                        RandomResize(params.scales, max_size=params.max_size),
                    ]
                ),
            ],
            p=[1.0 - params.crop_branch_prob, params.crop_branch_prob],
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

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

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

    return {key: value.item() for key, value in loss_dict_scaled.items()}


def run(
    bundle: Bundle,
    train_loader: Iterable,
    val_loader: Iterable,
    device: str,
    params: Train | None = None,
    output_dir: Path | None = None,
    v_file: Path | None = None,
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

    # Work on a copy so the caller's bundle is not mutated
    new_model = copy.copy(bundle)
    new_model.logs = list(bundle.logs)

    param_dicts = [
        {
            "params": [
                p
                for n, p in new_model.ai_model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in new_model.ai_model.named_parameters()
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
    for epoch in tqdm(range(params.epochs), desc="Epochs"):
        train_stats = train_one_epoch(
            new_model.ai_model,
            new_model.criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            params.clip_max_norm,
        )
        lr_scheduler.step()

        if v_file is not None:
            outputs = coco.inference(new_model, val_loader, device)
            val_stats = coco.run_eval(v_file, outputs)
        else:
            val_stats = {}

        log_entry = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        new_model.logs.append(log_entry)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training time {elapsed}")

    return new_model
