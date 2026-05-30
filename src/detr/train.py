from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from detr.engine import evaluate, train_one_epoch
from detr.model import Bundle


@dataclasses.dataclass
class Parameters:
    """Hyperparameters for a :func:`run` call.

    Mirrors ``parameters.Train`` but is intentionally kept separate so that
    callers can tweak training behaviour without touching the bundle's stored
    ``train_params``.
    """

    epochs: int = 2
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    lr_drop: int = 40
    clip_max_norm: float = 0.1
    start_epoch: int = 0
    output_dir: Path | None = None


def run(
    bundle: Bundle,
    train_loader: Iterable,
    val_loader: Iterable,
    params: Parameters | None = None,
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
        Training hyperparameters.  Defaults to ``Parameters()`` (sensible
        defaults for fine-tuning).
    base_ds:
        COCO API object for the validation set, used by the evaluator.
        Pass ``None`` to skip COCO evaluation (loss stats are still logged).

    Returns
    -------
    Bundle
        A new ``Bundle`` with updated weights and ``logs`` appended.
    """
    if params is None:
        params = Parameters()

    output_dir = Path(params.output_dir) if params.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

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
    for epoch in range(params.start_epoch, params.epochs):
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

        test_stats, _ = evaluate(
            result.ai_model,
            result.criterion,
            result.postprocessors,
            val_loader,
            base_ds,
            device,
            str(output_dir) if output_dir else "",
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
