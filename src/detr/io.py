from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

import detr
import detr.parameters as parameters
from detr._types import Model
from detr.model import factory


def load_categories(ann_file: Path) -> list[str]:
    """Extract category names from a COCO annotation file."""
    with open(ann_file, "r") as f:
        data = json.load(f)
    return [cat["name"] for cat in data["categories"]]


def load_checkpoint(file: Path, device: str) -> dict:
    """Load a raw torch checkpoint dict from *file*."""
    logger.info(f"Loading checkpoint from {file}")
    return torch.load(file, map_location=device, weights_only=False)


def save_model(model: Model, file: Path) -> None:
    """Save weights + architecture params to *file*.pth and logs to *file*.csv.

    Enough information is stored to fully reconstruct the model via
    ``load_model()``.
    """
    if file.exists():
        raise FileExistsError(f"File {file} already exists!")

    logger.info(f"Exporting model to {file}")
    model.meta.train_info["params"] = asdict(model.meta.train_params)
    torch.save(model.serialize(), Path(file).with_suffix(".pth"))

    pd.DataFrame(model.meta.train_info["logs"]).to_csv(
        Path(file).with_suffix(".csv"), index=False
    )


def load_model(
    file: Path,
    device: str,
    categories: list[str],
) -> Model:
    """Reconstruct a ``Model`` from a ``.pth`` file written by ``save_model()``.

    Parameters
    ----------
    file:
        Path to the ``.pth`` file (the ``.pth`` suffix may be omitted).
    device:
        Override the device stored in the checkpoint. Defaults to the
        value recorded at export time, or auto-detects CUDA when absent.
    """
    logger.info(f"Loading model from {file}")
    model_data = torch.load(file, map_location=device, weights_only=False)

    meta_data = model_data.get("meta", {})
    model_params: parameters.Model = parameters.Model(
        **meta_data.get("model_params", model_data.get("model_params", {}))
    )
    loss_params: parameters.Loss = parameters.Loss(
        **meta_data.get("loss_params", model_data.get("loss_params", {}))
    )
    train_params: parameters.Train = parameters.Train(
        **meta_data.get("train_params", model_data.get("train_params", {}))
    )

    model = factory(model_params, loss_params, train_params, categories)

    state_dict = model_data.get("state_dict") or model_data.get("model")
    if state_dict is None:
        raise KeyError(f"Checkpoint {file} has neither 'state_dict' nor 'model' key")

    model_state = model.ai.state_dict()
    incompatible = {
        k for k, v in state_dict.items()
        if k in model_state and v.shape != model_state[k].shape
    }
    if incompatible:
        logger.warning(
            f"Skipping incompatible keys (shape mismatch): {incompatible}"
        )
    state_dict = {k: v for k, v in state_dict.items() if k not in incompatible}

    missing, unexpected = model.ai.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            f"load_state_dict (strict=False) — missing: {missing}, unexpected: {unexpected}"
        )

    model.meta.source = meta_data.get("source", str(file))
    model.meta.transforms = meta_data.get(
        "transforms", model_data.get("transforms", detr.transforms.default())
    )
    model.meta.categories = meta_data.get(
        "categories", model_data.get("cats", {})
    )
    model.meta.train_info = meta_data.get(
        "train_info", {"params": {}, "logs": []}
    )
    return model


def output_filename(dir_output: Path, name: str) -> Path:
    """Add a suffix with an index if the output folder already exists."""
    suffix = "pth"
    file_new = dir_output / f"{name}_00.{suffix}"
    for idx in range(1, 99, 1):
        if not file_new.exists():
            logger.info(f"Output file: {file_new}")
            return file_new
        file_new = dir_output / f"{name}_{idx:02d}.{suffix}"

    raise ValueError("Unable to find a non-existing output file!")
