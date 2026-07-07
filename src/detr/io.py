from __future__ import annotations

import json
import pprint
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

import detr
import detr.parameters as parameters
from detr._types import Model, ModelSubType, ModelType, ModelMeta
from detr import model

_MODEL_TYPES = "|".join(t.value for t in ModelType)
_MODEL_SUBTYPES = "|".join(s.value for s in ModelSubType)
_NEW_NAMING_RE = re.compile(
    rf"^({_MODEL_TYPES})_({_MODEL_SUBTYPES})_\d{{2}}$"
)


def is_legacy_model(file: Path) -> bool:
    """Return True if *file* does not follow the ``<ModelType>_<ModelSubType>_xx`` naming scheme."""
    stem = Path(file).stem
    return _NEW_NAMING_RE.match(stem) is None


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

    meta_data = model_data["meta"]
    meta = ModelMeta(
        categories=categories,
        model_type=ModelType(meta_data["model_type"]),
        subtype=ModelSubType(meta_data["subtype"]),
        train_params=parameters.Train(**meta_data["train_params"]),
        transforms=meta_data["transforms"],
        train_info=meta_data["train_info"],
        settings=meta_data["settings"],
        source=meta_data["source"],
    )

    ai_model = model.factory(meta)

    _load_state_dict(ai_model, model_data["state_dict"])

    return ai_model


def load_model_legacy(
    file: Path,
    device: str,
    categories: list[str],
    *,
    model_params: parameters.Model = parameters.Model(),
    loss_params: parameters.Loss = parameters.Loss(),
    train_params: parameters.Train = parameters.Train(),
) -> Model:
    """Reconstruct a ``Model`` from a legacy ``.pth`` checkpoint.

    Legacy models just contain the model state dict, so we need to
    reconstruct the model from the state dict and the provided parameters.
    ----------
    file:
        Path to the legacy ``.pth`` file.
    device:
        Device to map tensors to when loading.
    categories:
        Category names for the dataset (used to build the model).
    """
    logger.info(f"Loading legacy model from {file}")
    model_data = torch.load(file, map_location=device, weights_only=False)
    logger.debug(f"Model structure:\n{pprint.pformat(model_data, depth=3, compact=True)}")

    # Extract metadata
    if "-r50" in file.stem and not "-dc" in file.stem:
        subtype = ModelSubType.RESNET50
    elif "-r50" in file.stem and "-dc" in file.stem:
        subtype = ModelSubType.RESNET50_DC5
    elif "-r101" in file.stem and not "-dc" in file.stem:
        subtype = ModelSubType.RESNET101
    elif "-r101" in file.stem and "-dc" in file.stem:
        subtype = ModelSubType.RESNET101_DC5
    else:
        raise ValueError(f"Unknown model type: {file}")

    metadata = ModelMeta(
        categories=categories,
        dataset="coco",
        model_type=ModelType.DETR_BBOX,
        subtype=subtype,
        source="legacy",
        transforms=detr.transforms.default(),
        train_info={"logs": []},
        train_params=train_params,
        settings={
            "model_params": asdict(model_params),
            "loss_params": asdict(loss_params),
        },
    )

    ai_model = model.factory(metadata)

    state_dict = model_data.get("model")
    if state_dict is None:
        raise KeyError(f"Checkpoint {file} has no 'model' key")

    _load_state_dict(ai_model, state_dict)

    return ai_model


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

def _load_state_dict(model: Model, state_dict: dict) -> None:
    """Load *state_dict* into *model.ai*, skipping shape-mismatched keys."""
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
