"""Auxiliary helpers shared by the public training / inference scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from detr.dataset import CocoDetection
from detr.misc import collate_fn
from detr.transforms import FinalizeTargets

DATA_ROOT: Path = Path(os.environ.get("DETR_DATA_ROOT", "/mnt/data"))
"""Root directory for datasets and model artefacts.

Override via the ``DETR_DATA_ROOT`` environment variable.
"""


def load_dataset(
    ann_file: Path | str,
    transforms: Sequence | v2.Compose,
    shuffle: bool = True,
    batch_size: int = 2,
    num_workers: int = 2,
    return_masks: bool = False,
    img_folder: Path | str | None = None,
) -> DataLoader:
    """Build a COCO ``DataLoader`` from an annotation file.

    Parameters
    ----------
    ann_file:
        Path to a COCO-format ``*.json`` file. Image files are expected to live
        in the same directory unless ``img_folder`` is provided.
    transforms:
        Either a :class:`v2.Compose` or a list of transforms.
        :class:`~detr.transforms.FinalizeTargets` is appended automatically if
        not already present, so callers can freely concatenate base and
        augmentation transforms in either order.
    shuffle:
        Whether the loader should reshuffle each epoch.
    batch_size, num_workers, return_masks:
        Standard DataLoader / dataset knobs.
    img_folder:
        Override for the image directory; defaults to ``ann_file.parent``.
    """
    ann_file = Path(ann_file)
    img_folder = Path(img_folder) if img_folder is not None else ann_file.parent

    if isinstance(transforms, v2.Compose):
        items = list(transforms.transforms)
    elif isinstance(transforms, Iterable):
        items = list(transforms)
    else:
        raise TypeError(
            f"transforms must be a v2.Compose or iterable, got {type(transforms).__name__}"
        )

    if not any(isinstance(t, FinalizeTargets) for t in items):
        items.append(FinalizeTargets())

    dataset = CocoDetection(
        img_folder, ann_file, transforms=v2.Compose(items), return_masks=return_masks
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

def to_device(images: list[torch.Tensor], targets: list[dict], device: str):
    """Move images and targets to device."""
    keys = ["boxes", "labels"]
    images = [img.to(device) for img in images]
    targets = [{k: t[k].to(device) for k in keys} for t in targets]
    return images, targets