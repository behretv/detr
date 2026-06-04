"""Auxiliary helpers shared by the public training / inference scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from detr.dataset import CocoDetection
from detr.misc import collate_fn
from detr.transforms import make_coco_transforms

DATA_ROOT: Path = Path(os.environ.get("DETR_DATA_ROOT", "/mnt/data"))
"""Root directory for datasets and model artefacts.

Override via the ``DETR_DATA_ROOT`` environment variable.
"""


def load_dataset(
    ann_file: Path,
    transforms: Sequence | v2.Compose,
    shuffle: bool = True,
    batch_size: int = 2,
    num_workers: int = 2,
    return_masks: bool = False,
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
    transforms = make_coco_transforms(ann_file.parent.name.split(".")[0])

    dataset = CocoDetection(
        ann_file.parent,
        ann_file,
        transforms=transforms,
        return_masks=return_masks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


