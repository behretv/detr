"""Auxiliary helpers shared by the public training / inference scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from detr.dataset import CocoDetection
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


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (list[list[int]]) -> list[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Tensor | None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: list[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)

