# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR transforms / augmentation.

Thin orchestration over ``torchvision.transforms.v2``; the only DETR-specific
step is :class:`FinalizeTargets`, which drops degenerate boxes, refreshes
``target['size']`` / ``target['area']`` from the post-augmentation state and
converts boxes to normalised cxcywh in ``[0, 1]``.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from torchvision.ops import box_convert

from detr.parameters import Augmentation


class RandomResize(v2.Transform):
    """Resize the inputs to a random short side picked uniformly from *sizes*.

    Delegates to :class:`v2.Resize`, whose ``size=int`` + ``max_size`` knobs
    implement aspect-preserving short-side scaling natively.
    """

    def __init__(self, sizes: Sequence[int], max_size: int | None = None):
        super().__init__()
        self.sizes = list(sizes)
        self.max_size = max_size

    def forward(self, *inputs):
        size = random.choice(self.sizes)
        return v2.Resize(size, max_size=self.max_size, antialias=True)(*inputs)


class RandomSizeCrop(v2.Transform):
    """Crop a random box with height / width uniformly drawn from
    ``[min_size, max_size]`` (capped by the input dimensions)."""

    def __init__(self, min_size: int, max_size: int):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, *inputs):
        h0, w0 = v2.functional.get_size(inputs[0])
        h = random.randint(self.min_size, min(h0, self.max_size))
        w = random.randint(self.min_size, min(w0, self.max_size))
        return v2.RandomCrop([h, w])(*inputs)


class FinalizeTargets:
    """Last step of the DETR pipeline.

    * Drops degenerate boxes (and their sibling per-instance fields).
    * Recomputes ``target['area']`` from the surviving pixel-space boxes.
    * Converts boxes to normalised cxcywh in ``[0, 1]``.
    * Stores the current image size as ``target['size']``.
    """

    _PER_INSTANCE_KEYS = ("boxes", "labels", "area", "iscrowd", "masks", "keypoints")

    def __call__(self, image, target):
        target = dict(target)
        boxes = target.get("boxes")
        if isinstance(boxes, tv_tensors.BoundingBoxes):
            xyxy = boxes.as_subclass(torch.Tensor).float()
            keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
            n = xyxy.shape[0]
            for key in self._PER_INSTANCE_KEYS:
                val = target.get(key)
                if torch.is_tensor(val) and val.shape and val.shape[0] == n:
                    target[key] = val[keep]
            xyxy = xyxy[keep]
            h, w = boxes.canvas_size
            target["area"] = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            cxcywh = box_convert(xyxy, "xyxy", "cxcywh")
            target["boxes"] = cxcywh / torch.tensor([w, h, w, h], dtype=torch.float32)
        masks = target.get("masks")
        if isinstance(masks, tv_tensors.Mask):
            target["masks"] = masks.as_subclass(torch.Tensor)
        target["size"] = torch.as_tensor(image.shape[-2:])
        return image, target


def _make_tail(params: Augmentation) -> list[v2.Transform]:
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=params.normalize_mean, std=params.normalize_std),
        FinalizeTargets(),
    ]


def augmentation(params: Augmentation | None = None) -> v2.Compose:
    """Build the training COCO pipeline."""
    params = params or Augmentation()
    return v2.Compose(
        [
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
                p=[1 - params.crop_branch_prob, params.crop_branch_prob],
            ),
            *_make_tail(params),
        ]
    )


def default(params: Augmentation | None = None) -> v2.Compose:
    """Transformation for inference/validation/test."""
    params = params or Augmentation()
    return v2.Compose(
        [
            v2.Resize(max(params.scales), max_size=params.max_size, antialias=True),
            *_make_tail(params),
        ]
    )
