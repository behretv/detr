# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.

Built on top of :mod:`torchvision.transforms.v2`, which natively syncs
bounding boxes and masks with image-level geometric transforms.  The public
classes preserve the legacy ``(image, target_dict) -> (image, target_dict)``
calling convention used throughout the project.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision import tv_tensors

from detr.misc import box_xyxy_to_cxcywh
from detr.parameters import Augmentation


# ---------------------------------------------------------------------------
# tv_tensor helpers
# ---------------------------------------------------------------------------


def _canvas_size(image) -> tuple[int, int]:
    """Return (H, W) for a PIL image or CHW tensor."""
    if isinstance(image, Image.Image):
        w, h = image.size
        return h, w
    return int(image.shape[-2]), int(image.shape[-1])


def _wrap_target(target: dict, canvas_size: tuple[int, int]) -> dict:
    """Wrap ``boxes`` / ``masks`` as v2 tv_tensors (no-op if already wrapped)."""
    target = dict(target)
    boxes = target.get("boxes")
    if boxes is not None and not isinstance(boxes, tv_tensors.BoundingBoxes):
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=canvas_size
        )
    masks = target.get("masks")
    if masks is not None and not isinstance(masks, tv_tensors.Mask):
        target["masks"] = tv_tensors.Mask(masks)
    return target


def _filter_instances(target: dict, keep: torch.Tensor) -> None:
    """Drop instances for which *keep* is False across all per-instance fields."""
    n = keep.shape[0]
    for field in ("boxes", "labels", "area", "iscrowd", "masks", "keypoints"):
        val = target.get(field)
        if torch.is_tensor(val) and val.shape and val.shape[0] == n:
            target[field] = val[keep]


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        target = _wrap_target(target, _canvas_size(image))
        image = F.horizontal_flip(image)
        target["boxes"] = F.horizontal_flip(target["boxes"])
        if "masks" in target:
            target["masks"] = F.horizontal_flip(target["masks"])
        return image, target


def _get_resized_size(
    canvas_size: tuple[int, int], size: int, max_size: int | None
) -> tuple[int, int]:
    """Aspect-preserving short-side resize, optionally capped by *max_size*."""
    h, w = canvas_size
    if max_size is not None:
        min_orig = float(min(h, w))
        max_orig = float(max(h, w))
        if max_orig / min_orig * size > max_size:
            size = int(round(max_size * min_orig / max_orig))

    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


class RandomResize:
    def __init__(self, sizes: Sequence[int], max_size: int | None = None):
        if not isinstance(sizes, (list, tuple)):
            raise TypeError(
                f"sizes must be a list or tuple, got {type(sizes).__name__}"
            )
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = random.choice(self.sizes)
        h0, w0 = _canvas_size(image)
        new_h, new_w = _get_resized_size((h0, w0), size, self.max_size)
        image = F.resize(image, [new_h, new_w])

        if target is None:
            return image, None

        target = _wrap_target(target, (h0, w0))
        target["boxes"] = F.resize(target["boxes"], [new_h, new_w])
        if "masks" in target:
            target["masks"] = F.resize(
                target["masks"], [new_h, new_w], interpolation=F.InterpolationMode.NEAREST
            )
        if "area" in target:
            target["area"] = target["area"] * (new_h / h0) * (new_w / w0)
        target["size"] = torch.tensor([new_h, new_w])
        return image, target


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        h0, w0 = _canvas_size(image)
        w = random.randint(self.min_size, min(w0, self.max_size))
        h = random.randint(self.min_size, min(h0, self.max_size))
        i, j, ch, cw = v2.RandomCrop.get_params(image, [h, w])

        target = _wrap_target(target, (h0, w0))
        image = F.crop(image, i, j, ch, cw)
        target["boxes"] = F.crop(target["boxes"], i, j, ch, cw)
        if "masks" in target:
            target["masks"] = F.crop(target["masks"], i, j, ch, cw)
        target["size"] = torch.tensor([ch, cw])

        boxes = target["boxes"]
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        _filter_instances(target, keep)
        if "area" in target and "boxes" in target:
            b = target["boxes"]
            target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return image, target


# ---------------------------------------------------------------------------
# Compositional / image-level transforms
# ---------------------------------------------------------------------------


class RandomSelect:
    """Randomly applies *transforms1* with probability *p*, else *transforms2*."""

    def __init__(self, transforms1, transforms2, p: float = 0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        chosen = self.transforms1 if random.random() < self.p else self.transforms2
        return chosen(image, target)


class ToTensor:
    def __init__(self):
        self._t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __call__(self, image, target):
        return self._t(image), target


class Normalize:
    """Normalize image and convert boxes to normalised cxcywh (DETR format)."""

    def __init__(self, mean, std):
        self._n = v2.Normalize(mean=list(mean), std=list(std))

    def __call__(self, image, target=None):
        image = self._n(image)
        if target is None:
            return image, None
        target = dict(target)
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            if isinstance(boxes, tv_tensors.BoundingBoxes):
                boxes = boxes.as_subclass(torch.Tensor)
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "masks" in target and isinstance(target["masks"], tv_tensors.Mask):
            target["masks"] = target["masks"].as_subclass(torch.Tensor)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        body = "\n".join(f"    {t}" for t in self.transforms)
        return f"{self.__class__.__name__}(\n{body}\n)"


# ---------------------------------------------------------------------------
# COCO transform pipelines
# ---------------------------------------------------------------------------


def make_coco_transforms(
    image_set: str, params: Augmentation | None = None
) -> Compose:
    if params is None:
        params = Augmentation()

    normalize = Compose(
        [ToTensor(), Normalize(params.normalize_mean, params.normalize_std)]
    )

    if image_set == "train":
        return Compose(
            [
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
                normalize,
            ]
        )

    if image_set == "val":
        return Compose(
            [
                RandomResize([max(params.scales)], max_size=params.max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")
