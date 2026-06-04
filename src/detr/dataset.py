# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from detr.aux import collate_fn


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def load_dataset(
    ann_file: Path,
    transforms: v2.Compose,
    return_masks: bool = False,
    batch_size: int = 2,
    num_workers: int = 2,
) -> DataLoader:
    """Build a COCO ``DataLoader`` from an annotation file.

    The sampler is chosen automatically from the file name stem:
    ``train`` uses a :class:`RandomSampler`, while anything else uses a
    :class:`SequentialSampler`.

    Parameters
    ----------
    ann_file:
        Path to a COCO-format ``*.json`` file. Image files are expected to live
        in the same directory.
    transforms:
        Pre-built transform pipeline (e.g. ``detr.transforms.train()`` or
        ``detr.transforms.default()``).
    return_masks:
        Whether to include COCO segmentation masks in the target dict.
    batch_size, num_workers:
        Standard DataLoader knobs.
    """
    ann_file = Path(ann_file)
    image_set = ann_file.name.split(".")[0]
    is_train = image_set == "train"

    dataset = CocoDetection(
        ann_file.parent,
        ann_file,
        transforms=transforms,
        return_masks=return_masks,
    )

    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
        loader_kwargs = {
            "dataset": dataset,
            "batch_sampler": torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last=True
            ),
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "sampler": sampler,
            "drop_last": False,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

    return DataLoader(**loader_kwargs)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        classes = torch.tensor([obj["category_id"] for obj in anno], dtype=torch.int64)

        tmp_target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
            ),
            "labels": classes,
        }

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            tmp_target["masks"] = tv_tensors.Mask(masks)

        image, tmp_target = v2.SanitizeBoundingBoxes()(image, tmp_target)

        boxes = tmp_target["boxes"]
        target = {
            "boxes": boxes,
            "labels": tmp_target["labels"],
            "image_id": image_id,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        if self.return_masks:
            target["masks"] = tmp_target["masks"]

        return image, target
