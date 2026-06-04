# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as v2
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from torchvision import tv_tensors

import detr.parameters as parameters
from detr.aux import collate_fn
from detr.transforms import make_coco_transforms


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    @classmethod
    def build(
        cls,
        file: Path,
        model_params: parameters.Model,
        aug_params: parameters.Augmentation | None = None,
    ) -> CocoDetection:
        image_set = file.name.split(".")[0]
        return cls(
            file.parent,
            file,
            transforms=make_coco_transforms(image_set, aug_params),
            return_masks=model_params.masks,
        )

    def coco_api(self):
        """Return the underlying pycocotools COCO API object for evaluation."""
        dataset = self
        for _ in range(10):
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
        return dataset.coco

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
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        )
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = tv_tensors.Mask(masks)
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target
