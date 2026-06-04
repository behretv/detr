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

import detr.parameters as parameters
from detr.aux import collate_fn
from detr.transforms import default as test_transforms
from detr.transforms import train as train_transforms


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
        tfms = (
            train_transforms(aug_params)
            if image_set == "train"
            else test_transforms(aug_params)
        )
        return cls(
            file.parent,
            file,
            transforms=tfms,
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


def load_simple_dataset(
    file: Path,
    transforms: list[v2.Transform] | v2.Compose,
    shuffle=True,
    return_masks=False,
):
    """Wrapper for torch-based dataset"""
    transforms = v2.Compose(transforms) if isinstance(transforms, list) else transforms
    dataset_coco = torchvision.datasets.CocoDetection(
        file.parent, str(file), transforms=transforms, return_masks=return_masks
    )
    dataset_coco = torchvision.datasets.wrap_dataset_for_transforms_v2(
        dataset_coco,
        target_keys=("boxes", "labels", "image_id")
        if return_masks
        else ("boxes", "labels", "image_id", "masks"),
    )

    return torch.utils.data.DataLoader(
        dataset_coco,
        batch_size=4,
        shuffle=shuffle,
        num_workers=8,
        collate_fn=collate_fn,
    )
