"""
Created on 2026-06-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import torch

import detr


def test_dataset_length(coco_root):
    # Act
    ds = detr.dataset.CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=detr.transforms.augmentation(),
        return_masks=False,
    )
    length = len(ds)

    # Assert
    assert length == 4


def test_dataset_item_keys(coco_root):
    # Act
    ds = detr.dataset.CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=detr.transforms.default(),
        return_masks=False,
    )
    img, target = ds[0]

    # Assert
    assert isinstance(img, torch.Tensor)
    for key in ("boxes", "labels", "image_id", "area", "iscrowd", "orig_size", "size"):
        assert key in target, f"missing key: {key}"


def test_dataset_boxes_normalised(coco_root):
    # Act
    ds = detr.dataset.CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=detr.transforms.default(),
        return_masks=False,
    )
    _, target = ds[0]
    boxes = target["boxes"]

    # Assert
    assert (boxes >= 0).all(), "box coords should be non-negative"
    assert (boxes <= 1).all(), "box coords should be ≤ 1 after normalisation"


def test_dataloader_batch(train_loader):
    # Act
    samples, targets = next(iter(train_loader))

    # Assert
    assert len(targets) == 2
    assert samples.tensors is not None
    assert samples.tensors.shape[0] == 2
