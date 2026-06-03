"""
Created on 2026-06-03
Copyright (c) 2026 Munich University of Applied Sciences
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detr


@pytest.fixture(scope="session")
def coco_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco")
    coco_json = _make_coco_json(root)
    for split_file in ("train.coco.json", "valid.coco.json"):
        (root / split_file).write_text(json.dumps(coco_json))
    return root


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model_bundle(device):
    model_params = _small_model_params()
    bundle = detr.model.factory(
        model_params=model_params,
        loss_params=detr.parameters.Loss(),
        train_params=detr.parameters.Train(),
    )
    bundle.set_device("cpu")
    return bundle


@pytest.fixture(scope="session")
def train_loader(coco_root):
    ds = detr.dataset.CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=detr.transforms.make_coco_transforms("valid"),
        return_masks=False,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=detr.misc.collate_fn, num_workers=0
    )


def _make_coco_json(tmp_dir: Path, n_images: int = 4) -> dict:
    """Write tiny JPEG images to *tmp_dir* and return a minimal COCO dict."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n_images + 1):
        w, h = 128, 96
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        fname = f"{i:012d}.jpg"
        Image.fromarray(arr).save(tmp_dir / fname)
        images.append({"id": i, "file_name": fname, "width": w, "height": h})
        for x0, y0, bw, bh in [(10, 10, 30, 20), (50, 40, 25, 25)]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x0, y0, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ball", "supercategory": "object"}],
    }


def _small_model_params() -> detr.parameters.Model:
    return detr.parameters.Model(
        backbone=detr.parameters.BackboneType.RESNET50,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=64,
        hidden_dim=32,
        nheads=2,
        num_queries=10,
        aux_loss=False,
    )
