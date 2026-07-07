"""Train the DETR instance-segmentation head on top of a pretrained detector.

Usage
-----
    python train_insseg.py \
        --file-train /data/coco/train.coco.json \
        --model output/detector/checkpoint.pth \
        --output-dir output/segmentation

The detector backbone and box head are frozen; only the mask head is trained.
Omit --model to fine-tune the full model end-to-end.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import detr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    train_params = detr.parameters.Train.from_args(args)
    model_params = detr.parameters.Model.from_args(args)
    loss_params = detr.parameters.Loss.from_args(args)
    aug_params = detr.parameters.Augmentation.from_args(args)

    if model_params.model_type is not detr.ModelType.DETR_SEGM:
        raise ValueError("model_type must be DETR_SEGM for segmentation training")

    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    # Freeze DETR backbone if pretrained weights are provided
    if args.model is not None:
        model_params.frozen = True

    # Build model with segmentation head
    categories = detr.io.load_categories(args.file_train)
    meta = detr.ModelMeta(
        categories=categories,
        model_type=model_params.model_type,
        subtype=model_params.backbone,
        model_params=model_params,
        loss_params=loss_params,
        train_params=train_params,
    )
    bundle = detr.model.factory(meta)
    bundle.set_device(DEVICE)

    # Load pretrained detector weights into the DETR sub-module
    if args.model is not None:
        checkpoint = detr.io.load_checkpoint(args.model, device=DEVICE)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model"))
        bundle.ai.detr.load_state_dict(state_dict)
        logger.info(f"Loaded frozen weights from {args.model}")

    t_file = args.file_train
    v_file = t_file.with_name(t_file.name.replace("train", "valid"))
    h_file = t_file.with_name("holdout.coco.json")

    t_transforms = detr.transforms.augmentation(aug_params)
    v_transforms = detr.transforms.default(aug_params)

    t_loader = detr.dataset.load(
        t_file,
        t_transforms,
        return_masks=True,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    v_loader = detr.dataset.load(
        v_file,
        v_transforms,
        return_masks=True,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    h_loader = detr.dataset.load(
        h_file,
        v_transforms,
        return_masks=True,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )

    bundle = detr.train.run(
        bundle,
        t_loader,
        v_loader,
        device=DEVICE,
        params=train_params,
        v_file=v_file,
    )

    outputs = detr.coco.inference(bundle, v_loader, device=DEVICE)
    stats = detr.coco.run_eval(v_file, outputs, iou_type="segm")
    logger.info(f"Validation metrics: {stats}")

    outputs = detr.coco.inference(bundle, h_loader, device=DEVICE)
    stats = detr.coco.run_eval(h_file, outputs, iou_type="segm")
    logger.info(f"Holdout metrics: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR instance-segmentation training", add_help=False
    )
    detr.parameters.add_args(parser, detr.parameters.Train)
    detr.parameters.add_args(parser, detr.parameters.Model)
    detr.parameters.add_args(parser, detr.parameters.Loss)
    detr.parameters.add_args(parser, detr.parameters.Augmentation)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to pretrained DETR weights to freeze",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--file-train", type=Path, required=True)
    args = parser.parse_args()
    args.model_type = detr.ModelType.DETR_SEGM  # force segmentation for this script
    main(args)
