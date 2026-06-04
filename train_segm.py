"""Train the DETR instance-segmentation head on top of a pretrained detector.

Usage
-----
    python train_insseg.py \
        --dataset /data/coco \
        --frozen-weights output/detector/checkpoint.pth \
        --output_dir output/segmentation

The detector backbone and box head are frozen; only the mask head is trained.
Omit --frozen-weights to fine-tune the full model end-to-end.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

import detr
import detr.aux as utils
import detr.parameters as parameters
import detr.train as train
from detr.dataset import CocoDetection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    train_params = parameters.Train.from_args(args)
    model_params = parameters.Model.from_args(args)
    loss_params = parameters.Loss.from_args(args)
    aug_params = parameters.Augmentation.from_args(args)

    if not model_params.masks:
        raise ValueError("--masks must be set for segmentation training")

    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    # Build model with segmentation head
    bundle = detr.model.factory(model_params, loss_params, train_params)
    bundle.set_device(DEVICE)

    # Load pretrained detector weights into the DETR sub-module
    if args.frozen_weights is not None:
        checkpoint = torch.load(
            args.frozen_weights, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint.get("model"))
        bundle.ai_model.detr.load_state_dict(state_dict)
        logger.info(f"Loaded frozen weights from {args.frozen_weights}")

    t_file = Path(args.dataset) / "train.coco.json"
    v_file = Path(args.dataset) / "valid.coco.json"
    h_file = Path(args.dataset) / "holdout.coco.json"

    t_dataset = CocoDetection.build(t_file, model_params, aug_params)
    v_dataset = CocoDetection.build(v_file, model_params, aug_params)
    h_dataset = CocoDetection.build(h_file, model_params, aug_params)

    t_sampler = torch.utils.data.RandomSampler(t_dataset)
    v_sampler = torch.utils.data.SequentialSampler(v_dataset)
    h_sampler = torch.utils.data.SequentialSampler(h_dataset)

    t_batch_sampler = torch.utils.data.BatchSampler(
        t_sampler, train_params.batch_size, drop_last=True
    )

    t_loader = DataLoader(
        t_dataset,
        batch_sampler=t_batch_sampler,
        collate_fn=utils.collate_fn,
        num_workers=train_params.num_workers,
    )
    v_loader = DataLoader(
        v_dataset,
        train_params.batch_size,
        sampler=v_sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=train_params.num_workers,
    )
    h_loader = DataLoader(
        h_dataset,
        train_params.batch_size,
        sampler=h_sampler,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=train_params.num_workers,
    )

    bundle = train.run(
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
    parameters.add_args(parser, parameters.Train)
    parameters.add_args(parser, parameters.Model)
    parameters.add_args(parser, parameters.Loss)
    parameters.add_args(parser, parameters.Augmentation)
    parser.add_argument(
        "--frozen-weights",
        type=Path,
        default=None,
        help="Path to pretrained DETR weights to freeze",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    args.masks = True  # force masks for this script
    main(args)
