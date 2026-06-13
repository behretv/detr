# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import detr
import detr.parameters as parameters
import detr.train as train
from detr.transforms import default as test_transforms
from detr.transforms import train as train_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parameters.add_args(parser, parameters.Train)
    parameters.add_args(parser, parameters.Model)
    parameters.add_args(parser, parameters.Loss)
    parameters.add_args(parser, parameters.Augmentation)
    return parser


def main(args):
    train_params = parameters.Train.from_args(args)
    model_params = parameters.Model.from_args(args)
    aug_params = parameters.Augmentation.from_args(args)

    # fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    bundle = detr.model.load_from_file(args.model, device=DEVICE)
    logger.info(f"Training parameters: {bundle.train_params}")

    t_file = Path(args.dataset) / "train.coco.json"
    v_file = Path(args.dataset) / "valid.coco.json"
    h_file = Path(args.dataset) / "holdout.coco.json"

    t_transforms = train_transforms(aug_params)
    v_transforms = test_transforms(aug_params)

    t_loader = detr.dataset.load_dataset(
        t_file,
        t_transforms,
        return_masks=model_params.masks,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    v_loader = detr.dataset.load_dataset(
        v_file,
        v_transforms,
        return_masks=model_params.masks,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    h_loader = detr.dataset.load_dataset(
        h_file,
        v_transforms,
        return_masks=model_params.masks,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )

    bundle.set_device(DEVICE)
    bundle = train.run(
        bundle,
        t_loader,
        v_loader,
        device=DEVICE,
        params=train_params,
        v_file=v_file,
    )

    outputs = detr.coco.inference(bundle, v_loader, device=DEVICE)
    stats = detr.coco.run_eval(v_file, outputs, iou_type="bbox")
    logger.info(f"Validation metrics: {stats}")

    outputs = detr.coco.inference(bundle, h_loader, device=DEVICE)
    stats = detr.coco.run_eval(h_file, outputs, iou_type="bbox")
    logger.info(f"Holdout metrics: {stats}")

    if args.output_dir:
        filename = detr.model.filename(args.output_dir, bundle.name)
        bundle.export(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    main(args)
