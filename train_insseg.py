# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Train the DETR segmentation head on top of a pretrained detector.

Usage
-----
    python train_insseg.py \
        --frozen_weights output/detector/checkpoint.pth \
        --coco_path /data/coco \
        --output_dir output/segmentation

The detector backbone and box head are frozen; only the mask head is trained.
Pass --no-frozen_weights to fine-tune the full model end-to-end instead.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

import detr.misc as utils
import detr.parameters as parameters
import detr.train as train
from detr.dataset import CocoDetection
from detr.evaluate import evaluate
from detr.model import Bundle


def get_args_parser():
    parser = argparse.ArgumentParser("DETR segmentation training", add_help=False)
    parameters.add_args(parser, parameters.Train)
    parameters.add_args(parser, parameters.Model)
    parameters.add_args(parser, parameters.Loss)
    parameters.add_args(parser, parameters.Run)
    return parser


def main(args):
    train_params = parameters.Train.from_args(args)
    model_params = parameters.Model.from_args(args)
    loss_params = parameters.Loss.from_args(args)
    run_params = parameters.Run.from_args(args)

    if not model_params.masks:
        raise ValueError("--masks must be set for segmentation training")

    logger.debug(f"Train params: {train_params}")
    logger.debug(f"Model params: {model_params}")
    logger.debug(f"Loss params: {loss_params}")
    logger.debug(f"Run params: {run_params}")

    device = torch.device(run_params.device)

    # fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    bundle = Bundle.load_from_file(run_params.model, device=run_params.device)

    # Load pretrained detector weights into the DETR sub-module (frozen backbone + box head)
    if model_params.frozen_weights is not None:
        checkpoint = torch.load(
            model_params.frozen_weights, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint.get("model"))
        bundle.ai_model.detr.load_state_dict(state_dict)

    n_parameters = sum(
        p.numel() for p in bundle.ai_model.parameters() if p.requires_grad
    )
    logger.info(f"Number of params: {n_parameters}")

    dataset_train = CocoDetection.build("train", run_params, model_params)
    dataset_val = CocoDetection.build("val", run_params, model_params)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, train_params.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=train_params.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        train_params.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=train_params.num_workers,
    )

    base_ds = dataset_val.coco_api()

    output_dir = Path(run_params.output_dir) if run_params.output_dir else None

    if run_params.eval:
        evaluate(
            bundle.ai_model,
            bundle.criterion,
            bundle.postprocessors,
            data_loader_val,
            base_ds,
            device,
        )
        return

    train.run(
        bundle,
        data_loader_train,
        data_loader_val,
        params=train_params,
        output_dir=output_dir,
        base_ds=base_ds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR segmentation training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    args.masks = True  # always enable masks for this script
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
