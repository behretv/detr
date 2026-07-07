"""
Created on 2026-06-13
Copyright (c) 2026 Munich University of Applied Sciences
"""

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import detr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    detr.parameters.add_args(parser, detr.parameters.Train)
    detr.parameters.add_args(parser, detr.parameters.Model)
    detr.parameters.add_args(parser, detr.parameters.Loss)
    detr.parameters.add_args(parser, detr.parameters.Augmentation)
    return parser


def main(args):
    train_params = detr.parameters.Train.from_args(args)
    model_params = detr.parameters.Model.from_args(args)
    aug_params = detr.parameters.Augmentation.from_args(args)

    # fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    categories = detr.io.load_categories(args.file_train)

    t_file = args.file_train
    v_file = t_file.with_name(t_file.name.replace("train", "valid"))
    h_file = t_file.with_name("holdout.coco.json")

    if detr.io.is_legacy_model(args.model):
        logger.warning(f"Model {args.model} is in legacy format, converting...")
        model = detr.io.load_model_legacy(args.model, device=DEVICE, categories=categories)
    else:
        model = detr.io.load_model(args.model, device=DEVICE, categories=categories)
    model.meta = detr.ModelMeta(
        categories=categories,
        dataset=str(t_file.relative_to(t_file.parents[1])),
        model_type=model.meta.model_type,
        subtype=model.meta.subtype,
        train_params=train_params,
        settings={
            "model_params": asdict(model_params),
            "aug_params": asdict(aug_params),
        },
    )
    logger.info(f"Training parameters: {model.meta.train_params}")


    t_transforms = detr.transforms.augmentation(aug_params)
    v_transforms = detr.transforms.default()

    t_loader = detr.dataset.load(
        t_file,
        t_transforms,
        return_masks=model.meta.model_type is detr.ModelType.DETR_SEGM,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    v_loader = detr.dataset.load(
        v_file,
        v_transforms,
        return_masks=model.meta.model_type is detr.ModelType.DETR_SEGM,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )
    h_loader = detr.dataset.load(
        h_file,
        v_transforms,
        return_masks=model.meta.model_type is detr.ModelType.DETR_SEGM,
        batch_size=train_params.batch_size,
        num_workers=train_params.num_workers,
    )

    model.set_device(DEVICE)
    model = detr.train.run(
        model,
        t_loader,
        v_loader,
        device=DEVICE,
        params=train_params,
        v_file=v_file,
    )

    outputs = detr.coco.inference(model, v_loader, device=DEVICE)
    stats = detr.coco.run_eval(v_file, outputs, iou_type="bbox")
    logger.info(f"Validation metrics: {stats}")

    outputs = detr.coco.inference(model, h_loader, device=DEVICE)
    stats = detr.coco.run_eval(h_file, outputs, iou_type="bbox")
    logger.info(f"Holdout metrics: {stats}")

    if args.output_dir:
        filename = detr.io.output_filename(args.output_dir, model.name)
        detr.io.save_model(model, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--file-train", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    main(args)
