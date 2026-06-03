"""
Created on 2026-03-13
Copyright (c) 2026 Munich University of Applied Sciences

Script to train a PyTorch model for object detection.
"""

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

import detr
from detr import aux, model, parameters, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start training...")
    t_file = args.dataset / "train.coco.json"
    v_file = args.dataset / "valid.coco.json"
    h_file = args.dataset / "holdout.coco.json"

    # Load transformations
    aug_params = parameters.Augmentation()

    # Load model
    model_data = model.load_from_file(args.model, args.device)
    logger.info(f"Training parameters: {model_data.train_params}")

    # Patch model_data
    with open(t_file, "r", encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    model_data.cats = {c["id"]: c["name"] for c in cats}

    # Load transforms
    t_transforms = model_data.transforms
    if args.augment:
        t_transforms = detr.transforms.make_coco_transforms("train")

    # Load dataset
    t_loader = aux.load_dataset(
        t_file, t_transforms, shuffle=True, batch_size=args.batch_size
    )
    v_loader = aux.load_dataset(
        v_file, model_data.transforms, shuffle=False, batch_size=args.batch_size
    )

    new_model_data = train.run(
        model_data,
        t_loader,
        v_loader,
        params=parameters.Train(batch_size=args.batch_size),
    )

    # Export model + logs
    if args.dir_output:
        new_model_data.export(args.dir_output / model_data.name)

    outputs = detr.coco.inference(new_model_data, v_loader, device=DEVICE)
    stats = detr.coco.run_eval(h_file, outputs)
    logger.info(f"Validation metrics: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=aux.DATA_ROOT / "datasets/accurate-balls"
    )
    parser.add_argument(
        "--dir-output", type=Path, default=aux.DATA_ROOT / "models/torch"
    )
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=Path,
        default=aux.DATA_ROOT / "models/detr/models/detr-r50-e632da11.pth",
    )

    main(parser.parse_args())
