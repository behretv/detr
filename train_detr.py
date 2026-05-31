"""
Created on 2026-03-13
Copyright (c) 2026 Munich University of Applied Sciences

Script to train a PyTorch model for object detection.
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from detr import aux, model, train


def main(args: argparse.Namespace):
    """Entrypoint: run --help for details."""
    logger.info("Start training...")
    t_file = args.dataset / "train.coco.json"
    v_file = args.dataset / "valid.coco.json"

    # Load transformations
    aug_params = train.Augmentation()

    # Load model
    model_data = model.load_from_file(args.model, args.device)

    # Patch model_data
    with open(t_file, "r", encoding="utf-8") as f:
        cats = json.load(f)["categories"]
    model_data.cats = {c["id"]: c["name"] for c in cats}

    # Load transforms
    t_transforms = model_data.transforms
    if args.augment:
        model_data.with_augmentation = True
        t_transforms += train.augmentation_transforms(aug_params)

    # Load dataset
    t_loader = aux.load_dataset(t_file, t_transforms, shuffle=True, batch_size=args.batch_size)
    v_loader = aux.load_dataset(v_file, model_data.transforms, shuffle=False, batch_size=args.batch_size)

    new_model_data = train.run(model_data, t_loader, v_loader, params=train.Parameter(batch_size=args.batch_size))

    # Export model + logs
    if args.dir_output:
        new_model_data.export(args.dir_output / model_data.name)


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
