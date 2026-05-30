# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import detr.parameters as parameters
import detr.train as train
import detr.util.misc as utils
from detr.data import build_dataset, get_coco_api_from_dataset
from detr.engine import evaluate
from detr.model import Bundle
from detr.models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parameters.add_args(parser, parameters.Train)
    parameters.add_args(parser, parameters.Model)
    parameters.add_args(parser, parameters.Loss)
    parameters.add_args(parser, parameters.Data)
    parameters.add_args(parser, parameters.Run)
    return parser


def main(args):
    train_params = parameters.Train.from_args(args)
    model_params = parameters.Model.from_args(args)
    loss_params = parameters.Loss.from_args(args)
    data_params = parameters.Data.from_args(args)
    run_params = parameters.Run.from_args(args)

    print(f"git:\n  {utils.get_sha()}\n")

    if model_params.frozen_weights is not None:
        if not model_params.masks:
            raise ValueError("Frozen training is meant for segmentation only")
    print(train_params, model_params, loss_params, data_params, run_params)

    device = torch.device(run_params.device)

    # fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    bundle = Bundle(
        *build_model(model_params, loss_params, train_params, run_params),
        model_params=model_params,
        loss_params=loss_params,
        train_params=train_params,
        name=data_params.dataset_file,
        source=run_params.resume or "",
        transforms=[],
        cats={},
        device=run_params.device,
    )

    n_parameters = sum(
        p.numel() for p in bundle.ai_model.parameters() if p.requires_grad
    )
    print("number of params:", n_parameters)

    dataset_train = build_dataset(
        image_set="train", data_params=data_params, model_params=model_params
    )
    dataset_val = build_dataset(
        image_set="val", data_params=data_params, model_params=model_params
    )

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

    base_ds = get_coco_api_from_dataset(dataset_val)

    if model_params.frozen_weights is not None:
        checkpoint = torch.load(
            model_params.frozen_weights, map_location="cpu", weights_only=False
        )
        bundle.ai_model.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(run_params.output_dir) if run_params.output_dir else None
    if run_params.resume:
        if run_params.resume.startswith("https"):
            raw = torch.hub.load_state_dict_from_url(
                run_params.resume, map_location="cpu", check_hash=True
            )
        else:
            raw = torch.load(run_params.resume, map_location="cpu", weights_only=False)
        # Support both Bundle exports (key: "state_dict") and raw checkpoints (key: "model")
        state_dict = raw.get("state_dict", raw.get("model"))
        bundle.ai_model.load_state_dict(state_dict)
        if not run_params.eval and "epoch" in raw:
            run_params.start_epoch = raw["epoch"] + 1

    if run_params.eval:
        test_stats, coco_evaluator = evaluate(
            bundle.ai_model,
            bundle.criterion,
            bundle.postprocessors,
            data_loader_val,
            base_ds,
            device,
            str(output_dir) if output_dir else "",
        )
        if output_dir:
            torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    bundle = train.run(
        bundle,
        data_loader_train,
        data_loader_val,
        params=train.Parameters(
            epochs=train_params.epochs,
            lr=train_params.lr,
            lr_backbone=train_params.lr_backbone,
            weight_decay=train_params.weight_decay,
            lr_drop=train_params.lr_drop,
            clip_max_norm=train_params.clip_max_norm,
            start_epoch=run_params.start_epoch,
            output_dir=output_dir,
        ),
        base_ds=base_ds,
    )

    if output_dir:
        with (output_dir / "log.txt").open("w") as f:
            for entry in bundle.logs:
                f.write(json.dumps({**entry, "n_parameters": n_parameters}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
