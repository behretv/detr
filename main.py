# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import detr.parameters as parameters
import detr.util.misc as utils
from detr.data import build_dataset, get_coco_api_from_dataset
from detr.engine import evaluate, train_one_epoch
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

    param_dicts = [
        {
            "params": [
                p
                for n, p in bundle.ai_model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in bundle.ai_model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": train_params.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=train_params.lr, weight_decay=train_params.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_params.lr_drop)

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

    output_dir = Path(run_params.output_dir)
    if run_params.resume:
        if run_params.resume.startswith("https"):
            raw = torch.hub.load_state_dict_from_url(
                run_params.resume, map_location="cpu", check_hash=True
            )
        else:
            raw = torch.load(run_params.resume, map_location="cpu", weights_only=False)
        # Support both ModelBundle exports (key: "state_dict") and raw checkpoints (key: "model")
        state_dict = raw.get("state_dict", raw.get("model"))
        bundle.ai_model.load_state_dict(state_dict)
        if (
            not run_params.eval
            and "optimizer" in raw
            and "lr_scheduler" in raw
            and "epoch" in raw
        ):
            optimizer.load_state_dict(raw["optimizer"])
            lr_scheduler.load_state_dict(raw["lr_scheduler"])
            run_params.start_epoch = raw["epoch"] + 1

    if run_params.eval:
        test_stats, coco_evaluator = evaluate(
            bundle.ai_model,
            bundle.criterion,
            bundle.postprocessors,
            data_loader_val,
            base_ds,
            device,
            run_params.output_dir,
        )
        if run_params.output_dir:
            torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    tqdm.write("Start training")
    start_time = time.time()
    epoch_bar = tqdm(
        range(run_params.start_epoch, train_params.epochs),
        desc="Epochs",
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        train_stats = train_one_epoch(
            bundle.ai_model,
            bundle.criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            train_params.clip_max_norm,
        )
        lr_scheduler.step()
        bundle.logs.append(
            {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        )
        if run_params.output_dir:
            # Save a ModelBundle checkpoint (loadable via ModelBundle.load_from_file)
            bundle.export(output_dir / "checkpoint")
            # Extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % train_params.lr_drop == 0 or (epoch + 1) % 100 == 0:
                bundle.export(output_dir / f"checkpoint{epoch:04}")
            # Also persist optimizer state separately for full resume support
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                output_dir / "optimizer.pth",
            )

        test_stats, coco_evaluator = evaluate(
            bundle.ai_model,
            bundle.criterion,
            bundle.postprocessors,
            data_loader_val,
            base_ds,
            device,
            run_params.output_dir,
        )

        log_stats = {
            "epoch": epoch,
            "n_parameters": n_parameters,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        }

        if run_params.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    tqdm.write(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
