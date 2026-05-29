# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "detr"))

import argparse
import datetime
import json
import random
import time

import numpy as np
import torch
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from params import (
    DataParameters,
    LossParameters,
    ModelParameters,
    RunParameters,
    TrainParameters,
    _add_dataclass_args,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    _add_dataclass_args(parser, TrainParameters)
    _add_dataclass_args(parser, ModelParameters)
    _add_dataclass_args(parser, LossParameters)
    _add_dataclass_args(parser, DataParameters)
    _add_dataclass_args(parser, RunParameters)
    return parser


def main(args):
    train_params = TrainParameters.from_args(args)
    model_params = ModelParameters.from_args(args)
    loss_params = LossParameters.from_args(args)
    data_params = DataParameters.from_args(args)
    run_params = RunParameters.from_args(args)

    print(f"git:\n  {utils.get_sha()}\n")

    if model_params.frozen_weights is not None:
        assert model_params.masks, "Frozen training is meant for segmentation only"
    print(train_params, model_params, loss_params, data_params, run_params)

    device = torch.device(run_params.device)

    # fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    model, criterion, postprocessors = build_model(
        model_params, loss_params, train_params, run_params
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
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
        checkpoint = torch.load(model_params.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(run_params.output_dir)
    if run_params.resume:
        if run_params.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                run_params.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(run_params.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not run_params.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            run_params.start_epoch = checkpoint["epoch"] + 1

    if run_params.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
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
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            train_params.clip_max_norm,
        )
        lr_scheduler.step()
        if run_params.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % train_params.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "train_params": train_params,
                        "model_params": model_params,
                        "loss_params": loss_params,
                        "data_params": data_params,
                        "run_params": run_params,
                    },
                    checkpoint_path,
                )

        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            run_params.output_dir,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
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
