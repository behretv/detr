# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import detr.misc as utils
import detr.parameters as parameters
import detr.train as train
from detr.dataset import CocoDetection
from detr.evaluate import evaluate
from detr.model import Bundle


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

    device = torch.device(run_params.device)

    # fix the seed for reproducibility
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)

    bundle = Bundle.load_from_file(run_params.base_model, device=run_params.device)

    n_parameters = sum(
        p.numel() for p in bundle.ai_model.parameters() if p.requires_grad
    )
    print("number of params:", n_parameters)

    dataset_train = CocoDetection.build("train", data_params, model_params)
    dataset_val = CocoDetection.build("val", data_params, model_params)

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

    bundle = train.run(
        bundle,
        data_loader_train,
        data_loader_val,
        params=train_params,
        output_dir=output_dir,
        base_ds=base_ds,
    )

    evaluate(
        bundle.ai_model,
        bundle.criterion,
        bundle.postprocessors,
        data_loader_val,
        base_ds,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
