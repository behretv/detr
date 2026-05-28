# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from _types import LossParameters, ModelParameters, RunParameters, TrainParameters

from .detr import build


def build_model(
    model_params: ModelParameters,
    loss_params: LossParameters,
    train_params: TrainParameters,
    run_params: RunParameters,
):
    return build(model_params, loss_params, train_params, run_params)
