# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import detr.parameters as parameters

from .detr import build


def build_model(
    model_params: parameters.Model,
    loss_params: parameters.Loss,
    train_params: parameters.Train,
    run_params: parameters.Run,
):
    return build(model_params, loss_params, train_params, run_params)
