# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d

import detr.parameters as parameters
from detr.aux import NestedTensor

from .position_encoding import build_position_encoding


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            if m is None:
                raise ValueError("mask must not be None in BackboneBase.forward")
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: parameters.ModelSubType,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        name_str = name.value
        weights = torchvision.models.get_model_weights(name_str).DEFAULT
        backbone = getattr(torchvision.models, name_str)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name_str in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: list[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(
    model_params: parameters.Model,
    train_params: parameters.Train,
    subtype: parameters.ModelSubType,
    model_type: parameters.ModelType,
):
    position_embedding = build_position_encoding(model_params)
    train_backbone = train_params.lr_backbone > 0
    return_interm_layers = model_type is parameters.ModelType.DETR_SEGM
    backbone = Backbone(
        subtype,
        train_backbone,
        return_interm_layers,
        model_params.dilation,
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
