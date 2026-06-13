"""Backbone model tests."""

import torch

from detr.models.backbone import Backbone
from detr.parameters import BackboneType


def test_backbone_script():
    backbone = Backbone(BackboneType.RESNET50, True, False, False)
    torch.jit.script(backbone)  # noqa
