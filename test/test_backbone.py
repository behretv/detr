"""Backbone model tests."""

import torch

from detr.models.backbone import Backbone
from detr.parameters import ModelSubType


def test_backbone_script():
    # Arrange
    backbone = Backbone(ModelSubType.RESNET50, True, False, False)

    # Act + Assert (should not raise)
    torch.jit.script(backbone)  # noqa
