"""Shared helpers for model tests."""

from detr.models.backbone import Backbone, Joiner
from detr.models.detr import DETR
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.transformer import Transformer
from detr.parameters import BackboneType


def detr_resnet50(pretrained=False):
    hidden_dim = 256
    backbone = Backbone(
        BackboneType.RESNET50,
        train_backbone=True,
        return_interm_layers=False,
        dilation=False,
    )
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    return DETR(backbone_with_pos_enc, transformer, num_classes=91, num_queries=100)


def indices_torch2python(indices):
    return [(i.tolist(), j.tolist()) for i, j in indices]
