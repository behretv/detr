# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pytest
import torch

from detr.models.backbone import Backbone, Joiner
from detr.models.detr import DETR
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.transformer import Transformer

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

pytestmark = pytest.mark.skipif(onnxruntime is None, reason="ONNX Runtime unavailable")


def detr_resnet50():
    hidden_dim = 256
    backbone = Backbone(
        "resnet50", train_backbone=True, return_interm_layers=False, dilation=False
    )
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    return DETR(backbone_with_pos_enc, transformer, num_classes=91, num_queries=100)


@pytest.fixture(autouse=True, scope="module")
def fixed_seed():
    torch.manual_seed(123)


def test_model_onnx_detection():
    from detr.onnx import export_and_validate

    model = detr_resnet50().eval()
    dummy_image = torch.ones(1, 3, 800, 800) * 0.3
    model(dummy_image)

    export_and_validate(
        model,
        [(torch.rand(1, 3, 750, 800),)],
        input_names=["inputs"],
        output_names=["pred_logits", "pred_boxes"],
        tolerate_small_mismatch=True,
    )
