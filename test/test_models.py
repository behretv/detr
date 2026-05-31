# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import Tensor, nn

from detr.models.backbone import Backbone, Joiner
from detr.models.detr import DETR
from detr.models.matcher import HungarianMatcher
from detr.models.position_encoding import (
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
)
from detr.models.transformer import Transformer
from torchvision.ops import box_convert

from detr.misc import nested_tensor_from_tensor_list


def detr_resnet50(pretrained=False):
    hidden_dim = 256
    backbone = Backbone(
        "resnet50", train_backbone=True, return_interm_layers=False, dilation=False
    )
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    return DETR(backbone_with_pos_enc, transformer, num_classes=91, num_queries=100)


def _indices_torch2python(indices):
    return [(i.tolist(), j.tolist()) for i, j in indices]


# ---------------------------------------------------------------------------
# General tests
# ---------------------------------------------------------------------------


def test_box_cxcywh_to_xyxy():
    t = torch.rand(10, 4)
    r = box_convert(box_convert(t, "cxcywh", "xyxy"), "xyxy", "cxcywh")
    assert (t - r).abs().max() < 1e-5


def test_hungarian():
    n_queries, n_targets, n_classes = 100, 15, 91
    logits = torch.rand(1, n_queries, n_classes + 1)
    boxes = torch.rand(1, n_queries, 4)
    tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
    tgt_boxes = torch.rand(n_targets, 4)
    matcher = HungarianMatcher()
    targets = [{"labels": tgt_labels, "boxes": tgt_boxes}]
    indices_single = matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)
    indices_batched = matcher(
        {
            "pred_logits": logits.repeat(2, 1, 1),
            "pred_boxes": boxes.repeat(2, 1, 1),
        },
        targets * 2,
    )
    assert len(indices_single[0][0]) == n_targets
    assert len(indices_single[0][1]) == n_targets
    assert _indices_torch2python(indices_single) == _indices_torch2python(
        [indices_batched[0]]
    )
    assert _indices_torch2python(indices_single) == _indices_torch2python(
        [indices_batched[1]]
    )

    # test with empty targets
    tgt_labels_empty = torch.randint(high=n_classes, size=(0,))
    tgt_boxes_empty = torch.rand(0, 4)
    targets_empty = [{"labels": tgt_labels_empty, "boxes": tgt_boxes_empty}]
    indices = matcher(
        {
            "pred_logits": logits.repeat(2, 1, 1),
            "pred_boxes": boxes.repeat(2, 1, 1),
        },
        targets + targets_empty,
    )
    assert len(indices[1][0]) == 0
    indices = matcher(
        {
            "pred_logits": logits.repeat(2, 1, 1),
            "pred_boxes": boxes.repeat(2, 1, 1),
        },
        targets_empty * 2,
    )
    assert len(indices[0][0]) == 0


def test_position_encoding_script():
    m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
    torch.jit.script(m1), torch.jit.script(m2)  # noqa


def test_backbone_script():
    backbone = Backbone("resnet50", True, False, False)
    torch.jit.script(backbone)  # noqa


def test_model_script_detection():
    model = detr_resnet50(pretrained=False).eval()
    scripted_model = torch.jit.script(model)
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )
    out = model(x)
    out_script = scripted_model(x)
    assert out["pred_logits"].equal(out_script["pred_logits"])
    assert out["pred_boxes"].equal(out_script["pred_boxes"])


def test_model_detection_different_inputs():
    model = detr_resnet50(pretrained=False).eval()
    # support NestedTensor
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )
    out = model(x)
    assert "pred_logits" in out
    # and 4d Tensor
    x = torch.rand(1, 3, 200, 200)
    out = model(x)
    assert "pred_logits" in out
    # and List[Tensor[C, H, W]]
    x = torch.rand(3, 200, 200)
    out = model([x])
    assert "pred_logits" in out


def test_wrapped_model_script_detection():
    class WrappedDETR(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs: list[Tensor]):
            sample = nested_tensor_from_tensor_list(inputs)
            return self.model(sample)

    model = detr_resnet50(pretrained=False)
    wrapped_model = WrappedDETR(model)
    wrapped_model.eval()
    scripted_model = torch.jit.script(wrapped_model)
    x = [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    out = wrapped_model(x)
    out_script = scripted_model(x)
    assert out["pred_logits"].equal(out_script["pred_logits"])
    assert out["pred_boxes"].equal(out_script["pred_boxes"])
