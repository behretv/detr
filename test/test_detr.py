"""DETR model tests."""

import torch
from torch import Tensor, nn

from detr.aux import nested_tensor_from_tensor_list

from ._aux import detr_resnet50


def test_model_script_detection():
    model = detr_resnet50().eval()
    scripted_model = torch.jit.script(model)
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )
    out = model(x)
    out_script = scripted_model(x)
    assert out["pred_logits"].equal(out_script["pred_logits"])
    assert out["pred_boxes"].equal(out_script["pred_boxes"])


def test_model_detection_different_inputs():
    model = detr_resnet50().eval()
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

    model = detr_resnet50()
    wrapped_model = WrappedDETR(model)
    wrapped_model.eval()
    scripted_model = torch.jit.script(wrapped_model)
    x = [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    out = wrapped_model(x)
    out_script = scripted_model(x)
    assert out["pred_logits"].equal(out_script["pred_logits"])
    assert out["pred_boxes"].equal(out_script["pred_boxes"])


def test_train_eval_mode_outputs_differ():
    """Dropout is active in train mode, so outputs must differ from eval mode."""
    torch.manual_seed(42)
    model = detr_resnet50()
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )

    # eval mode is deterministic
    model.eval()
    out_eval_1 = model(x)
    out_eval_2 = model(x)
    assert out_eval_1["pred_logits"].equal(out_eval_2["pred_logits"])
    assert out_eval_1["pred_boxes"].equal(out_eval_2["pred_boxes"])

    # train mode differs from eval because dropout is active
    model.train()
    out_train_1 = model(x)
    assert not out_eval_1["pred_logits"].equal(out_train_1["pred_logits"])

    # train mode is stochastic: two forward passes give different results
    out_train_2 = model(x)
    assert not out_train_1["pred_logits"].equal(out_train_2["pred_logits"])


def test_predict_matches_torchvision_format():
    """DETR.predict() must return the same structure as torchvision detection models."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    images = [torch.rand(3, 200, 200), torch.rand(3, 250, 180)]

    # DETR predict
    detr_model = detr_resnet50()
    detr_out = detr_model.predict(images, score_threshold=0.0)

    # Faster R-CNN eval (random weights, no download)
    rcnn = fasterrcnn_resnet50_fpn(weights=None)
    rcnn.eval()
    rcnn_out = rcnn(images)

    # Both return List[Dict[str, Tensor]]
    assert isinstance(detr_out, list)
    assert isinstance(rcnn_out, list)
    assert len(detr_out) == len(images)
    assert len(rcnn_out) == len(images)

    # Same keys
    for d in detr_out:
        assert set(d.keys()) == {"scores", "labels", "boxes"}
        assert d["boxes"].dim() == 2 and d["boxes"].shape[1] == 4
        assert d["labels"].dim() == 1
        assert d["scores"].dim() == 1
        assert d["labels"].shape[0] == d["boxes"].shape[0] == d["scores"].shape[0]

    for r in rcnn_out:
        assert set(r.keys()) == {"scores", "labels", "boxes"}
        assert r["boxes"].dim() == 2 and r["boxes"].shape[1] == 4
        assert r["labels"].dim() == 1
        assert r["scores"].dim() == 1
        assert r["labels"].shape[0] == r["boxes"].shape[0] == r["scores"].shape[0]

    # DETR boxes should be absolute (larger than 1 for 200x200 images)
    for d in detr_out:
        if d["boxes"].numel():
            assert d["boxes"].max() > 1.0

    # Score threshold should filter DETR results
    detr_filtered = detr_model.predict(images, score_threshold=0.99)
    for d in detr_filtered:
        if d["scores"].numel():
            assert d["scores"].min() > 0.99
