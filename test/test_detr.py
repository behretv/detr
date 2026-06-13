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
