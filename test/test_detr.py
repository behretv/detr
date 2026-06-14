"""DETR model tests."""

import pytest
import torch
from torch import Tensor, nn

from detr.aux import nested_tensor_from_tensor_list

from ._aux import detr_resnet50


def test_model_script_detection():
    # Arrange
    model = detr_resnet50().eval()
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )

    # Act
    scripted_model = torch.jit.script(model)
    out = model(x)
    out_script = scripted_model(x)

    # Assert
    assert out["pred_logits"].equal(out_script["pred_logits"])
    assert out["pred_boxes"].equal(out_script["pred_boxes"])


@pytest.mark.parametrize(
    "x",
    [
        # NestedTensor
        nested_tensor_from_tensor_list(
            [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
        ),
        # 4d Tensor
        torch.rand(1, 3, 200, 200),
        # List[Tensor[C, H, W]]
        [torch.rand(3, 200, 200)],
    ],
)
def test_model_detection_different_inputs(x):
    # Arrange
    model = detr_resnet50().eval()

    # Act
    out = model(x)

    # Assert
    assert "pred_logits" in out


def test_wrapped_model_script_detection():
    # Arrange
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
    x = [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]

    # Act
    scripted_model = torch.jit.script(wrapped_model)
    out = wrapped_model(x)
    out_script = scripted_model(x)

    # Assert
    assert out["pred_logits"].equal(out_script["pred_logits"])
    assert out["pred_boxes"].equal(out_script["pred_boxes"])


def test_eval_mode_is_deterministic():
    """Eval mode produces identical outputs for the same input."""
    # Arrange
    torch.manual_seed(42)
    model = detr_resnet50().eval()
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )

    # Act
    out_1 = model(x)
    out_2 = model(x)

    # Assert
    assert out_1["pred_logits"].equal(out_2["pred_logits"])
    assert out_1["pred_boxes"].equal(out_2["pred_boxes"])


def test_train_mode_differs_from_eval():
    """Dropout in train mode changes the output compared to eval."""
    # Arrange
    torch.manual_seed(42)
    model = detr_resnet50()
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )

    # Act
    model.eval()
    out_eval = model(x)
    model.train()
    out_train = model(x)

    # Assert
    assert not out_eval["pred_logits"].equal(out_train["pred_logits"])


def test_train_mode_is_stochastic():
    """Two train-mode forward passes produce different results."""
    # Arrange
    torch.manual_seed(42)
    model = detr_resnet50().train()
    x = nested_tensor_from_tensor_list(
        [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    )

    # Act
    out_1 = model(x)
    out_2 = model(x)

    # Assert
    assert not out_1["pred_logits"].equal(out_2["pred_logits"])


def _assert_torchvision_detection_format(results: list, n_images: int) -> None:
    """Shared assertion helper: verify torchvision-compatible detection format."""
    assert isinstance(results, list)
    assert len(results) == n_images
    for r in results:
        assert set(r.keys()) == {"scores", "labels", "boxes"}
        assert r["boxes"].dim() == 2 and r["boxes"].shape[1] == 4
        assert r["labels"].dim() == 1
        assert r["scores"].dim() == 1
        assert r["labels"].shape[0] == r["boxes"].shape[0] == r["scores"].shape[0]


def test_predict_matches_torchvision_format():
    """DETR.predict() must return the same structure as torchvision detection models."""
    # Arrange
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    images = [torch.rand(3, 200, 200), torch.rand(3, 250, 180)]
    detr_model = detr_resnet50()
    rcnn = fasterrcnn_resnet50_fpn(weights=None)
    rcnn.eval()

    # Act
    detr_out = detr_model.predict(images, score_threshold=0.0)
    rcnn_out = rcnn(images)

    # Assert
    _assert_torchvision_detection_format(detr_out, len(images))
    _assert_torchvision_detection_format(rcnn_out, len(images))


def test_predict_boxes_are_absolute():
    """DETR predict boxes must be in absolute pixel coordinates."""
    # Arrange
    images = [torch.rand(3, 200, 200)]

    # Act
    out = detr_resnet50().predict(images, score_threshold=0.0)

    # Assert
    assert out[0]["boxes"].max() > 1.0


def test_predict_score_threshold_filters():
    """Higher score_threshold should only return detections above the threshold."""
    # Arrange
    images = [torch.rand(3, 200, 200)]

    # Act
    out = detr_resnet50().predict(images, score_threshold=0.99)

    # Assert
    assert (out[0]["scores"] > 0.99).all()
