"""Utility / general model tests."""

import torch
from torchvision.ops import box_convert


def test_box_cxcywh_to_xyxy():
    # Arrange
    t = torch.rand(10, 4)

    # Act
    r = box_convert(box_convert(t, "cxcywh", "xyxy"), "xyxy", "cxcywh")

    # Assert
    assert (t - r).abs().max() < 1e-5
