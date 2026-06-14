"""Matcher model tests."""

import pytest
import torch

from detr.models.matcher import HungarianMatcher

from ._aux import indices_torch2python


@pytest.fixture
def matcher_data():
    """Shared arrange data for Hungarian matcher tests."""
    n_queries, n_targets, n_classes = 100, 15, 91
    logits = torch.rand(1, n_queries, n_classes + 1)
    boxes = torch.rand(1, n_queries, 4)
    tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
    tgt_boxes = torch.rand(n_targets, 4)
    return logits, boxes, [{"labels": tgt_labels, "boxes": tgt_boxes}]


def test_hungarian_single_batch(matcher_data):
    # Arrange
    logits, boxes, targets = matcher_data
    matcher = HungarianMatcher()

    # Act
    indices = matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)

    # Assert
    assert len(indices[0][0]) == len(targets[0]["labels"])
    assert len(indices[0][1]) == len(targets[0]["labels"])


def test_hungarian_batched_consistency(matcher_data):
    # Arrange
    logits, boxes, targets = matcher_data
    matcher = HungarianMatcher()

    # Act
    indices_single = matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)
    indices_batched = matcher(
        {
            "pred_logits": logits.repeat(2, 1, 1),
            "pred_boxes": boxes.repeat(2, 1, 1),
        },
        targets * 2,
    )

    # Assert
    assert indices_torch2python(indices_single) == indices_torch2python(
        [indices_batched[0]]
    )
    assert indices_torch2python(indices_single) == indices_torch2python(
        [indices_batched[1]]
    )


def test_hungarian_mixed_empty_targets(matcher_data):
    """Second batch item has no targets; its matching should be empty."""
    # Arrange
    logits, boxes, targets_with = matcher_data
    targets = targets_with + [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
    ]
    matcher = HungarianMatcher()

    # Act
    indices = matcher(
        {"pred_logits": logits.repeat(2, 1, 1), "pred_boxes": boxes.repeat(2, 1, 1)},
        targets,
    )

    # Assert
    assert len(indices[1][0]) == 0


def test_hungarian_all_targets_empty(matcher_data):
    """All batch items have no targets; every matching should be empty."""
    # Arrange
    logits, boxes, _ = matcher_data
    targets = [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
        for _ in range(2)
    ]
    matcher = HungarianMatcher()

    # Act
    indices = matcher(
        {"pred_logits": logits.repeat(2, 1, 1), "pred_boxes": boxes.repeat(2, 1, 1)},
        targets,
    )

    # Assert
    assert len(indices[0][0]) == 0
    assert len(indices[1][0]) == 0
