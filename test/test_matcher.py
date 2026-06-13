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
    logits, boxes, targets = matcher_data
    matcher = HungarianMatcher()
    indices = matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)
    assert len(indices[0][0]) == len(targets[0]["labels"])
    assert len(indices[0][1]) == len(targets[0]["labels"])


def test_hungarian_batched_consistency(matcher_data):
    logits, boxes, targets = matcher_data
    matcher = HungarianMatcher()
    indices_single = matcher({"pred_logits": logits, "pred_boxes": boxes}, targets)
    indices_batched = matcher(
        {
            "pred_logits": logits.repeat(2, 1, 1),
            "pred_boxes": boxes.repeat(2, 1, 1),
        },
        targets * 2,
    )
    assert indices_torch2python(indices_single) == indices_torch2python(
        [indices_batched[0]]
    )
    assert indices_torch2python(indices_single) == indices_torch2python(
        [indices_batched[1]]
    )


@pytest.mark.parametrize(
    "target_counts,empty_idx",
    [
        ([15, 0], 1),  # mixed: first has targets, second is empty
        ([0, 0], None),  # all empty
    ],
)
def test_hungarian_empty_targets(matcher_data, target_counts, empty_idx):
    logits, boxes, _ = matcher_data
    n_classes = logits.shape[-1] - 1
    targets = []
    for count in target_counts:
        labels = torch.randint(high=n_classes, size=(count,))
        bboxes = torch.rand(count, 4)
        targets.append({"labels": labels, "boxes": bboxes})

    matcher = HungarianMatcher()
    indices = matcher(
        {
            "pred_logits": logits.repeat(len(target_counts), 1, 1),
            "pred_boxes": boxes.repeat(len(target_counts), 1, 1),
        },
        targets,
    )
    if empty_idx is not None:
        assert len(indices[empty_idx][0]) == 0
    else:
        for idx in range(len(target_counts)):
            assert len(indices[idx][0]) == 0
