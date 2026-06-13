"""Matcher model tests."""

import torch

from detr.models.matcher import HungarianMatcher

from ._aux import indices_torch2python


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
    assert indices_torch2python(indices_single) == indices_torch2python(
        [indices_batched[0]]
    )
    assert indices_torch2python(indices_single) == indices_torch2python(
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
