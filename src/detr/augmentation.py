"""
Created on 2026-07-06
Copyright (c) 2026 Munich University of Applied Sciences
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields


@dataclass
class Augmentation:
    """Parameters controlling the COCO data-augmentation pipeline."""

    hflip_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of horizontal-flipping training images"},
    )
    scales: list[int] = field(
        default_factory=lambda: [
            480,
            512,
            544,
            576,
            608,
            640,
            672,
            704,
            736,
            768,
            800,
        ],
        metadata={"help": "Candidate short-side sizes for the final random resize"},
    )
    max_size: int = field(
        default=1333,
        metadata={"help": "Maximum long-side size enforced during random resize"},
    )
    crop_branch_prob: float = field(
        default=0.5,
        metadata={
            "help": "Probability of taking the resize+crop+resize branch instead of a plain resize"
        },
    )
    pre_crop_scales: list[int] = field(
        default_factory=lambda: [400, 500, 600],
        metadata={"help": "Short-side sizes used to resize before the random crop"},
    )
    crop_min_size: int = field(
        default=384, metadata={"help": "Minimum size for RandomSizeCrop"}
    )
    crop_max_size: int = field(
        default=600, metadata={"help": "Maximum size for RandomSizeCrop"}
    )
    normalize_mean: list[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406],
        metadata={"help": "Per-channel mean used by Normalize"},
    )
    normalize_std: list[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225],
        metadata={"help": "Per-channel std used by Normalize"},
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Augmentation:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})
