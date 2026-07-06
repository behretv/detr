"""
Created on 2026-07-06
Copyright (c) 2026 Munich University of Applied Sciences
"""

from enum import Enum

class ModelType(Enum):
    """Model types for easier import export."""

    DETR_BBOX = "detr_bbox"
    DETR_SEGM = "detr_segm"

class ModelSubType(Enum):
    """Model subtypes for easier import export."""

    RESNET50 = "resnet50"
    RESNET50_DC5 = "resnet50_dc5"
    RESNET101 = "resnet101"
    RESNET101_DC5 = "resnet101_dc5"
