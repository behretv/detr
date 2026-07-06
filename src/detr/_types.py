"""
Created on 2026-07-06
Copyright (c) 2026 Munich University of Applied Sciences
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from . import transforms

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


@dataclass
class ModelMeta:
    """Metadata associated with a trained or loaded model."""

    categories: list[str]
    model_type: ModelType
    subtype: ModelSubType
    transforms: list[Any] = transforms.default()
    train_info: dict[str, Any] = field(
        default_factory=lambda: {"params": {}, "logs": []}
    )
    source: str = ""

    def serialize4pth(self) -> dict:
        """Serialize the model meta data, including the attributes."""

        tmp_dict = asdict(self).copy()

        for k, v in tmp_dict.items():
            if isinstance(v, Enum):
                tmp_dict[k] = v.value

        return tmp_dict