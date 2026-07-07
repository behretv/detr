"""
Created on 2026-07-06
Copyright (c) 2026 Munich University of Applied Sciences
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch.nn as nn
from loguru import logger

from . import transforms

if TYPE_CHECKING:
    import detr.parameters as parameters

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
    model_params: parameters.Model
    loss_params: parameters.Loss
    train_params: parameters.Train
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


@dataclass
class Model:
    """Wraps a trained DETR model together with its metadata and training logs.

    Fields
    ------
    ai            : the ``nn.Module`` (moved to *device* in ``__post_init__``).
    criterion     : ``SetCriterion`` used during training / evaluation.
    postprocessors: dict mapping output type (``"bbox"``, ``"segm"``) to postprocessor modules.
    name          : human-readable name for this run / experiment.
    meta          : ``ModelMeta`` with categories, model/subtype, params, train_info, transforms, source.
    """

    meta: ModelMeta
    ai: nn.Module

    criterion: nn.Module
    postprocessors: dict[str, nn.Module]
    name: str = ""
    state_dict: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.name = self.name or f"detr_{self.meta.subtype.value}"

        if not self.state_dict:
            self.state_dict = self.ai.state_dict()

        n_parameters = sum(
            p.numel() for p in self.ai.parameters() if p.requires_grad
        )
        logger.info(f"Loaded model with {n_parameters / 1e6:.2f} M parameters")

    def set_device(self, device: str) -> None:
        self.ai = self.ai.to(device)
        self.criterion = self.criterion.to(device)

    def serialize(self) -> dict:
        """Serialize the model data, including the attributes."""

        # Older versions might not support loading ai-model directly
        tmp_dict = self.__dict__.copy()
        del tmp_dict["ai"]

        for k, v in tmp_dict.items():
            if isinstance(v, ModelMeta):
                tmp_dict[k] = v.serialize4pth()

        return tmp_dict