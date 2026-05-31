from __future__ import annotations

import argparse
import types as _types
from dataclasses import MISSING, dataclass, field, fields
from typing import Union, get_args, get_origin, get_type_hints


def _default_value(f):
    """Resolve a dataclass field's default, calling its factory if necessary."""
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()
    return None


def add_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """Auto-generate argparse arguments from a dataclass's fields."""
    hints = get_type_hints(cls)
    for f in fields(cls):
        arg_name = f"--{f.name}"
        ftype = hints[f.name]
        default = _default_value(f)

        help_text = f.metadata.get("help", None)
        required = f.metadata.get("required", False)

        origin = get_origin(ftype)
        if origin is Union or origin is _types.UnionType:
            type_args = [a for a in get_args(ftype) if a is not type(None)]
            ftype = type_args[0] if type_args else str
            origin = get_origin(ftype)

        if origin is list:
            (elem_type,) = get_args(ftype) or (str,)
            parser.add_argument(
                arg_name,
                type=elem_type,
                nargs="+",
                default=default,
                help=help_text,
                required=required,
            )
        elif ftype is bool:
            if default is False:
                parser.add_argument(
                    arg_name, action="store_true", default=default, help=help_text
                )
            else:
                parser.add_argument(
                    f"--no-{f.name}",
                    dest=f.name,
                    action="store_false",
                    default=default,
                    help=help_text,
                )
        else:
            parser.add_argument(
                arg_name,
                type=ftype,
                default=default,
                help=help_text,
                required=required,
            )


@dataclass
class Train:
    lr: float = field(
        default=1e-4, metadata={"help": "Learning rate for the transformer"}
    )
    lr_backbone: float = field(
        default=1e-5, metadata={"help": "Learning rate for the backbone network"}
    )
    batch_size: int = field(default=2, metadata={"help": "Batch size for training"})
    weight_decay: float = field(
        default=1e-4, metadata={"help": "Weight decay for optimizer"}
    )
    epochs: int = field(default=2, metadata={"help": "Number of training epochs"})
    lr_drop: int = field(
        default=40, metadata={"help": "Epoch interval to drop learning rate"}
    )
    clip_max_norm: float = field(
        default=0.1, metadata={"help": "Maximum norm for gradient clipping"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    num_workers: int = field(
        default=2, metadata={"help": "Number of data loader workers"}
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Train:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class Model:
    backbone: str = field(
        default="resnet50",
        metadata={"help": "Name of the convolutional backbone to use"},
    )
    dilation: bool = field(
        default=False,
        metadata={
            "help": "If true, we replace stride with dilation in the last convolutional block (DC5)"
        },
    )
    position_embedding: str = field(
        default="sine",
        metadata={
            "help": "Type of positional embedding to use on top of the image features"
        },
    )
    enc_layers: int = field(
        default=6, metadata={"help": "Number of encoding layers in the transformer"}
    )
    dec_layers: int = field(
        default=6, metadata={"help": "Number of decoding layers in the transformer"}
    )
    dim_feedforward: int = field(
        default=2048,
        metadata={
            "help": "Intermediate size of the feedforward layers in the transformer blocks"
        },
    )
    hidden_dim: int = field(
        default=256,
        metadata={"help": "Size of the embeddings (dimension of the transformer)"},
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout applied in the transformer"}
    )
    nheads: int = field(
        default=8,
        metadata={
            "help": "Number of attention heads inside the transformer's attentions"
        },
    )
    num_queries: int = field(default=100, metadata={"help": "Number of query slots"})
    pre_norm: bool = field(
        default=False,
        metadata={"help": "Whether to use pre-normalization in transformer"},
    )
    masks: bool = field(
        default=False,
        metadata={"help": "Train segmentation head if the flag is provided"},
    )
    frozen_weights: str | None = field(
        default=None,
        metadata={
            "help": "Path to the pretrained model. If set, only the mask head will be trained"
        },
    )
    aux_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to use auxiliary decoding losses (loss at each layer)"
        },
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Model:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class Loss:
    set_cost_class: float = field(
        default=1.0, metadata={"help": "Class coefficient in the matching cost"}
    )
    set_cost_bbox: float = field(
        default=5.0, metadata={"help": "L1 box coefficient in the matching cost"}
    )
    set_cost_giou: float = field(
        default=2.0, metadata={"help": "giou box coefficient in the matching cost"}
    )
    mask_loss_coef: float = field(
        default=1.0, metadata={"help": "Mask loss coefficient"}
    )
    dice_loss_coef: float = field(
        default=1.0, metadata={"help": "Dice loss coefficient for segmentation"}
    )
    bbox_loss_coef: float = field(
        default=5.0, metadata={"help": "Bounding box L1 loss coefficient"}
    )
    giou_loss_coef: float = field(
        default=2.0, metadata={"help": "GIoU loss coefficient"}
    )
    eos_coef: float = field(
        default=0.1,
        metadata={"help": "Relative classification weight of the no-object class"},
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Loss:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class Data:
    coco_path: str | None = field(
        default=None, metadata={"help": "Path to COCO dataset directory"}
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Data:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class Run:
    output_dir: str = field(
        default="", metadata={"help": "path where to save, empty for no saving"}
    )
    device: str = field(
        default="cuda", metadata={"help": "Device to run on (cuda or cpu)"}
    )
    base_model: str = field(
        default="",
        metadata={
            "help": "Path to the base model checkpoint to load (required)",
            "required": True,
        },
    )
    eval: bool = field(
        default=False, metadata={"help": "Whether to run in evaluation mode only"}
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Run:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class Augmentation:
    """Parameters controlling the COCO data-augmentation pipeline."""

    hflip_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of horizontal-flipping training images"},
    )
    scales: list[int] = field(
        default_factory=lambda: [
            480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800,
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
