from __future__ import annotations

import argparse
import types
from dataclasses import dataclass, fields, field
from typing import Union, get_args, get_origin, get_type_hints


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """Auto-generate argparse arguments from a dataclass's fields."""
    hints = get_type_hints(cls)
    for f in fields(cls):
        arg_name = f"--{f.name}"
        ftype = hints[f.name]

        # Get help text from field metadata if available
        help_text = f.metadata.get("help", None)

        # Resolve Optional / Union[X, None] / X | None to X
        origin = get_origin(ftype)
        if origin is Union or origin is types.UnionType:
            type_args = [a for a in get_args(ftype) if a is not type(None)]
            ftype = type_args[0] if type_args else str

        if ftype is bool:
            if f.default is False:
                parser.add_argument(arg_name, action="store_true", default=f.default, help=help_text)
            else:
                # For bools that default to True, add a --no-<name> flag
                parser.add_argument(
                    f"--no-{f.name}",
                    dest=f.name,
                    action="store_false",
                    default=f.default,
                    help=help_text,
                )
        else:
            parser.add_argument(arg_name, type=ftype, default=f.default, help=help_text)


@dataclass
class TrainParameters:
    lr: float = field(default=1e-4, metadata={"help": "Learning rate for the transformer"})
    lr_backbone: float = field(default=1e-5, metadata={"help": "Learning rate for the backbone network"})
    batch_size: int = field(default=2, metadata={"help": "Batch size for training"})
    weight_decay: float = field(default=1e-4, metadata={"help": "Weight decay for optimizer"})
    epochs: int = field(default=2, metadata={"help": "Number of training epochs"})
    lr_drop: int = field(default=40, metadata={"help": "Epoch interval to drop learning rate"})
    clip_max_norm: float = field(default=0.1, metadata={"help": "Maximum norm for gradient clipping"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    num_workers: int = field(default=2, metadata={"help": "Number of data loader workers"})

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class ModelParameters:
    backbone: str = field(default="resnet50", metadata={"help": "Name of the convolutional backbone to use"})
    dilation: bool = field(default=False, metadata={"help": "If true, we replace stride with dilation in the last convolutional block (DC5)"})
    position_embedding: str = field(default="sine", metadata={"help": "Type of positional embedding to use on top of the image features"})
    enc_layers: int = field(default=6, metadata={"help": "Number of encoding layers in the transformer"})
    dec_layers: int = field(default=6, metadata={"help": "Number of decoding layers in the transformer"})
    dim_feedforward: int = field(default=2048, metadata={"help": "Intermediate size of the feedforward layers in the transformer blocks"})
    hidden_dim: int = field(default=256, metadata={"help": "Size of the embeddings (dimension of the transformer)"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout applied in the transformer"})
    nheads: int = field(default=8, metadata={"help": "Number of attention heads inside the transformer's attentions"})
    num_queries: int = field(default=100, metadata={"help": "Number of query slots"})
    pre_norm: bool = field(default=False, metadata={"help": "Whether to use pre-normalization in transformer"})
    masks: bool = field(default=False, metadata={"help": "Train segmentation head if the flag is provided"})
    frozen_weights: str | None = field(default=None, metadata={"help": "Path to the pretrained model. If set, only the mask head will be trained"})
    aux_loss: bool = field(default=True, metadata={"help": "Whether to use auxiliary decoding losses (loss at each layer)"})

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ModelParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class LossParameters:
    set_cost_class: float = field(default=1.0, metadata={"help": "Class coefficient in the matching cost"})
    set_cost_bbox: float = field(default=5.0, metadata={"help": "L1 box coefficient in the matching cost"})
    set_cost_giou: float = field(default=2.0, metadata={"help": "giou box coefficient in the matching cost"})
    mask_loss_coef: float = field(default=1.0, metadata={"help": "Mask loss coefficient"})
    dice_loss_coef: float = field(default=1.0, metadata={"help": "Dice loss coefficient for segmentation"})
    bbox_loss_coef: float = field(default=5.0, metadata={"help": "Bounding box L1 loss coefficient"})
    giou_loss_coef: float = field(default=2.0, metadata={"help": "GIoU loss coefficient"})
    eos_coef: float = field(default=0.1, metadata={"help": "Relative classification weight of the no-object class"})

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LossParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class DataParameters:
    dataset_file: str = field(default="coco", metadata={"help": "Name of the dataset file to use"})
    coco_path: str | None = field(default=None, metadata={"help": "Path to COCO dataset directory"})
    coco_panoptic_path: str | None = field(default=None, metadata={"help": "Path to COCO panoptic dataset directory"})
    remove_difficult: bool = field(default=False, metadata={"help": "Whether to remove difficult objects from training"})

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DataParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class RunParameters:
    output_dir: str = field(default="", metadata={"help": "path where to save, empty for no saving"})
    device: str = field(default="cuda", metadata={"help": "Device to run on (cuda or cpu)"})
    resume: str = field(default="", metadata={"help": "Path to checkpoint to resume from"})
    start_epoch: int = field(default=0, metadata={"help": "Starting epoch number"})
    eval: bool = field(default=False, metadata={"help": "Whether to run in evaluation mode only"})

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})
