from __future__ import annotations

import argparse
import types
from dataclasses import dataclass, fields
from typing import Union, get_args, get_origin, get_type_hints


def _add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """Auto-generate argparse arguments from a dataclass's fields."""
    hints = get_type_hints(cls)
    for f in fields(cls):
        arg_name = f"--{f.name}"
        ftype = hints[f.name]

        # Resolve Optional / Union[X, None] / X | None to X
        origin = get_origin(ftype)
        if origin is Union or origin is types.UnionType:
            type_args = [a for a in get_args(ftype) if a is not type(None)]
            ftype = type_args[0] if type_args else str

        if ftype is bool:
            if f.default is False:
                parser.add_argument(arg_name, action="store_true", default=f.default)
            else:
                # For bools that default to True, add a --no-<name> flag
                parser.add_argument(
                    f"--no-{f.name}",
                    dest=f.name,
                    action="store_false",
                    default=f.default,
                )
        else:
            parser.add_argument(arg_name, type=ftype, default=f.default)


@dataclass
class TrainParameters:
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    batch_size: int = 2
    weight_decay: float = 1e-4
    epochs: int = 2
    lr_drop: int = 40
    clip_max_norm: float = 0.1
    seed: int = 42
    num_workers: int = 2

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class ModelParameters:
    backbone: str = "resnet50"
    dilation: bool = False
    position_embedding: str = "sine"
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 100
    pre_norm: bool = False
    masks: bool = False
    frozen_weights: str | None = None
    aux_loss: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ModelParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class LossParameters:
    set_cost_class: float = 1.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    mask_loss_coef: float = 1.0
    dice_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> LossParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class DataParameters:
    coco_path: str | None = None
    remove_difficult: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DataParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class RunParameters:
    output_dir: str = ""
    device: str = "cuda"
    resume: str = ""
    start_epoch: int = 0
    eval: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunParameters:
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})
