from dataclasses import dataclass
from typing import Optional


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
    frozen_weights: Optional[str] = None
    aux_loss: bool = True


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


@dataclass
class DataParameters:
    coco_path: Optional[str] = None
    remove_difficult: bool = False


@dataclass
class RunParameters:
    output_dir: str = ""
    device: str = "cuda"
    resume: str = ""
    start_epoch: int = 0
    eval: bool = False
