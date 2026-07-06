from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

import detr
import detr.parameters as parameters
from detr._types import ModelMeta, ModelType
from detr.models.backbone import build_backbone
from detr.models.detr import DETR, PostProcess, SetCriterion
from detr.models.matcher import build_matcher
from detr.models.segmentation import DETRsegm, PostProcessSegm
from detr.models.transformer import build_transformer


@dataclass
class Bundle:
    """Wraps a trained DETR model together with its metadata and training logs.

    Fields
    ------
    ai_model      : the ``nn.Module`` (moved to *device* in ``__post_init__``).
    criterion     : ``SetCriterion`` used during training / evaluation.
    postprocessors: dict mapping output type (``"bbox"``, ``"segm"``) to postprocessor modules.
    model_params  : ``parameters.Model`` that describes the architecture.
    loss_params   : ``parameters.Loss`` used to build the criterion.
    train_params  : ``parameters.Train`` used during training.
    name          : human-readable name for this run / experiment.
    meta          : ``ModelMeta`` with categories, model/subtype, train_info, transforms, source.
    """

    ai_model: nn.Module
    criterion: nn.Module
    postprocessors: dict[str, nn.Module]
    model_params: parameters.Model
    loss_params: parameters.Loss
    train_params: parameters.Train
    name: str = ""
    meta: ModelMeta = field(default_factory=ModelMeta)

    def __post_init__(self):
        self.name = self.name or f"detr_{self.meta.subtype.value}"

        n_parameters = sum(
            p.numel() for p in self.ai_model.parameters() if p.requires_grad
        )
        logger.info(f"Loaded model with {n_parameters / 1e6:.2f} M parameters")

    def set_device(self, device: str) -> None:
        self.ai_model = self.ai_model.to(device)
        self.criterion = self.criterion.to(device)

    def export(self, file: Path) -> None:
        """Save weights + architecture params to *file*.pth and logs to *file*.csv.

        Enough information is stored to fully reconstruct the bundle via
        ``Bundle.load_from_file()``.
        """
        if file.exists():
            raise FileExistsError(f"File {file} already exists!")

        logger.info(f"Exporting bundle to {file}")
        self.meta.train_info["params"] = asdict(self.train_params)
        torch.save(
            {
                "state_dict": self.ai_model.state_dict(),
                "model_params": asdict(self.model_params),
                "loss_params": asdict(self.loss_params),
                "train_params": asdict(self.train_params),
                "name": self.name,
                "meta": {
                    "categories": self.meta.categories,
                    "model_type": self.meta.model_type,
                    "subtype": self.meta.subtype,
                    "train_info": self.meta.train_info,
                    "transforms": self.meta.transforms,
                    "source": self.meta.source,
                },
            },
            Path(file).with_suffix(".pth"),
        )

        pd.DataFrame(self.meta.train_info["logs"]).to_csv(
            Path(file).with_suffix(".csv"), index=False
        )


def factory(
    model_params: parameters.Model,
    loss_params: parameters.Loss,
    train_params: parameters.Train,
    categories: list[str],
) -> Bundle:
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = len(categories)

    backbone = build_backbone(model_params, train_params)
    transformer = build_transformer(model_params)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=model_params.num_queries,
        aux_loss=model_params.aux_loss,
    )
    if model_params.model_type is ModelType.DETR_SEGM:
        model = DETRsegm(model, freeze_detr=model_params.frozen)
    matcher = build_matcher(loss_params)
    weight_dict = {"loss_ce": 1, "loss_bbox": loss_params.bbox_loss_coef}
    weight_dict["loss_giou"] = loss_params.giou_loss_coef
    if model_params.model_type is ModelType.DETR_SEGM:
        weight_dict["loss_mask"] = loss_params.mask_loss_coef
        weight_dict["loss_dice"] = loss_params.dice_loss_coef
    # TODO this is a hack
    if model_params.aux_loss:
        aux_weight_dict = {}
        for i in range(model_params.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if model_params.model_type is ModelType.DETR_SEGM:
        losses += ["masks"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=loss_params.eos_coef,
        losses=losses,
    )
    postprocessors = {"bbox": PostProcess()}
    if model_params.model_type is ModelType.DETR_SEGM:
        postprocessors["segm"] = PostProcessSegm()

    meta = ModelMeta(
        model_type=model_params.model_type,
        subtype=model_params.backbone,
        categories=categories,
    )

    return Bundle(
        ai_model=model,
        criterion=criterion,
        postprocessors=postprocessors,
        model_params=model_params,
        loss_params=loss_params,
        train_params=train_params,
        meta=meta,
    )


def load_from_file(
    file: Path,
    device: str,
    categories: list[str],
) -> Bundle:
    """Reconstruct a ``Bundle`` from a ``.pth`` file written by ``export()``.

    Parameters
    ----------
    file:
        Path to the ``.pth`` file (the ``.pth`` suffix may be omitted).
    device:
        Override the device stored in the checkpoint. Defaults to the
        value recorded at export time, or auto-detects CUDA when absent.
    """
    logger.info(f"Loading model from {file}")
    model_data = torch.load(file, map_location=device, weights_only=False)

    model_params: parameters.Model = model_data.get("model_params", parameters.Model())
    loss_params: parameters.Loss = model_data.get("loss_params", parameters.Loss())
    train_params: parameters.Train = model_data.get("train_params", parameters.Train())

    bundle = factory(model_params, loss_params, train_params, categories)

    state_dict = model_data.get("state_dict") or model_data.get("model")
    if state_dict is None:
        raise KeyError(f"Checkpoint {file} has neither 'state_dict' nor 'model' key")

    model_state = bundle.ai_model.state_dict()
    incompatible = {
        k for k, v in state_dict.items()
        if k in model_state and v.shape != model_state[k].shape
    }
    if incompatible:
        logger.warning(
            f"Skipping incompatible keys (shape mismatch): {incompatible}"
        )
    state_dict = {k: v for k, v in state_dict.items() if k not in incompatible}

    missing, unexpected = bundle.ai_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            f"load_state_dict (strict=False) — missing: {missing}, unexpected: {unexpected}"
        )

    meta_data = model_data.get("meta", {})
    bundle.meta.source = meta_data.get("source", str(file))
    bundle.meta.transforms = meta_data.get(
        "transforms", model_data.get("transforms", detr.transforms.default())
    )
    bundle.meta.categories = meta_data.get(
        "categories", model_data.get("cats", {})
    )
    bundle.meta.train_info = meta_data.get(
        "train_info", {"params": {}, "logs": []}
    )
    return bundle


def filename(dir_output: Path, name: str) -> Path:
    """Add a suffix wit an index if the output folder already exists."""
    suffix = "pth"
    file_new = dir_output / f"{name}_00.{suffix}"
    for idx in range(1, 99, 1):
        if not file_new.exists():
            logger.info(f"Output file: {file_new}")
            return file_new
        file_new = dir_output / f"{name}_{idx:02d}.{suffix}"

    raise ValueError("Unable to find a non-existing output file!")
