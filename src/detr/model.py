from __future__ import annotations

import torch

import detr.parameters as parameters
from detr._types import Model, ModelMeta, ModelType
from detr.models.backbone import build_backbone
from detr.models.detr import DETR, PostProcess, SetCriterion
from detr.models.matcher import build_matcher
from detr.models.segmentation import DETRsegm, PostProcessSegm
from detr.models.transformer import build_transformer


def factory(
    model_params: parameters.Model,
    loss_params: parameters.Loss,
    train_params: parameters.Train,
    categories: list[str],
) -> Model:
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
        categories=categories,
        model_type=model_params.model_type,
        subtype=model_params.backbone,
        model_params=model_params,
        loss_params=loss_params,
        train_params=train_params,
    )

    return Model(
        ai=model,
        criterion=criterion,
        postprocessors=postprocessors,
        meta=meta,
    )
