# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""COCO evaluator.

Adapted from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
with the distributed-sync plumbing dropped and pycocotools' chatty
``evaluate`` replaced by a stdout-suppressed copy.
"""

import contextlib
import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import ops
from tqdm import tqdm

from . import model


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(
                f"iou_types must be a list or tuple, got {type(iou_types).__name__}"
            )
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def accumulate(self):
        img_ids, idx = np.unique(np.array(self.img_ids), return_index=True)
        for iou_type, coco_eval in self.coco_eval.items():
            eval_imgs = np.concatenate(self.eval_imgs[iou_type], axis=2)[..., idx]
            coco_eval.evalImgs = eval_imgs.flatten().tolist()
            coco_eval.params.imgIds = list(img_ids)
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            logger.info(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        logger.warning(
            f"useSegm (deprecated) is not None. Running {p.iouType} evaluation"
        )
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################


def inference(
    model_data: model.Bundle, loader: torch.utils.data.DataLoader, device: str
) -> list:
    """Runs COCO evaluation and prints results to stdout.

    Args:
        model: A PyTorch instance segmentation model.
        data_loader: A PyTorch data loader for the COCO dataset.
    """
    ai_model = model_data.ai_model
    ai_model.to(device)
    ai_model.eval()

    results = []
    for images, targets in tqdm(loader, desc="Evaluating model"):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = ai_model(images)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        batch_results = model_data.postprocessors["bbox"](outputs, orig_target_sizes)

        for out, target in zip(batch_results, targets):
            # Convert boxes from xyxy to xywh format
            boxes = ops.box_convert(out["boxes"], in_fmt="xyxy", out_fmt="xywh")
            for box, label, score in zip(boxes, out["labels"], out["scores"]):
                results.append(
                    {
                        "image_id": int(target["image_id"].item()),
                        "category_id": int(label.item()),
                        "bbox": box.tolist(),
                        "score": float(score.item()),
                    }
                )

    # Add name to results
    df = pd.DataFrame(results)
    df["name"] = df["category_id"].map(model_data.cats)
    results = df.to_dict("records")
    return results


def run_eval(file_holdout: Path, results: list[dict]) -> dict:
    """Runs COCO evaluation and prints results to stdout."""
    coco_gt = COCO(file_holdout)

    if len(results) == 0:
        logger.warning("No results to evaluate.")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Convert stats to dict
    coco_eval.stats = coco_eval.stats.round(3)
    return {
        "ap_mean": coco_eval.stats[0],
        "ap_50": coco_eval.stats[1],
        "ap_75": coco_eval.stats[2],
        "ap_small": coco_eval.stats[3],
        "ap_medium": coco_eval.stats[4],
        "ap_large": coco_eval.stats[5],
        "ar_max_1": coco_eval.stats[6],
        "ar_max_10": coco_eval.stats[7],
        "ar_max_100": coco_eval.stats[8],
        "ar_small": coco_eval.stats[9],
        "ar_medium": coco_eval.stats[10],
        "ar_large": coco_eval.stats[11],
    }
