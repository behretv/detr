"""COCO evaluator."""

from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import ops
from tqdm import tqdm

from . import model


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


def run_eval(file_holdout: Path, results: list[dict], iou_type: str) -> dict:
    """Runs COCO evaluation and prints results to stdout."""
    coco_gt = COCO(file_holdout)

    if len(results) == 0:
        logger.warning("No results to evaluate.")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.params.useCats = 1
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Convert stats to dict
    stats = list(coco_eval.stats.round(3))
    return {
        "ap_mean": stats[0],
        "ap_50": stats[1],
        "ap_75": stats[2],
        "ap_small": stats[3],
        "ap_medium": stats[4],
        "ap_large": stats[5],
        "ar_max_1": stats[6],
        "ar_max_10": stats[7],
        "ar_max_100": stats[8],
        "ar_small": stats[9],
        "ar_medium": stats[10],
        "ar_large": stats[11],
    }
