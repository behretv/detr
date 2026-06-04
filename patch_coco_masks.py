#!/usr/bin/env python3
"""Patch COCO annotations by adding ellipse segmentations where only bboxes exist."""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd


def _ellipse_poly(bbox: list[float], n_points: int) -> list[float]:
    x, y, w, h = bbox
    cx, cy = int(x + w / 2), int(y + h / 2)
    ax, ay = int(w / 2), int(h / 2)
    pts = cv2.ellipse2Poly((cx, cy), (ax, ay), 0, 0, 360, 360 // n_points)
    return pts.reshape(-1).tolist()


def patch_coco(input_path: Path, output_path: Path, n_points: int) -> None:
    data = json.loads(input_path.read_text())
    df = pd.DataFrame(data.get("annotations", []))
    if df.empty:
        print("No annotations found.")
        return

    seg_col = "segmentation"
    if seg_col not in df.columns:
        df[seg_col] = None

    mask = df[seg_col].apply(
        lambda s: s is None or (isinstance(s, list) and len(s) == 0)
    )
    df.loc[mask, seg_col] = df.loc[mask, "bbox"].apply(
        lambda b: [_ellipse_poly(b, n_points)]
    )

    data["annotations"] = df.to_dict(orient="records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data))
    print(f"Patched {mask.sum()} annotations → {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add ellipse masks to COCO annotations.")
    p.add_argument("--input", type=Path, help="Input COCO JSON file")
    p.add_argument("--output", type=Path, help="Output COCO JSON file")
    p.add_argument(
        "--n-points", type=int, default=96, help="Ellipse perimeter resolution"
    )
    args = p.parse_args()
    patch_coco(args.input, args.output, args.n_points)
