import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import detr.parameters as parameters
from detr.dataset import CocoDetection
from detr.evaluate import evaluate
from detr.misc import collate_fn
from detr.model import Bundle
from detr.train import train_one_epoch
from detr.transforms import make_coco_transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coco_json(tmp_dir: Path, n_images: int = 4) -> dict:
    """Write tiny JPEG images to *tmp_dir* and return a minimal COCO dict."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n_images + 1):
        w, h = 128, 96
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        fname = f"{i:012d}.jpg"
        Image.fromarray(arr).save(tmp_dir / fname)
        images.append({"id": i, "file_name": fname, "width": w, "height": h})
        for x0, y0, bw, bh in [(10, 10, 30, 20), (50, 40, 25, 25)]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x0, y0, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "ball", "supercategory": "object"}],
    }


def _make_target(n_boxes: int = 3) -> dict:
    boxes = torch.rand(n_boxes, 4)
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:].abs().clamp(min=0.05)
    boxes[:, 0::2].clamp_(0, 1)
    boxes[:, 1::2].clamp_(0, 1)
    boxes = boxes * 100
    return {
        "boxes": boxes,
        "labels": torch.randint(0, 5, (n_boxes,)),
        "area": torch.rand(n_boxes) * 500 + 100,
        "iscrowd": torch.zeros(n_boxes, dtype=torch.int64),
        "size": torch.tensor([100, 100]),
        "orig_size": torch.tensor([100, 100]),
    }


def _make_pil(h: int = 100, w: int = 100) -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def _small_model_params() -> parameters.Model:
    return parameters.Model(
        backbone="resnet50",
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=64,
        hidden_dim=32,
        nheads=2,
        num_queries=10,
        aux_loss=False,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def coco_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco")
    coco_json = _make_coco_json(root)
    for split_file in ("train.coco.json", "valid.coco.json"):
        (root / split_file).write_text(json.dumps(coco_json))
    return root


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model_bundle(device):
    model_params = _small_model_params()
    bundle = Bundle.build(
        model_params=model_params,
        loss_params=parameters.Loss(),
        train_params=parameters.Train(),
        run_params=parameters.Run(device="cpu"),
    )
    return bundle.ai_model, bundle.criterion, bundle.postprocessors


@pytest.fixture(scope="session")
def train_loader(coco_root):
    ds = CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=make_coco_transforms("val"),
        return_masks=False,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=collate_fn, num_workers=0
    )


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


def _make_tv_target(n_boxes: int = 3, h: int = 100, w: int = 100) -> dict:
    """Build a target dict whose boxes/masks are wrapped as tv_tensors."""
    from torchvision import tv_tensors

    t = _make_target(n_boxes)
    t["boxes"] = tv_tensors.BoundingBoxes(
        t["boxes"], format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
    )
    return t


def test_make_coco_transforms_train():
    img_out, target_out = make_coco_transforms("train")(
        _make_pil(300, 400), _make_tv_target(h=300, w=400)
    )
    assert isinstance(img_out, torch.Tensor)
    assert "boxes" in target_out


def test_make_coco_transforms_val():
    img_out, target_out = make_coco_transforms("val")(
        _make_pil(300, 400), _make_tv_target(h=300, w=400)
    )
    assert isinstance(img_out, torch.Tensor)
    # Boxes must end up normalised (cxcywh in [0, 1]).
    assert (target_out["boxes"] >= 0).all()
    assert (target_out["boxes"] <= 1).all()


def test_make_coco_transforms_unknown_raises():
    with pytest.raises(ValueError):
        make_coco_transforms("test")


# ---------------------------------------------------------------------------
# Dataset / DataLoader tests
# ---------------------------------------------------------------------------


def test_dataset_length(coco_root):
    ds = CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=make_coco_transforms("train"),
        return_masks=False,
    )
    assert len(ds) == 4


def test_dataset_item_keys(coco_root):
    ds = CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=make_coco_transforms("val"),
        return_masks=False,
    )
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    for key in ("boxes", "labels", "image_id", "area", "iscrowd", "orig_size", "size"):
        assert key in target, f"missing key: {key}"


def test_dataset_boxes_normalised(coco_root):
    ds = CocoDetection(
        coco_root,
        coco_root / "train.coco.json",
        transforms=make_coco_transforms("val"),
        return_masks=False,
    )
    _, target = ds[0]
    boxes = target["boxes"]
    assert (boxes >= 0).all(), "box coords should be non-negative"
    assert (boxes <= 1).all(), "box coords should be ≤ 1 after normalisation"


def test_build_dataset(coco_root):
    ds = CocoDetection.build(
        "train", parameters.Data(coco_path=str(coco_root)), parameters.Model()
    )
    assert len(ds) > 0


def test_build_dataset_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        CocoDetection.build(
            "train", parameters.Data(coco_path="/nonexistent/path"), parameters.Model()
        )


def test_dataloader_batch(train_loader):
    samples, targets = next(iter(train_loader))
    assert len(targets) == 2
    assert samples.tensors is not None
    assert samples.tensors.shape[0] == 2


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------


def test_forward_pass_shapes(model_bundle, train_loader, device):
    model, _, _ = model_bundle
    model.eval()
    samples, _ = next(iter(train_loader))
    samples = samples.to(device)
    with torch.no_grad():
        out = model(samples)
    bs = samples.tensors.shape[0]
    assert "pred_logits" in out
    assert "pred_boxes" in out
    assert out["pred_logits"].shape[0] == bs
    assert out["pred_boxes"].shape[0] == bs


def test_loss_is_finite(model_bundle, train_loader, device):
    model, criterion, _ = model_bundle
    model.train()
    criterion.train()
    samples, targets = next(iter(train_loader))
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = criterion(model(samples), targets)
    for name, val in loss_dict.items():
        assert torch.isfinite(val), f"loss '{name}' is not finite: {val}"


def test_train_one_epoch_runs(model_bundle, train_loader, device):
    model, criterion, _ = model_bundle
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    stats = train_one_epoch(
        model, criterion, train_loader, optimizer, device, epoch=0, max_norm=0.1
    )
    assert "loss" in stats
    assert torch.isfinite(torch.tensor(stats["loss"])), (
        f"training loss is not finite: {stats['loss']}"
    )


def test_evaluate_runs(model_bundle, coco_root, device):
    model, criterion, postprocessors = model_bundle
    ds = CocoDetection(
        coco_root,
        coco_root / "valid.coco.json",
        transforms=make_coco_transforms("val"),
        return_masks=False,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=collate_fn, num_workers=0
    )
    stats = evaluate(
        model,
        criterion,
        postprocessors,
        loader,
        ds.coco_api(),
        device,
    )
    assert "loss" in stats
