import numpy as np
import torch
from PIL import Image

import detr
from detr.aux import collate_fn
from detr.dataset import CocoDetection
from detr.train import train_one_epoch
from detr.transforms import default as test_transforms
from detr.transforms import train as train_transforms



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


def _make_tv_target(n_boxes: int = 3, h: int = 100, w: int = 100) -> dict:
    """Build a target dict whose boxes/masks are wrapped as tv_tensors."""
    from torchvision import tv_tensors

    t = _make_target(n_boxes)
    t["boxes"] = tv_tensors.BoundingBoxes(
        t["boxes"], format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
    )
    return t


def test_train_transforms():
    img_out, target_out = train_transforms()(
        _make_pil(300, 400), _make_tv_target(h=300, w=400)
    )
    assert isinstance(img_out, torch.Tensor)
    assert "boxes" in target_out


def test_test_transforms():
    img_out, target_out = test_transforms()(
        _make_pil(300, 400), _make_tv_target(h=300, w=400)
    )
    assert isinstance(img_out, torch.Tensor)
    # Boxes must end up normalised (cxcywh in [0, 1]).
    assert (target_out["boxes"] >= 0).all()
    assert (target_out["boxes"] <= 1).all()


# ---------------------------------------------------------------------------
# Training step tests
# ---------------------------------------------------------------------------


def test_forward_pass_shapes(model_bundle, train_loader, device):
    model_bundle.ai_model.eval()
    samples, _ = next(iter(train_loader))
    samples = samples.to(device)
    with torch.no_grad():
        out = model_bundle.ai_model(samples)
    bs = samples.tensors.shape[0]
    assert "pred_logits" in out
    assert "pred_boxes" in out
    assert out["pred_logits"].shape[0] == bs
    assert out["pred_boxes"].shape[0] == bs


def test_loss_is_finite(model_bundle, train_loader, device):
    model_bundle.ai_model.train()
    model_bundle.criterion.train()
    samples, targets = next(iter(train_loader))
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model_bundle.criterion(model_bundle.ai_model(samples), targets)
    for name, val in loss_dict.items():
        assert torch.isfinite(val), f"loss '{name}' is not finite: {val}"


def test_train_one_epoch_runs(model_bundle, train_loader, device):
    optimizer = torch.optim.AdamW(model_bundle.ai_model.parameters(), lr=1e-4)
    stats = train_one_epoch(
        model_bundle.ai_model,
        model_bundle.criterion,
        train_loader,
        optimizer,
        device,
        epoch=0,
        max_norm=0.1,
    )
    assert "loss_bbox" in stats
    assert torch.isfinite(torch.tensor(stats["loss_bbox"])), (
        f"training loss is not finite: {stats['loss_bbox']}"
    )


def test_evaluate_runs(model_bundle, coco_root, device):
    ds = CocoDetection(
        coco_root,
        coco_root / "valid.coco.json",
        transforms=test_transforms(),
        return_masks=False,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=collate_fn, num_workers=0
    )
    outputs = detr.coco.inference(model_bundle, loader, device)
    stats = detr.coco.run_eval(coco_root / "valid.coco.json", outputs, iou_type="bbox")
    assert "ap_mean" in stats
