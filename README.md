# DETR — fine-tuning fork

A trimmed fork of [facebookresearch/detr](https://github.com/facebookresearch/detr) focused on **fine-tuning a pretrained DETR checkpoint on a custom COCO-format dataset**. The original distributed-training, panoptic-segmentation, ONNX-export, and Detectron2 code paths have been removed; what remains is a small, dependency-light core built on top of `torchvision.transforms.v2`.

For background on the model, see *End-to-End Object Detection with Transformers* (Carion et al., 2020).

## Installation

```bash
pip install -e .[coco]
```

Core deps: `torch>=2.0`, `torchvision>=0.15`, `scipy`, `cython`, `submitit`.
Optional extras: `pycocotools` (`coco`), `panopticapi` (`panoptic`), `flake8` (`dev`).

## Dataset layout

`train_detr.py` expects a single directory containing two COCO-format annotation files plus the corresponding images:

```
<dataset>/
├── train.coco.json
├── valid.coco.json
└── *.jpg                # images referenced by either JSON
```

Image paths inside the JSON files are resolved relative to the JSON's parent directory.

## Pretrained weights

Drop a DETR checkpoint anywhere on disk — for example the official R50 weights:

```bash
mkdir -p /mnt/data/models/detr
curl -L -o /mnt/data/models/detr/detr-r50-e632da11.pth \
    https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
```

`Bundle.load_from_file` accepts both the original Facebook checkpoint format (`{"model": state_dict}`) and the format produced by this fork's `Bundle.export` (`{"state_dict": ..., "model_params": ..., ...}`).

## Quick start: `train_detr.py`

```bash
python train_detr.py \
    --dataset    /path/to/your/dataset \
    --model      /path/to/detr-r50-e632da11.pth \
    --batch-size 4 \
    --augment \
    --device     cuda \
    --dir-output /path/to/save
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `$DETR_DATA_ROOT/datasets/accurate-balls` | Directory containing `train.coco.json` / `valid.coco.json`. |
| `--model` | `$DETR_DATA_ROOT/models/detr/models/detr-r50-e632da11.pth` | Pretrained checkpoint to fine-tune. **Required.** |
| `--dir-output` | `$DETR_DATA_ROOT/models/torch` | Where to write the trained `<name>.pth` + `<name>.csv`. Pass an empty value to skip saving. |
| `--augment` | off | Enable horizontal flip + multi-scale resize / crop training augmentation. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--batch-size` | `4` | Per-step batch size for both training and validation loaders. |

`DETR_DATA_ROOT` (default `/mnt/data`) is the root used when constructing default paths; override with:

```bash
export DETR_DATA_ROOT=/your/data/root
```

### What the script does

1. Loads the checkpoint into a `Bundle` (model + criterion + post-processors + parameters).
2. Reads category names from `train.coco.json` and attaches them to the bundle as `cats`.
3. Builds a transform pipeline: the model's base `[Resize, ToImage, ToDtype, Normalize]` plus, with `--augment`, `[RandomHorizontalFlip, RandomChoice(<resize | resize+crop+resize>)]`. `FinalizeTargets` (drop empty boxes, recompute `area`/`size`, normalise boxes to cxcywh in `[0, 1]`) is appended automatically.
4. Builds train / val `DataLoader`s over the two COCO splits.
5. Trains for `parameters.Train().epochs` (default **2**) using AdamW with separate LRs for the backbone (`1e-5`) and the rest (`1e-4`), GIoU + L1 + focal loss matching the DETR paper. Validation runs every epoch, including pycocotools AP metrics.
6. On exit, exports `<dir-output>/<bundle.name>.pth` (state dict + parameter dataclasses + cats) and `<dir-output>/<bundle.name>.csv` (per-epoch metrics).

To change training hyperparameters from defaults, edit `train_detr.py:43` to pass an explicit `parameters.Train(...)` — for example:

```python
parameters.Train(batch_size=args.batch_size, epochs=50, lr_drop=40)
```

## Lower-level scripts

Two further scripts wrap `detr.train.run` directly with the full argparse-from-dataclass surface for finer control:

- `train_objdet.py` — full-control object-detection training; every field of `parameters.{Train,Model,Loss,Data,Augmentation,Run}` becomes a CLI flag (e.g. `--lr`, `--epochs`, `--num_queries`, `--scales 480 512 ...`).
- `train_insseg.py` — train the segmentation head on top of a frozen detector. Requires `--masks` and a `--frozen_weights` checkpoint.

## Programmatic API

```python
from detr import Bundle, parameters
from detr.aux import load_dataset
from detr.train import augmentation_transforms, run

bundle = Bundle.load_from_file("detr-r50-e632da11.pth", device="cuda")
train_loader = load_dataset("data/train.coco.json",
                             bundle.transforms + augmentation_transforms(),
                             shuffle=True, batch_size=4)
val_loader   = load_dataset("data/valid.coco.json",
                             bundle.transforms,
                             shuffle=False, batch_size=4)

bundle = run(bundle, train_loader, val_loader,
             params=parameters.Train(epochs=10))
bundle.export("checkpoints/finetuned")          # writes .pth and .csv
```

## Tests

```bash
pytest test/
```

Covers model construction (CPU + TorchScript), the COCO transform pipeline, dataset loading, a one-epoch training step, and COCO evaluation on a synthetic mini-dataset.

