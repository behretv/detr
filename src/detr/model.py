from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

import detr.parameters as parameters


def default_transforms() -> list[v2.Transform]:
    """Standard inference / base transforms.

    The short-side ``v2.Resize`` matches the validation pipeline of
    :func:`detr.transforms.make_coco_transforms` and keeps the DETR encoder's
    attention map within a tractable memory budget; without it, native-resolution
    images quickly blow up GPU memory.
    """
    return [
        v2.Resize(800, max_size=1333, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


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
    source        : origin of the weights (e.g. a file path or URL).
    transforms    : inference / validation transforms applied to inputs.
    cats          : mapping from COCO category id → category name.
    device        : ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.
    with_augmentation : whether training-time augmentation is enabled.
    logs          : list of per-epoch stat dicts appended during training.
    """

    ai_model: nn.Module
    criterion: nn.Module
    postprocessors: dict[str, nn.Module]
    model_params: parameters.Model
    loss_params: parameters.Loss
    train_params: parameters.Train
    name: str
    source: str
    transforms: list[v2.Transform]
    cats: dict[int, str]
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    with_augmentation: bool = False
    logs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        model_params: parameters.Model | None = None,
        loss_params: parameters.Loss | None = None,
        train_params: parameters.Train | None = None,
        run_params: parameters.Run | None = None,
        name: str = "",
        source: str = "",
        transforms: list[v2.Transform] | None = None,
        cats: dict[int, str] | None = None,
    ) -> Bundle:
        """Construct a ``Bundle`` from parameter dataclasses.

        All parameter arguments default to their respective dataclass defaults
        when omitted.
        """
        from detr.models import build  # local import to avoid circular deps

        model_params = model_params or parameters.Model()
        loss_params = loss_params or parameters.Loss()
        train_params = train_params or parameters.Train()
        run_params = run_params or parameters.Run()

        model, criterion, postprocessors = build(
            model_params, loss_params, train_params, run_params
        )
        return cls(
            ai_model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            model_params=model_params,
            loss_params=loss_params,
            train_params=train_params,
            name=name,
            source=source,
            transforms=transforms or [],
            cats=cats or {},
            device=run_params.device,
        )

    def __post_init__(self) -> None:
        self.ai_model = self.ai_model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def export(self, file: Path | str) -> None:
        """Save weights + architecture params to *file*.pth and logs to *file*.csv.

        Enough information is stored to fully reconstruct the bundle via
        ``Bundle.load_from_file()``.
        """
        torch.save(
            {
                "state_dict": self.ai_model.state_dict(),
                "model_params": self.model_params,
                "loss_params": self.loss_params,
                "train_params": self.train_params,
                "name": self.name,
                "source": self.source,
                "cats": self.cats,
                "device": self.device,
                "with_augmentation": self.with_augmentation,
            },
            Path(file).with_suffix(".pth"),
        )

        pd.DataFrame(self.logs).to_csv(Path(file).with_suffix(".csv"), index=False)

    @classmethod
    def load_from_file(
        cls,
        file: Path | str,
        device: str | None = None,
        transforms: list[v2.Transform] | None = None,
    ) -> Bundle:
        """Reconstruct a ``Bundle`` from a ``.pth`` file written by ``export()``.

        Parameters
        ----------
        file:
            Path to the ``.pth`` file (the ``.pth`` suffix may be omitted).
        device:
            Override the device stored in the checkpoint. Defaults to the
            value recorded at export time, or auto-detects CUDA when absent.
        transforms:
            Transforms to attach to the bundle. Defaults to an empty list if
            not provided (they are not stored in the checkpoint).
        """
        from detr.models import build  # local import to avoid circular deps

        file = Path(file).with_suffix(".pth")
        checkpoint = torch.load(file, map_location="cpu", weights_only=False)

        resolved_device = device or checkpoint.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        run_params = parameters.Run(device=resolved_device)

        model_params: parameters.Model = (
            checkpoint.get("model_params") or parameters.Model()
        )
        loss_params: parameters.Loss = (
            checkpoint.get("loss_params") or parameters.Loss()
        )
        train_params: parameters.Train = (
            checkpoint.get("train_params") or parameters.Train()
        )

        model, criterion, postprocessors = build(
            model_params, loss_params, train_params, run_params
        )

        state_dict = checkpoint.get("state_dict") or checkpoint.get("model")
        if state_dict is None:
            raise KeyError(
                f"Checkpoint {file} has neither 'state_dict' nor 'model' key"
            )
        model.load_state_dict(state_dict)

        logs: list[dict[str, Any]] = []
        csv_file = file.with_suffix(".csv")
        if csv_file.exists():
            logs = pd.read_csv(csv_file).to_dict(orient="records")

        return cls(
            ai_model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            model_params=model_params,
            loss_params=loss_params,
            train_params=train_params,
            name=checkpoint.get("name", ""),
            source=str(file),
            transforms=transforms or [],
            cats=checkpoint.get("cats", {}),
            device=resolved_device,
            with_augmentation=checkpoint.get("with_augmentation", False),
            logs=logs,
        )


def load_from_file(
    file: Path | str,
    device: str | None = None,
    transforms: list[v2.Transform] | None = None,
) -> Bundle:
    """Module-level convenience wrapper around :meth:`Bundle.load_from_file`.

    When *transforms* is omitted, the returned bundle is prepopulated with the
    standard image-only base pipeline (:func:`default_transforms`), so callers
    can simply append augmentation transforms on top.
    """
    if transforms is None:
        transforms = default_transforms()
    return Bundle.load_from_file(file, device=device, transforms=transforms)
