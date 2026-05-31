"""ONNX export utilities for DETR models.

This module is optional — it is only importable when ``onnxruntime`` is
installed.  Removing it has no effect on training or inference.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from detr.misc import nested_tensor_from_tensor_list

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def export(
    model: nn.Module,
    inputs_list: list,
    output_path: Path | str | None = None,
    do_constant_folding: bool = True,
    dynamic_axes: dict | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset_version: int = 12,
) -> io.BytesIO:
    """Export *model* to ONNX and return the serialised bytes.

    Parameters
    ----------
    model:
        The model to export (will be set to eval mode).
    inputs_list:
        List of sample inputs; the first element is used for tracing.
    output_path:
        Optional file path to write the ``.onnx`` file.  When ``None`` the
        bytes are only returned in-memory.
    """
    model.eval()
    # torch.onnx.export deprecated writing to BytesIO objects, so we go
    # through a temporary file and load the bytes back in-memory.
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        torch.onnx.export(
            model,
            inputs_list[0],
            tmp.name,
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )
        data = Path(tmp.name).read_bytes()
    if output_path is not None:
        Path(output_path).write_bytes(data)
    return io.BytesIO(data)


def validate(
    onnx_io: io.BytesIO,
    inputs,
    outputs,
    tolerate_small_mismatch: bool = False,
) -> None:
    """Assert that ORT outputs match the PyTorch *outputs* for the given *inputs*."""
    if onnxruntime is None:
        raise RuntimeError("onnxruntime is not installed")

    flat_inputs, _ = torch.jit._flatten(inputs)
    flat_outputs, _ = torch.jit._flatten(outputs)

    np_inputs = [
        (t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy())
        for t in flat_inputs
    ]
    np_outputs = [
        (t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy())
        for t in flat_outputs
    ]

    ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
    ort_inputs = {
        ort_session.get_inputs()[i].name: inp for i, inp in enumerate(np_inputs)
    }
    ort_outs = ort_session.run(None, ort_inputs)

    for i, element in enumerate(np_outputs):
        try:
            torch.testing.assert_close(
                torch.as_tensor(ort_outs[i]),
                torch.as_tensor(element),
                rtol=1e-03,
                atol=1e-05,
            )
        except AssertionError as error:
            if tolerate_small_mismatch:
                assert "(0.00%)" in str(error), str(error)
            else:
                raise


def export_and_validate(
    model: nn.Module,
    inputs_list: list,
    tolerate_small_mismatch: bool = False,
    do_constant_folding: bool = True,
    dynamic_axes: dict | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> io.BytesIO:
    """Export model to ONNX and validate against ORT for every entry in *inputs_list*."""
    onnx_io = export(
        model,
        inputs_list,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
    )
    for test_inputs in inputs_list:
        with torch.no_grad():
            if isinstance(test_inputs, (torch.Tensor, list)):
                test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
            test_outputs = model(*test_inputs)
            if isinstance(test_outputs, torch.Tensor):
                test_outputs = (test_outputs,)
        validate(onnx_io, test_inputs, test_outputs, tolerate_small_mismatch)
    return onnx_io
