"""ONNX export utilities for MixFormer trackers.

This module exposes a small, well-typed API that converts the PyTorch
implementation of MixFormerV2 into an ONNX graph ready for TensorRT 10.x.
It follows the recommendations from the TensorRT migration guide: the
graph is exported in explicit batch mode, constant folding is enabled by
default and a configurable opset is exposed to match the target parser
version.  The helper accepts both static and dynamic shapes so the same
code path can export engines for Jetson devices (fixed resolution) and
for datacenter GPUs (dynamic batch/spatial axes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import onnx
import torch

from model.torch_tracker_wrapper import TorchTrackerWrapper

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OnnxInputSpec:
    """Specification of a single ONNX input tensor.

    Attributes:
        name: Tensor name that will be visible in the exported graph.
        shape: A tuple representing the static shape used for dummy inputs.
            Provide the *post-batch* dimensions (e.g. ``(3, 224, 224)``).
        dynamic_axes: Optional mapping of axis index to symbolic name.  This is
            consumed directly by :func:`torch.onnx.export`.
    """

    name: str
    shape: Tuple[int, ...]
    dynamic_axes: MutableMapping[int, str] | None = None

    def resolved_axes(self, batch_size: int) -> Tuple[int, ...]:
        """Return the full shape including the batch dimension."""

        return (batch_size, *self.shape)


@dataclass(slots=True)
class OnnxExportConfig:
    """Configuration container for :func:`export_mixformer_to_onnx`.

    The defaults are tuned for MixFormerV2 on UAV123: a template tensor of
    shape ``(1, 3, 112, 112)`` and a search tensor of ``(1, 3, 224, 224)``.
    Users can supply their own :class:`OnnxInputSpec` objects to describe the
    network inputs explicitly.  Output tensor names are exposed for clarity and
    to make the downstream TensorRT builder independent from implicit indices.
    """

    opset_version: int = 18
    keep_initializers_as_inputs: bool = False
    do_constant_folding: bool = True
    use_external_data_format: bool = False
    input_specs: Sequence[OnnxInputSpec] = field(
        default_factory=lambda: (
            OnnxInputSpec("template", (3, 112, 112)),
            OnnxInputSpec("online_template", (3, 112, 112)),
            OnnxInputSpec("search", (3, 224, 224)),
        )
    )
    output_names: Sequence[str] = ("bbox", "confidence")
    batch_size: int = 1
    dynamic_batch: bool = False
    dynamic_spatial: bool = False
    validate_model: bool = True
    verbose: bool = False


def _create_dynamic_axes(specs: Sequence[OnnxInputSpec],
                         output_names: Sequence[str],
                         enable_batch_axis: bool,
                         enable_spatial_axes: bool) -> Optional[Dict[str, Dict[int, str]]]:
    """Assemble the ``dynamic_axes`` dictionary for torch.onnx.export."""

    dynamic_axes: Dict[str, Dict[int, str]] = {}
    for spec in specs:
        if spec.dynamic_axes:
            dynamic_axes[spec.name] = dict(spec.dynamic_axes)
        else:
            axes: Dict[int, str] = {}
            if enable_batch_axis:
                axes[0] = "batch"
            if enable_spatial_axes:
                axes[2] = "height"
                axes[3] = "width"
            if axes:
                dynamic_axes[spec.name] = axes

    if dynamic_axes:
        for name in output_names:
            axes = dynamic_axes.setdefault(name, {})
            if enable_batch_axis:
                axes.setdefault(0, "batch")
        return dynamic_axes
    if enable_batch_axis or enable_spatial_axes:
        for spec in specs:
            axes: Dict[int, str] = {}
            if enable_batch_axis:
                axes[0] = "batch"
            if enable_spatial_axes:
                axes[2] = "height"
                axes[3] = "width"
            if axes:
                dynamic_axes[spec.name] = axes
        for name in output_names:
            axes = dynamic_axes.setdefault(name, {})
            if enable_batch_axis:
                axes.setdefault(0, "batch")
        return dynamic_axes if dynamic_axes else None
    return None


def _resolve_input_shapes(wrapper: TorchTrackerWrapper,
                          config: OnnxExportConfig) -> Sequence[OnnxInputSpec]:
    """Infer missing shapes from the MixFormer wrapper if necessary."""

    resolved_specs: list[OnnxInputSpec] = []
    lookup: Mapping[str, int] = {
        "template": wrapper.template_size,
        "online_template": wrapper.template_size,
        "search": wrapper.search_size,
    }
    for spec in config.input_specs:
        if spec.shape[0] != 3:
            resolved_specs.append(spec)
            continue
        spatial = spec.shape[1:]
        if spatial and spatial[0] > 0 and spatial[0] == spatial[1]:
            resolved_specs.append(spec)
            continue
        if spec.name not in lookup:
            resolved_specs.append(spec)
            continue
        size = lookup[spec.name]
        resolved_specs.append(OnnxInputSpec(spec.name, (3, size, size), spec.dynamic_axes))
    return tuple(resolved_specs)


def export_mixformer_to_onnx(
    tracker_config: str,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    config: Optional[OnnxExportConfig] = None,
) -> Path:
    """Export MixFormerV2 to ONNX following TensorRT 10.x best practices.

    Args:
        tracker_config: Path to the original MixFormer configuration YAML.
        checkpoint_path: Path to the ``.pth`` or ``.pt`` weight file.
        output_path: Optional destination for the ONNX file.  When omitted the
            ``.onnx`` suffix is appended to ``checkpoint_path``.
        config: Optional :class:`OnnxExportConfig` object describing export
            settings.  If ``None`` the defaults defined in the dataclass are
            used.

    Returns:
        Absolute :class:`pathlib.Path` to the generated ONNX file.
    """

    cfg = config or OnnxExportConfig()
    output = Path(output_path) if output_path else Path(checkpoint_path).with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading MixFormer weights from %s", checkpoint_path)
    wrapper = TorchTrackerWrapper(tracker_config, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.network.to(device)
    wrapper.network.eval()

    resolved_specs = _resolve_input_shapes(wrapper, cfg)

    dummy_inputs = [
        torch.randn(spec.resolved_axes(cfg.batch_size), dtype=torch.float32, device=device)
        for spec in resolved_specs
    ]

    dynamic_axes = _create_dynamic_axes(
        resolved_specs,
        tuple(cfg.output_names),
        enable_batch_axis=cfg.dynamic_batch,
        enable_spatial_axes=cfg.dynamic_spatial,
    )

    LOGGER.info("Exporting ONNX graph to %s", output)
    with torch.no_grad():
        torch.onnx.export(
            wrapper.network,
            tuple(dummy_inputs),
            str(output),
            opset_version=cfg.opset_version,
            do_constant_folding=cfg.do_constant_folding,
            keep_initializers_as_inputs=cfg.keep_initializers_as_inputs,
            input_names=[spec.name for spec in resolved_specs],
            output_names=list(cfg.output_names),
            dynamic_axes=dynamic_axes,
            use_external_data_format=cfg.use_external_data_format,
            verbose=cfg.verbose,
        )

    if cfg.validate_model:
        LOGGER.info("Validating exported ONNX model")
        model = onnx.load(str(output))
        onnx.checker.check_model(model)

    LOGGER.info("ONNX export complete: %%s", output)
    return output.resolve()


__all__ = [
    "OnnxInputSpec",
    "OnnxExportConfig",
    "export_mixformer_to_onnx",
]
