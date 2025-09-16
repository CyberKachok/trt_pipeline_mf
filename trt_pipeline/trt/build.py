"""TensorRT 10.13 builder utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency in CI containers
    import tensorrt as trt
except Exception as exc:  # pragma: no cover
    raise ImportError("TensorRT is required to use the builder module") from exc

from .calibration import CalibrationSettings, ImageEntropyCalibrator

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProfileSpec:
    """Min/optimal/max shapes for a single input tensor."""

    minimum: Tuple[int, ...]
    optimum: Tuple[int, ...]
    maximum: Tuple[int, ...]


@dataclass(slots=True)
class BuilderSettings:
    """Configuration values exposed through the CLI layer."""

    workspace_gb: float = 2.0
    fp16: bool = True
    int8: bool = False
    strongly_typed: bool = False
    profiles: Mapping[str, ProfileSpec] = field(default_factory=dict)
    enable_refit: bool = False
    sparsity: str = "disable"
    tactic_sources: Sequence[str] | None = None
    timing_cache: str | None = None
    profiling_verbosity: str = "layer_names_only"
    use_dla: bool = False
    dla_core: int = 0
    allow_gpu_fallback: bool = True
    heuristics_enable: bool = True
    calibrator: CalibrationSettings | None = None
    calibration_input_metadata: Mapping[str, Tuple[int, int, int]] | None = None


class TrtBuilder:
    """High level helper around :mod:`tensorrt`'s builder API."""

    def __init__(self, settings: BuilderSettings) -> None:
        self._settings = settings
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._calibrator: ImageEntropyCalibrator | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, onnx_path: str, engine_path: str) -> Path:
        """Compile ``onnx_path`` into ``engine_path`` and return the output path."""

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if self._settings.strongly_typed and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
            network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

        with trt.Builder(self._logger) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, self._logger) as parser:
            self._parse_onnx(parser, onnx_path)
            config = builder.create_builder_config()
            self._configure_memory(config)
            self._configure_precision(builder, config, network)
            self._configure_device(config)
            self._configure_profiles(builder, config, network)
            self._configure_misc_flags(config)
            self._configure_tactics(builder, config)

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT failed to build the engine")

            output = Path(engine_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(serialized)
            LOGGER.info("TensorRT engine saved to %s", output)

            self._persist_timing_cache(config)
            return output.resolve()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_onnx(self, parser: trt.OnnxParser, onnx_path: str) -> None:
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    def _configure_memory(self, config: trt.IBuilderConfig) -> None:
        workspace_bytes = int(self._settings.workspace_gb * (1024 ** 3))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        LOGGER.info("Workspace limit set to %.2f GiB", workspace_bytes / (1024 ** 3))

    def _configure_precision(self, builder: trt.Builder, config: trt.IBuilderConfig, network: trt.INetworkDefinition) -> None:
        if self._settings.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            LOGGER.info("Enabled FP16 mode")

        if self._settings.int8:
            config.set_flag(trt.BuilderFlag.INT8)
            metadata = self._settings.calibration_input_metadata or self._derive_calibration_shapes(network)
            if self._settings.calibrator is None:
                raise ValueError("INT8 mode requested but no CalibrationSettings were provided")
            self._calibrator = ImageEntropyCalibrator(self._settings.calibrator, metadata)
            config.int8_calibrator = self._calibrator
            LOGGER.info("Enabled INT8 mode with calibration cache %s", self._settings.calibrator.cache_file)

    def _configure_device(self, config: trt.IBuilderConfig) -> None:
        if not self._settings.use_dla:
            return
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = self._settings.dla_core
        LOGGER.info("Targeting DLA core %d", self._settings.dla_core)
        if self._settings.allow_gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            LOGGER.info("GPU fallback enabled for unsupported layers")

    def _configure_profiles(self, builder: trt.Builder, config: trt.IBuilderConfig, network: trt.INetworkDefinition) -> None:
        if not self._settings.profiles:
            if any(-1 in tensor.shape for tensor in self._iterate_inputs(network)):
                raise RuntimeError("Dynamic shapes detected but no optimisation profiles were provided")
            return

        profile = builder.create_optimization_profile()
        for name, spec in self._settings.profiles.items():
            profile.set_shape(name, tuple(spec.minimum), tuple(spec.optimum), tuple(spec.maximum))
            LOGGER.info("Profile for %s set to min=%s opt=%s max=%s", name, spec.minimum, spec.optimum, spec.maximum)
        config.add_optimization_profile(profile)

    def _configure_misc_flags(self, config: trt.IBuilderConfig) -> None:
        if self._settings.enable_refit:
            config.set_flag(trt.BuilderFlag.REFIT)
        if self._settings.sparsity.lower() in {"enable", "force"} and hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        if self._settings.sparsity.lower() == "force" and hasattr(trt.BuilderFlag, "FORCE_SPARSITY"):
            config.set_flag(trt.BuilderFlag.FORCE_SPARSITY)
        if not self._settings.heuristics_enable and hasattr(trt.BuilderFlag, "DISABLE_TACTIC_HEURISTIC"):
            config.set_flag(trt.BuilderFlag.DISABLE_TACTIC_HEURISTIC)

        verbosity = self._settings.profiling_verbosity.upper()
        if hasattr(trt.ProfilingVerbosity, verbosity):
            config.profiling_verbosity = getattr(trt.ProfilingVerbosity, verbosity)

    def _configure_tactics(self, builder: trt.Builder, config: trt.IBuilderConfig) -> None:
        if self._settings.timing_cache:
            cache_path = Path(self._settings.timing_cache)
            cache_bytes = cache_path.read_bytes() if cache_path.exists() else b""
            timing_cache = builder.create_timing_cache(cache_bytes)
            config.set_timing_cache(timing_cache, ignore_mismatch=False)
            LOGGER.info("Loaded timing cache from %s", cache_path)

        if not self._settings.tactic_sources:
            return
        mask = 0
        for name in self._settings.tactic_sources:
            attr = name.upper()
            if hasattr(trt.TacticSource, attr):
                mask |= getattr(trt.TacticSource, attr)
        if mask:
            config.set_tactic_sources(mask)
            LOGGER.info("Tactic sources mask set to %s", mask)

    def _persist_timing_cache(self, config: trt.IBuilderConfig) -> None:
        if not self._settings.timing_cache:
            return
        cache = config.get_timing_cache()
        if cache is None:
            return
        cache_path = Path(self._settings.timing_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(cache.serialize())
        LOGGER.info("Timing cache saved to %s", cache_path)

    # Utility methods --------------------------------------------------
    def _derive_calibration_shapes(self, network: trt.INetworkDefinition) -> Dict[str, Tuple[int, int, int]]:
        metadata: Dict[str, Tuple[int, int, int]] = {}
        for tensor in self._iterate_inputs(network):
            shape = tensor.shape
            if len(shape) < 4:
                raise RuntimeError("Calibration currently expects NCHW tensors")
            if any(dim < 0 for dim in shape):
                if tensor.name not in self._settings.profiles:
                    raise RuntimeError(
                        f"Dynamic input {tensor.name} requires an optimisation profile"
                    )
                optimum = self._settings.profiles[tensor.name].optimum
                metadata[tensor.name] = tuple(int(dim) for dim in optimum[1:4])
            else:
                metadata[tensor.name] = tuple(int(dim) for dim in shape[1:4])
        return metadata

    def _iterate_inputs(self, network: trt.INetworkDefinition) -> Iterable[trt.ITensor]:
        for idx in range(network.num_inputs):
            yield network.get_input(idx)


__all__ = [
    "ProfileSpec",
    "BuilderSettings",
    "TrtBuilder",
]
