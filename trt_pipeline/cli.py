"""Command line entry point for the MixFormer TensorRT pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to use the CLI") from exc

from export.onnx_export import OnnxExportConfig, OnnxInputSpec, export_mixformer_to_onnx
from trt.build import BuilderSettings, ProfileSpec, TrtBuilder
from trt.calibration import CalibrationSettings

LOGGER = logging.getLogger("trt_pipeline.cli")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_pipeline_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def parse_input_specs(model_cfg: Mapping[str, Any]) -> Sequence[OnnxInputSpec]:
    inputs_cfg = model_cfg.get("inputs")
    if not inputs_cfg:
        return OnnxExportConfig().input_specs

    specs: list[OnnxInputSpec] = []
    for key, spec in inputs_cfg.items():
        name = spec.get("name", key)
        shape = tuple(int(x) for x in spec.get("shape", (3, 224, 224)))
        dynamic_axes_cfg = spec.get("dynamic_axes") or None
        dynamic_axes = None
        if dynamic_axes_cfg:
            dynamic_axes = {int(idx): str(sym) for idx, sym in dynamic_axes_cfg.items()}
        specs.append(OnnxInputSpec(name=name, shape=shape, dynamic_axes=dynamic_axes))
    return tuple(specs)


def parse_output_names(model_cfg: Mapping[str, Any]) -> Sequence[str]:
    outputs_cfg = model_cfg.get("outputs")
    if not outputs_cfg:
        return OnnxExportConfig().output_names
    if isinstance(outputs_cfg, Mapping):
        return tuple(str(v) for v in outputs_cfg.values())
    return tuple(str(v) for v in outputs_cfg)


def build_export_config(pipeline_cfg: Mapping[str, Any], args: argparse.Namespace) -> OnnxExportConfig:
    model_cfg = pipeline_cfg.get("model", {})
    onnx_cfg = pipeline_cfg.get("onnx", {})

    input_specs = parse_input_specs(model_cfg)
    output_names = parse_output_names(model_cfg)

    config = OnnxExportConfig(
        opset_version=args.opset or onnx_cfg.get("opset", OnnxExportConfig.opset_version),
        keep_initializers_as_inputs=onnx_cfg.get("keep_initializers", False),
        do_constant_folding=onnx_cfg.get("constant_folding", True),
        use_external_data_format=onnx_cfg.get("use_external_data_format", False),
        input_specs=input_specs,
        output_names=output_names,
        batch_size=onnx_cfg.get("batch_size", OnnxExportConfig.batch_size),
        dynamic_batch=_resolve_cli_bool(args.dynamic_batch, onnx_cfg.get("dynamic_batch", False)),
        dynamic_spatial=_resolve_cli_bool(args.dynamic_spatial, onnx_cfg.get("dynamic_spatial", False)),
        validate_model=onnx_cfg.get("validate", True),
        verbose=args.verbose_onnx,
    )
    return config


def build_builder_settings(pipeline_cfg: Mapping[str, Any], args: argparse.Namespace) -> BuilderSettings:
    trt_cfg = pipeline_cfg.get("trt", {})
    profiles_cfg = trt_cfg.get("profiles", {})

    profiles: Dict[str, ProfileSpec] = {}
    for key, spec in profiles_cfg.items():
        name = spec.get("name", key)
        minimum = tuple(int(x) for x in spec.get("min"))
        optimum = tuple(int(x) for x in spec.get("opt"))
        maximum = tuple(int(x) for x in spec.get("max"))
        profiles[name] = ProfileSpec(minimum=minimum, optimum=optimum, maximum=maximum)

    settings = BuilderSettings(
        workspace_gb=args.workspace_gb or trt_cfg.get("workspace_gb", BuilderSettings.workspace_gb),
        fp16=_resolve_cli_bool(args.fp16, trt_cfg.get("fp16", BuilderSettings.fp16)),
        int8=_resolve_cli_bool(args.int8, trt_cfg.get("int8", BuilderSettings.int8)),
        strongly_typed=_resolve_cli_bool(args.strongly_typed, trt_cfg.get("strongly_typed", BuilderSettings.strongly_typed)),
        profiles=profiles,
        enable_refit=_resolve_cli_bool(args.enable_refit, trt_cfg.get("enable_refit", BuilderSettings.enable_refit)),
        sparsity=args.sparsity or trt_cfg.get("sparsity", BuilderSettings.sparsity),
        tactic_sources=_resolve_tactic_sources(args.tactic_sources, trt_cfg.get("tactic_sources")),
        timing_cache=args.timing_cache or trt_cfg.get("timing_cache"),
        profiling_verbosity=args.profiling_verbosity or trt_cfg.get("profiling_verbosity", BuilderSettings.profiling_verbosity),
        use_dla=_resolve_cli_bool(args.use_dla, trt_cfg.get("use_dla", BuilderSettings.use_dla)),
        dla_core=args.dla_core if args.dla_core is not None else trt_cfg.get("dla_core", BuilderSettings.dla_core),
        allow_gpu_fallback=_resolve_cli_bool(args.allow_gpu_fallback, trt_cfg.get("allow_gpu_fallback", BuilderSettings.allow_gpu_fallback)),
        heuristics_enable=_resolve_cli_bool(args.heuristics_enable, trt_cfg.get("heuristics_enable", BuilderSettings.heuristics_enable)),
    )

    if settings.int8:
        calib_cfg = pipeline_cfg.get("calibration", {})
        data_dir = args.calib_dir or calib_cfg.get("data_dir")
        cache_file = args.calib_cache or calib_cfg.get("cache_file")
        if not data_dir or not cache_file:
            raise ValueError("INT8 mode requires --calib-dir and --calib-cache (or config entries)")
        settings.calibrator = CalibrationSettings(
            data_dir=data_dir,
            cache_file=cache_file,
            batch_size=args.calib_batch_size or calib_cfg.get("batch_size", 1),
            max_samples=args.calib_max_samples or calib_cfg.get("max_samples"),
            mean=_resolve_float_list(args.calib_mean, calib_cfg.get("mean", (0.485, 0.456, 0.406))),
            std=_resolve_float_list(args.calib_std, calib_cfg.get("std", (0.229, 0.224, 0.225))),
        )
        metadata_cfg = calib_cfg.get("input_metadata")
        if metadata_cfg:
            metadata = {key: tuple(int(x) for x in value) for key, value in metadata_cfg.items()}
            settings.calibration_input_metadata = metadata
        if not settings.calibration_input_metadata and profiles:
            metadata = {name: tuple(int(dim) for dim in spec.optimum[1:4]) for name, spec in profiles.items()}
            settings.calibration_input_metadata = metadata

    return settings


def _resolve_cli_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _resolve_tactic_sources(cli_value: str | None, cfg_value: Sequence[str] | None) -> Sequence[str] | None:
    if cli_value:
        return [item.strip() for item in cli_value.split(",") if item.strip()]
    if cfg_value:
        return tuple(cfg_value)
    return None


def _resolve_float_list(cli_value: str | None, default: Iterable[float]) -> Sequence[float]:
    if cli_value is None:
        return tuple(float(x) for x in default)
    return tuple(float(x) for x in cli_value.split(","))


def handle_export(args: argparse.Namespace) -> None:
    pipeline_cfg = load_pipeline_config(args.pipeline_config)
    export_config = build_export_config(pipeline_cfg, args)
    output = export_mixformer_to_onnx(
        tracker_config=args.tracker_config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config=export_config,
    )
    print(output)


def handle_build(args: argparse.Namespace) -> None:
    pipeline_cfg = load_pipeline_config(args.pipeline_config)
    settings = build_builder_settings(pipeline_cfg, args)
    builder = TrtBuilder(settings)
    engine = builder.build(args.onnx, args.engine)
    print(engine)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MixFormer TensorRT pipeline")
    parser.add_argument("--pipeline-config", help="YAML file with pipeline defaults")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-onnx", help="Export PyTorch checkpoint to ONNX")
    export_parser.add_argument("--tracker-config", required=True, help="Path to MixFormer config YAML")
    export_parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pth/.pt)")
    export_parser.add_argument("--output", help="Destination ONNX path")
    export_parser.add_argument("--opset", type=int, help="ONNX opset version")
    export_parser.add_argument("--dynamic-batch", dest="dynamic_batch", action="store_true")
    export_parser.add_argument("--static-batch", dest="dynamic_batch", action="store_false")
    export_parser.add_argument("--dynamic-spatial", dest="dynamic_spatial", action="store_true")
    export_parser.add_argument("--static-spatial", dest="dynamic_spatial", action="store_false")
    export_parser.add_argument("--verbose-onnx", action="store_true", help="Verbose torch.onnx.export output")
    export_parser.set_defaults(dynamic_batch=None, dynamic_spatial=None)
    export_parser.set_defaults(func=handle_export)

    build_parser = subparsers.add_parser("build-trt", help="Compile an ONNX model into a TensorRT engine")
    build_parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    build_parser.add_argument("--engine", required=True, help="Destination for TensorRT engine")
    build_parser.add_argument("--workspace-gb", type=float, help="Workspace memory limit in GiB")
    build_parser.add_argument("--fp16", dest="fp16", action="store_true")
    build_parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    build_parser.add_argument("--int8", dest="int8", action="store_true")
    build_parser.add_argument("--no-int8", dest="int8", action="store_false")
    build_parser.add_argument("--strongly-typed", dest="strongly_typed", action="store_true")
    build_parser.add_argument("--no-strongly-typed", dest="strongly_typed", action="store_false")
    build_parser.add_argument("--enable-refit", dest="enable_refit", action="store_true")
    build_parser.add_argument("--disable-refit", dest="enable_refit", action="store_false")
    build_parser.add_argument("--sparsity", choices=["disable", "enable", "force"], help="Weight sparsity mode")
    build_parser.add_argument("--tactic-sources", help="Comma separated TensorRT tactic sources")
    build_parser.add_argument("--timing-cache", help="Path to timing cache file")
    build_parser.add_argument("--profiling-verbosity", choices=[
        "layer_names_only",
        "detailed",
        "default",
    ], help="TensorRT profiling verbosity")
    build_parser.add_argument("--use-dla", dest="use_dla", action="store_true")
    build_parser.add_argument("--gpu", dest="use_dla", action="store_false")
    build_parser.add_argument("--dla-core", type=int, help="Index of DLA core to target")
    build_parser.add_argument("--allow-gpu-fallback", dest="allow_gpu_fallback", action="store_true")
    build_parser.add_argument("--disallow-gpu-fallback", dest="allow_gpu_fallback", action="store_false")
    build_parser.add_argument("--enable-heuristics", dest="heuristics_enable", action="store_true")
    build_parser.add_argument("--disable-heuristics", dest="heuristics_enable", action="store_false")
    build_parser.add_argument("--calib-dir", help="Directory with calibration images or .npy tensors")
    build_parser.add_argument("--calib-cache", help="INT8 calibration cache file")
    build_parser.add_argument("--calib-max-samples", type=int, help="Limit number of calibration samples")
    build_parser.add_argument("--calib-batch-size", type=int, help="Batch size for calibration")
    build_parser.add_argument("--calib-mean", help="Comma separated mean values (R,G,B)")
    build_parser.add_argument("--calib-std", help="Comma separated std values (R,G,B)")
    build_parser.set_defaults(fp16=None, int8=None, strongly_typed=None, enable_refit=None, use_dla=None,
                              allow_gpu_fallback=None, heuristics_enable=None)
    build_parser.set_defaults(func=handle_build)

    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        args.func(args)
        return 0
    except Exception as exc:  # pragma: no cover - CLI convenience
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
