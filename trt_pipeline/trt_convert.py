"""Legacy CLI wrapper around :mod:`trt.build`."""

from __future__ import annotations

import argparse
from pathlib import Path

from trt.build import BuilderSettings, TrtBuilder
from trt.calibration import CalibrationSettings


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    calib_dir: str | None = None,
    workspace: int = 1 << 31,
    calib_cache: str | None = None,
    calib_max_samples: int | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX file using the new builder stack."""

    settings = BuilderSettings(
        workspace_gb=workspace / (1024 ** 3),
        fp16=fp16,
        int8=int8,
    )

    if int8:
        if calib_dir is None:
            raise ValueError("INT8 mode requires a calibration directory")
        cache = calib_cache or (engine_path + ".calib")
        settings.calibrator = CalibrationSettings(
            data_dir=calib_dir,
            cache_file=cache,
            max_samples=calib_max_samples,
        )

    builder = TrtBuilder(settings)
    return builder.build(onnx_path, engine_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated TensorRT conversion helper")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision")
    parser.add_argument("--calib_dir", help="Directory with calibration images")
    parser.add_argument("--calib_cache", help="Calibration cache location")
    parser.add_argument("--calib_max_samples", type=int, help="Limit number of calibration samples")
    parser.add_argument("--workspace", type=int, default=1 << 31, help="Workspace size in bytes")
    args = parser.parse_args()

    result = build_engine(
        args.onnx,
        args.engine,
        fp16=args.fp16,
        int8=args.int8,
        calib_dir=args.calib_dir,
        workspace=args.workspace,
        calib_cache=args.calib_cache,
        calib_max_samples=args.calib_max_samples,
    )
    print(result)


if __name__ == "__main__":
    main()
