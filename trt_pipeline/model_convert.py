"""Compatibility wrappers for exporting MixFormer checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from export.onnx_export import OnnxExportConfig, export_mixformer_to_onnx


def convert_to_onnx(cfg_path: str, ckpt_path: str, opset: int = 18, output: str | None = None) -> Path:
    """Export a PyTorch checkpoint to ONNX using :mod:`export.onnx_export`."""

    config = OnnxExportConfig(opset_version=opset)
    return export_mixformer_to_onnx(cfg_path, ckpt_path, output, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated wrapper around export.onnx_export")
    parser.add_argument("--config", required=True, help="Tracker YAML configuration")
    parser.add_argument("--ckpt", required=True, help="PyTorch checkpoint (.pth)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--output", help="Destination .onnx path")
    args = parser.parse_args()

    result = convert_to_onnx(args.config, args.ckpt, opset=args.opset, output=args.output)
    print(result)


if __name__ == "__main__":
    main()
